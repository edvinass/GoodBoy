"""Session context built across agent loop turns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.memory import load_project_memory
from agent.task_policy import is_verification_command
from agent.types import AgentAction, AgentStep, PlanItem, PlanItemStatus, ToolResult, TurnRecord

_TERMINAL_ACTIONS = frozenset(
    {
        AgentAction.NEED_USER_INPUT,
        AgentAction.TASK_COMPLETE,
        AgentAction.FAILED,
    }
)

_READ_FILE_ACTIONS = frozenset({AgentAction.READ_FILE})

_EDIT_ACTIONS = frozenset(
    {
        AgentAction.STR_REPLACE,
        AgentAction.APPLY_PATCH,
    }
)

# Actions that change a file on disk, making any prior read of that file's
# contents stale. ``move_file`` invalidates both the old and new paths.
_FILE_MUTATING_ACTIONS = frozenset(
    {
        AgentAction.STR_REPLACE,
        AgentAction.APPLY_PATCH,
        AgentAction.DELETE_FILE,
        AgentAction.MOVE_FILE,
    }
)


def _strikethrough(text: str) -> str:
    return "".join(f"{char}\u0336" for char in text)


def format_plan_items(items: list[PlanItem]) -> str:
    """Render plan rows for model context and terminal output."""
    lines: list[str] = []
    for item in items:
        if item.status == PlanItemStatus.DONE:
            mark = "✓"
        elif item.status == PlanItemStatus.CANCELLED:
            mark = "–"
        elif item.status == PlanItemStatus.IN_PROGRESS:
            mark = "→"
        else:
            mark = " "
        label = item.text
        if item.status == PlanItemStatus.CANCELLED:
            label = _strikethrough(label)
        lines.append(f"[{mark}] {item.id}. {label}")
    return "\n".join(lines)


def _plan_step_line(index: int, total: int, text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) > 72:
        cleaned = cleaned[:69] + "..."
    return f'{index}/{total} step "{cleaned}"'


def count_plan_done(items: list[PlanItem]) -> int:
    """Number of plan rows marked done (excludes cancelled)."""
    return sum(1 for item in items if item.status == PlanItemStatus.DONE)


def should_print_plan_progress(
    before: list[PlanItem], after: list[PlanItem]
) -> bool:
    """Whether the terminal should show an updated plan panel."""
    if not after:
        return False
    if not before:
        return count_plan_done(after) == 0
    return count_plan_done(after) > count_plan_done(before)


def format_plan_step_progress(items: list[PlanItem]) -> str | None:
    """One-line plan progress for the Neo spinner (e.g. ``2/5 step "…"``)."""
    active = [item for item in items if item.status != PlanItemStatus.CANCELLED]
    if not active:
        return None
    total = len(active)
    for index, item in enumerate(active, start=1):
        if item.status == PlanItemStatus.IN_PROGRESS:
            return _plan_step_line(index, total, item.text)
    for index, item in enumerate(active, start=1):
        if item.status == PlanItemStatus.PENDING:
            return _plan_step_line(index, total, item.text)
    return None


def plan_items_from_response(raw: str | None) -> list[PlanItem] | None:
    """Last ``update_plan`` in a model response, if any."""
    if not raw:
        return None
    from agent.types import parse_all_agent_steps

    latest: list[PlanItem] | None = None
    for step in parse_all_agent_steps(raw):
        if step.action == AgentAction.UPDATE_PLAN and step.plan_items:
            latest = list(step.plan_items)
    return latest


def effective_plan_items(
    ctx: SessionContext, raw: str | None = None
) -> list[PlanItem]:
    """Plan rows for UI spinners: session context plus same-response updates."""
    items = list(ctx.plan_items)
    from_response = plan_items_from_response(raw)
    if from_response is not None:
        return from_response
    return items


DEFAULT_RECENT_FULL_TURNS = 8
_MAX_COMPLEX_WINDOW = 30
_MIN_FULL_TURNS_UNDER_BUDGET = 2
_SUMMARY_LINE_PREVIEW_CHARS = 200

TurnRenderMode = Literal["full", "summary"]


@dataclass(frozen=True)
class ContextWatermark:
    """Snapshot of context lengths after a successful LLM call.

    Used by the response-chain code path (`previous_response_id`) to send only
    items appended since the last call instead of the full transcript.
    """

    turns: int = 0
    parse_errors: int = 0
    user_replies: int = 0


def effective_recent_full_turns(
    base: int,
    task: str,
    *,
    multiplier: float = 2.0,
    cap: int = _MAX_COMPLEX_WINDOW,
) -> int:
    """Widen the full-turn window for complex tasks (keyword heuristic)."""
    from agent.task_policy import is_complex_task

    if not is_complex_task(task):
        return base
    return min(cap, max(1, int(base * multiplier)))


def _estimated_tokens(text: str) -> int:
    """Rough token count for budget checks (tiktoken when available)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line.rstrip()
    return None


def _last_nonempty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.rstrip()
    return None


def _preview_line(line: str, *, limit: int = _SUMMARY_LINE_PREVIEW_CHARS) -> str:
    """Cap a single-line preview so unsplit output (JSON/CSV dumps) doesn't bloat the summary."""
    if len(line) <= limit:
        return line
    return line[:limit] + " ..."


def _read_file_key(step: AgentStep) -> tuple[str, int | None, int | None]:
    return (step.path or "", step.start_line, step.end_line)


_INF_LINE = 10**12


def _read_range(step: AgentStep) -> tuple[int, int]:
    """Normalised inclusive line range for a read_file step.

    Open-ended bounds collapse to ``1`` / a sentinel "end of file" so any two
    reads can be compared with simple integer overlap checks.
    """
    start = step.start_line if step.start_line is not None else 1
    end = step.end_line if step.end_line is not None else _INF_LINE
    if end < start:
        end = start
    return start, end


def _range_covers(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    """True when ``outer`` fully contains ``inner`` (inclusive)."""
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def _supersedes_by_index(turns: list[TurnRecord]) -> dict[int, int]:
    """Map ``read_file`` turn index → index of the latest turn that supersedes it.

    A later turn supersedes an earlier read when either holds:

    * It is another ``read_file`` of the same path whose line range fully
      contains the earlier range (exact-key duplicates are the trivial case).
    * It mutates the file (``str_replace``, ``apply_patch``, ``delete_file``,
      ``move_file`` source path), making the earlier read's body stale.

    Only the most recent superseding turn is kept so the model sees a single
    pointer to the authoritative version.
    """
    superseded: dict[int, int] = {}
    for inner_index, inner_record in enumerate(turns):
        inner_step = inner_record.step
        if inner_step.action not in _READ_FILE_ACTIONS or not inner_step.path:
            continue
        inner_path = inner_step.path
        inner_range = _read_range(inner_step)
        latest: int | None = None
        for outer_index in range(inner_index + 1, len(turns)):
            outer_step = turns[outer_index].step
            outer_path = outer_step.path
            if not outer_path:
                continue
            if outer_step.action in _READ_FILE_ACTIONS:
                if outer_path != inner_path:
                    continue
                if not _range_covers(_read_range(outer_step), inner_range):
                    continue
                latest = outer_index
                continue
            if outer_step.action in _FILE_MUTATING_ACTIONS:
                if outer_path == inner_path:
                    latest = outer_index
                    continue
                # ``move_file`` also invalidates reads of the destination — a
                # later read against ``dest_path`` would target a freshly
                # moved file whose contents differ from anything seen before.
                if (
                    outer_step.action == AgentAction.MOVE_FILE
                    and outer_step.dest_path == inner_path
                ):
                    latest = outer_index
                    continue
        if latest is not None:
            superseded[inner_index] = latest
    return superseded


def _sticky_turn_indices(turns: list[TurnRecord]) -> set[int]:
    """Turn indices that stay in full mode even when outside the recent window."""
    sticky: set[int] = set()

    for index, record in enumerate(turns):
        if record.step.action in _TERMINAL_ACTIONS:
            sticky.add(index)

    for index in range(len(turns) - 1, -1, -1):
        tr = turns[index].tool_result
        if tr is not None and (tr.exit_code not in (0, None) or tr.timed_out):
            sticky.add(index)
            break

    for index in range(len(turns) - 1, -1, -1):
        record = turns[index]
        if (
            record.step.action == AgentAction.RUN_SHELL
            and record.step.command
            and is_verification_command(record.step.command)
        ):
            tr = record.tool_result
            if tr is not None and tr.exit_code == 0 and not tr.timed_out:
                sticky.add(index)
                break

    known_ids: set[str] = set()
    last_add_index: int | None = None
    for index, record in enumerate(turns):
        if record.step.action != AgentAction.UPDATE_PLAN or not record.step.plan_items:
            continue
        item_ids = {item.id for item in record.step.plan_items}
        if item_ids - known_ids:
            last_add_index = index
        known_ids |= item_ids
    if last_add_index is not None:
        sticky.add(last_add_index)

    return sticky


def _prior_plan_ids_before_turn(turns: list[TurnRecord], turn_index: int) -> set[str]:
    """Plan item ids known before ``turn_index`` (from earlier update_plan turns)."""
    known: set[str] = set()
    for index in range(turn_index):
        record = turns[index]
        if record.step.action == AgentAction.UPDATE_PLAN and record.step.plan_items:
            known |= {item.id for item in record.step.plan_items}
    return known


def _tool_pass_fail(tr: ToolResult) -> str:
    if tr.timed_out:
        return "FAIL"
    if tr.exit_code in (0, None):
        return "PASS"
    return "FAIL"


def _parse_search_top_paths(stdout: str, *, limit: int = 3) -> list[str]:
    """Extract path:line snippets from search_code stdout."""
    hits: list[str] = []
    seen: set[str] = set()
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("("):
            continue
        match = re.match(r"^([^:]+:\d+)", stripped)
        if match:
            loc = match.group(1)
            if loc not in seen:
                seen.add(loc)
                hits.append(loc)
                if len(hits) >= limit:
                    break
    return hits


def _count_list_entries(stdout: str) -> int:
    count = 0
    for line in stdout.splitlines():
        if line.strip() and not line.strip().startswith("("):
            count += 1
    return count


def _edit_line_delta(step: AgentStep) -> int | None:
    if step.action == AgentAction.STR_REPLACE:
        old_lines = (step.old_string or "").count("\n") + (
            1 if step.old_string else 0
        )
        new_lines = (step.new_string or "").count("\n") + (
            1 if step.new_string is not None and step.new_string != "" else 0
        )
        return new_lines - old_lines
    if step.action == AgentAction.APPLY_PATCH and step.patch:
        added = sum(1 for line in step.patch.splitlines() if line.startswith("+"))
        removed = sum(1 for line in step.patch.splitlines() if line.startswith("-"))
        return added - removed
    return None


def _format_read_range(step: AgentStep) -> str:
    start = step.start_line or 1
    end = step.end_line if step.end_line is not None else "end"
    return f"{start}-{end}"


class ConversationExchange(BaseModel):
    """One completed user message and the assistant's final reply."""

    user: str
    assistant: str


class SessionContext(BaseModel):
    """Accumulated state for one user task."""

    user_task: str
    workspace: str | None = None
    conversation_history: list[ConversationExchange] = Field(default_factory=list)
    turns: list[TurnRecord] = Field(default_factory=list)
    user_replies: list[str] = Field(default_factory=list)
    parse_errors: list[str] = Field(default_factory=list)
    active_hosted_tools: list[str] = Field(default_factory=list)
    recent_full_turns: int = DEFAULT_RECENT_FULL_TURNS
    token_budget: int | None = None
    plan_items: list[PlanItem] = Field(default_factory=list)
    working_memory: list[str] = Field(default_factory=list)
    last_edit_turn: int | None = None
    last_verify_turn: int | None = None

    def append_working_memory(self, items: list[str], *, max_items: int) -> None:
        """Append unique memory strings, newest last, capped at max_items."""
        for raw in items:
            text = raw.strip()
            if not text:
                continue
            if len(text) > 500:
                text = text[:500] + "..."
            if text in self.working_memory:
                continue
            self.working_memory.append(text)
        if len(self.working_memory) > max_items:
            self.working_memory = self.working_memory[-max_items:]

    def format_pinned_sections(self) -> str:
        """Durable plan and memory — always included in model input."""
        sections: list[str] = []
        if self.plan_items:
            sections.append("## Active plan\n" + format_plan_items(self.plan_items))
        if self.working_memory:
            mem_lines = ["## Working memory"]
            mem_lines.extend(f"- {line}" for line in self.working_memory)
            sections.append("\n".join(mem_lines))
        if not sections:
            return ""
        return "\n\n".join(sections) + "\n\n"

    def add_turn(self, record: TurnRecord) -> None:
        self.turns.append(record)

    def add_parse_error(self, error: str) -> None:
        self.parse_errors.append(error)

    def add_user_reply(self, reply: str) -> None:
        self.user_replies.append(reply)

    def watermark(self) -> ContextWatermark:
        """Snapshot current append-only counters; pair with to_prompt_delta."""
        return ContextWatermark(
            turns=len(self.turns),
            parse_errors=len(self.parse_errors),
            user_replies=len(self.user_replies),
        )

    def to_prompt_delta(self, since: ContextWatermark) -> str:
        """Render only items appended since ``since`` for stateful chained calls."""
        sections: list[str] = []
        new_turns = self.turns[since.turns:]
        new_errors = self.parse_errors[since.parse_errors:]
        new_replies = self.user_replies[since.user_replies:]

        pinned = self.format_pinned_sections()
        if not (new_turns or new_errors or new_replies):
            body = (
                "## Continue\nNo new turn output since your last response. "
                "Return the next JSON action object."
            )
            return pinned + body if pinned else body

        supersedes = _supersedes_by_index(self.turns)

        if new_turns:
            sections.append("## New turn output")
            for record in new_turns:
                index = self.turns.index(record)
                superseded = _superseded_turn_number(
                    index, supersedes, self.turns
                )
                sections.append(
                    self._format_turn(
                        record,
                        mode="full",
                        turn_index=index,
                        all_turns=self.turns,
                        superseded_by_turn=superseded,
                    )
                )

        if new_replies:
            sections.append(
                "\n## New user clarifications (act on these)"
            )
            for i, reply in enumerate(new_replies, start=since.user_replies + 1):
                sections.append(f"{i}. {reply}")

        if new_errors:
            sections.append("\n## New parse errors (fix your JSON)")
            for err in new_errors:
                sections.append(f"- {err}")

        sections.append(
            "\n## Your turn\n"
            "Return the next JSON action object to continue or finish the task."
        )
        body = "\n".join(sections)
        return pinned + body if pinned else body

    def to_prompt(self) -> str:
        """Serialize context for the model's user message."""
        full_count = max(1, self.recent_full_turns)
        if self.token_budget is not None:
            while True:
                text = self._build_prompt(full_count)
                if _estimated_tokens(text) <= self.token_budget:
                    return text
                if full_count <= _MIN_FULL_TURNS_UNDER_BUDGET:
                    return text
                full_count -= 1
        return self._build_prompt(full_count)

    def _build_prompt(self, full_count: int) -> str:
        sections: list[str] = []
        pinned = self.format_pinned_sections()
        if pinned:
            sections.append(pinned.rstrip())

        if self.conversation_history:
            sections.extend(
                [
                    "## Prior conversation",
                    "Earlier messages in this session (the current task may be a follow-up):",
                ]
            )
            for i, exchange in enumerate(self.conversation_history, start=1):
                sections.append(f"\n### Exchange {i}")
                sections.append(f"User: {exchange.user.strip()}")
                sections.append(f"Assistant: {exchange.assistant.strip()}")

        sections.extend(
            [
                "\n## User task",
                self.user_task.strip(),
            ]
        )

        if self.workspace:
            sections.extend(
                [
                    "\n## Workspace",
                    f"run_shell and run_python use cwd: {self.workspace}",
                ]
            )
            memory = load_project_memory(Path(self.workspace))
            if memory:
                sections.extend(["\n## Project memory", memory])

        if self.active_hosted_tools:
            tools = ", ".join(self.active_hosted_tools)
            sections.extend(
                [
                    "\n## Active API",
                    f"OpenAI hosted tools enabled for subsequent LLM calls: {tools}",
                ]
            )

        if self.turns:
            sticky = _sticky_turn_indices(self.turns)
            supersedes = _supersedes_by_index(self.turns)
            split = max(0, len(self.turns) - full_count)
            older = self.turns[:split]
            recent = self.turns[split:]

            if older:
                sections.append(
                    "\n## Older turns (summarised — outcomes only unless pinned)"
                )
                for index, record in enumerate(older):
                    mode: TurnRenderMode = (
                        "full" if index in sticky else "summary"
                    )
                    superseded = _superseded_turn_number(
                        index, supersedes, self.turns
                    )
                    sections.append(
                        self._format_turn(
                            record,
                            mode=mode,
                            turn_index=index,
                            all_turns=self.turns,
                            superseded_by_turn=superseded,
                        )
                    )

            if recent:
                header = (
                    "\n## Recent turns (full output)"
                    if older
                    else "\n## Prior turns"
                )
                sections.append(header)
                for index, record in enumerate(recent, start=split):
                    superseded = _superseded_turn_number(
                        index, supersedes, self.turns
                    )
                    sections.append(
                        self._format_turn(
                            record,
                            mode="full",
                            turn_index=index,
                            all_turns=self.turns,
                            superseded_by_turn=superseded,
                        )
                    )

        if self.user_replies:
            sections.append(
                "\n## User clarifications (answers to your questions — act on these)"
            )
            for i, reply in enumerate(self.user_replies, start=1):
                sections.append(f"{i}. {reply}")

        if self.parse_errors:
            sections.append("\n## Parse errors (fix your JSON)")
            for err in self.parse_errors:
                sections.append(f"- {err}")

        sections.append(
            "\n## Your turn\n"
            "Return the next JSON action object to continue or finish the task."
        )
        return "\n".join(sections)

    @staticmethod
    def _format_turn(
        record: TurnRecord,
        *,
        mode: TurnRenderMode = "full",
        turn_index: int | None = None,
        all_turns: list[TurnRecord] | None = None,
        superseded_by_turn: int | None = None,
    ) -> str:
        """Render one turn for the model transcript."""
        if mode == "summary":
            return SessionContext._format_turn_summary(
                record,
                turn_index=turn_index,
                all_turns=all_turns,
                superseded_by_turn=superseded_by_turn,
            )
        return SessionContext._format_turn_full(
            record,
            superseded_by_turn=superseded_by_turn,
        )

    @staticmethod
    def _format_turn_summary(
        record: TurnRecord,
        *,
        turn_index: int | None,
        all_turns: list[TurnRecord] | None,
        superseded_by_turn: int | None = None,
    ) -> str:
        step = record.step
        lines = [f"\n### Turn {record.turn}"]
        outcome: str | None = None

        if step.action in _TERMINAL_ACTIONS:
            if step.message:
                lines.append(f"Outcome: {step.message}")
            return "\n".join(lines)

        if step.action in _READ_FILE_ACTIONS and step.path:
            outcome = f"Read {step.path}:{_format_read_range(step)}"
            if superseded_by_turn is not None:
                outcome += f" (superseded by turn {superseded_by_turn})"
        elif step.action == AgentAction.SEARCH_CODE:
            scope = step.path or step.glob or "."
            pattern = _preview_line(step.pattern or "", limit=80)
            tr = record.tool_result
            if tr and tr.stdout:
                hits = _parse_search_top_paths(tr.stdout)
                n = _count_list_entries(tr.stdout)
                top = ", ".join(hits) if hits else "none"
                outcome = f'Searched "{pattern}" in {scope}: {n} hits, top: {top}'
            else:
                outcome = f'Searched "{pattern}" in {scope}'
        elif step.action == AgentAction.LIST_FILES:
            scope = step.path or "."
            glob_part = f" ({step.glob})" if step.glob else ""
            tr = record.tool_result
            n = _count_list_entries(tr.stdout) if tr and tr.stdout else 0
            outcome = f"Listed {scope}{glob_part}: {n} entries"
        elif step.action == AgentAction.RUN_SHELL and step.command:
            tr = record.tool_result
            code = tr.exit_code if tr else None
            status = _tool_pass_fail(tr) if tr else "?"
            outcome = f"Shell: {step.command} → exit {code} ({status})"
            if tr and status == "FAIL" and tr.stderr:
                err = _first_nonempty_line(tr.stderr)
                if err:
                    lines.append(f"  stderr: {_preview_line(err)}")
        elif step.action == AgentAction.RUN_PYTHON:
            tr = record.tool_result
            first = _first_nonempty_line(step.code or "") or "(python)"
            code = tr.exit_code if tr else None
            status = _tool_pass_fail(tr) if tr else "?"
            outcome = f"Python: {_preview_line(first, limit=80)} → exit {code} ({status})"
            if tr and status == "FAIL" and tr.stderr:
                err = _first_nonempty_line(tr.stderr)
                if err:
                    lines.append(f"  stderr: {_preview_line(err)}")
        elif step.action in _EDIT_ACTIONS and step.path:
            delta = _edit_line_delta(step)
            delta_s = f"{delta:+d} lines" if delta is not None else "edited"
            tr = record.tool_result
            code = tr.exit_code if tr else None
            outcome = f"Edited {step.path}: {delta_s} (exit {code})"
        elif step.action == AgentAction.UPDATE_PLAN and step.plan_items:
            done = sum(
                1 for p in step.plan_items if p.status == PlanItemStatus.DONE
            )
            outcome = f"Plan: {len(step.plan_items)} items, {done} done"
            if turn_index is not None and all_turns is not None:
                prior = _prior_plan_ids_before_turn(all_turns, turn_index)
                new_items = [
                    p for p in step.plan_items if p.id not in prior
                ]
                for item in new_items[:5]:
                    lines.append(f"  + {item.id}: {item.text}")
        elif step.action == AgentAction.REMEMBER and step.memory:
            parts = [_preview_line(m, limit=120) for m in step.memory[:5]]
            outcome = "Remembered: " + "; ".join(parts)
        elif step.action == AgentAction.GIT:
            tr = record.tool_result
            op = step.git_op or "git"
            code = tr.exit_code if tr else None
            outcome = f"Git {op} → exit {code}"
        elif record.tool_result is not None:
            tr = record.tool_result
            outcome = (
                f"{step.action.value} → exit {tr.exit_code} "
                f"({_tool_pass_fail(tr)})"
            )

        if outcome:
            lines.append(outcome)
        elif step.thought:
            lines.append(f"Thought: {_preview_line(step.thought)}")
        elif step.message:
            lines.append(f"Message: {step.message}")

        return "\n".join(lines)

    @staticmethod
    def _format_turn_full(
        record: TurnRecord,
        *,
        superseded_by_turn: int | None = None,
    ) -> str:
        step = record.step
        lines = [f"\n### Turn {record.turn}"]
        if record.call_model:
            effort = record.call_reasoning_effort or "default"
            lines.append(f"LLM: {record.call_model} (reasoning: {effort})")
        if step.thought:
            lines.append(f"Thought: {step.thought}")
        lines.append(f"Action: {step.action.value}")
        if step.tools:
            lines.append(f"Tools: {', '.join(step.tools)}")

        if record.parse_error:
            lines.append(f"Parse note: {record.parse_error}")

        if step.path:
            lines.append(f"Path: {step.path}")
        if step.dest_path:
            lines.append(f"Dest: {step.dest_path}")
        if step.pattern:
            lines.append(f"Pattern: {_preview_line(step.pattern)}")
        if step.glob:
            lines.append(f"Glob: {step.glob}")
        if step.git_op:
            lines.append(f"Git op: {step.git_op}")
        if step.start_line is not None or step.end_line is not None:
            lines.append(
                f"Lines: {step.start_line or 1}-{step.end_line or 'end'}"
            )
        if step.command:
            lines.append(f"Command: {step.command}")
        if step.code:
            lines.append(f"Code:\n```python\n{step.code}\n```")
        if step.patch:
            lines.append(f"Patch:\n```diff\n{step.patch}\n```")
        if step.old_string is not None:
            preview = _preview_line(step.old_string.replace("\n", "\\n"))
            lines.append(f"Old: {preview}")
        if step.new_string is not None:
            preview = _preview_line(step.new_string.replace("\n", "\\n"))
            lines.append(f"New: {preview}")
        if step.message:
            lines.append(f"Message: {step.message}")
        if step.action == AgentAction.UPDATE_PLAN and step.plan_items:
            lines.append(f"Plan items: {len(step.plan_items)}")
        if step.action == AgentAction.REMEMBER and step.memory:
            for mem in step.memory:
                lines.append(f"Remember: {_preview_line(mem)}")

        if record.tool_result is not None:
            tr = record.tool_result
            lines.append(f"Executed: {tr.executed}")
            lines.append(f"Exit code: {tr.exit_code}")
            lines.append(f"Timed out: {tr.timed_out}")
            if superseded_by_turn is not None and step.action in _READ_FILE_ACTIONS:
                lines.append(
                    f"Read {step.path}:{_format_read_range(step)} "
                    f"(superseded by turn {superseded_by_turn})"
                )
            elif tr.stdout:
                lines.append(f"Stdout:\n{tr.stdout}")
            if tr.stderr:
                lines.append(f"Stderr:\n{tr.stderr}")

        return "\n".join(lines)


def _superseded_turn_number(
    index: int,
    supersedes: dict[int, int],
    turns: list[TurnRecord],
) -> int | None:
    """Public turn number that superseded the read at ``index``, if any."""
    latest_index = supersedes.get(index)
    if latest_index is None:
        return None
    return turns[latest_index].turn
