"""Session context built across agent loop turns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.memory import load_project_memory
from agent.types import TurnRecord

DEFAULT_RECENT_FULL_TURNS = 3
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
        """Render only items appended since ``since`` for stateful chained calls.

        Used together with the OpenAI Responses API's ``previous_response_id``
        so the model already has the prior transcript on the server side and
        we only need to send what's new (latest tool result, new parse errors,
        new user reply). All new turns are rendered in `full` mode because
        they are the most recent context the model needs.
        """
        sections: list[str] = []
        new_turns = self.turns[since.turns:]
        new_errors = self.parse_errors[since.parse_errors:]
        new_replies = self.user_replies[since.user_replies:]

        if not (new_turns or new_errors or new_replies):
            return (
                "## Continue\nNo new turn output since your last response. "
                "Return the next JSON action object."
            )

        if new_turns:
            sections.append("## New turn output")
            for record in new_turns:
                sections.append(self._format_turn(record, mode="full"))

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
        return "\n".join(sections)

    def to_prompt(self) -> str:
        """Serialize context for the model's user message.

        Older turns are rendered in `summary` mode (command + exit code + a
        first/last stdout line); the most recent ``recent_full_turns`` turns
        keep their full stdout/stderr. The most recent turn is always rendered
        in full regardless of the cap.
        """
        sections: list[str] = []

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
            full_count = max(1, self.recent_full_turns)
            split = max(0, len(self.turns) - full_count)
            older = self.turns[:split]
            recent = self.turns[split:]

            if older:
                sections.append(
                    "\n## Older turns (summarised — full stdout/stderr elided)"
                )
                for record in older:
                    sections.append(self._format_turn(record, mode="summary"))

            if recent:
                header = (
                    "\n## Recent turns (full output)"
                    if older
                    else "\n## Prior turns"
                )
                sections.append(header)
                for record in recent:
                    sections.append(self._format_turn(record, mode="full"))

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
    def _format_turn(record: TurnRecord, *, mode: TurnRenderMode = "full") -> str:
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
        if step.model:
            next_effort = step.reasoning_effort or "(default)"
            lines.append(
                f"Next LLM: {step.model} (reasoning: {next_effort})"
            )

        if record.parse_error:
            lines.append(f"Parse note: {record.parse_error}")

        if step.path:
            lines.append(f"Path: {step.path}")
        if step.start_line is not None or step.end_line is not None:
            lines.append(
                f"Lines: {step.start_line or 1}-{step.end_line or 'end'}"
            )
        if step.command:
            lines.append(f"Command: {step.command}")
        if step.code:
            if mode == "summary":
                first = _first_nonempty_line(step.code) or ""
                lines.append(f"Code (first line): {_preview_line(first)}")
            else:
                lines.append(f"Code:\n```python\n{step.code}\n```")
        if step.patch and mode == "full":
            lines.append(f"Patch:\n```diff\n{step.patch}\n```")
        elif step.patch and mode == "summary":
            first = _first_nonempty_line(step.patch) or ""
            lines.append(f"Patch (first line): {_preview_line(first)}")
        if step.old_string is not None:
            preview = _preview_line(step.old_string.replace("\n", "\\n"))
            lines.append(f"Old: {preview}")
        if step.new_string is not None:
            preview = _preview_line(step.new_string.replace("\n", "\\n"))
            lines.append(f"New: {preview}")
        if step.message:
            # Routing/terminal turns carry their entire signal in `message`;
            # keep it whole even when summarising.
            lines.append(f"Message: {step.message}")

        if record.tool_result is not None:
            tr = record.tool_result
            lines.append(f"Executed: {tr.executed}")
            lines.append(f"Exit code: {tr.exit_code}")
            lines.append(f"Timed out: {tr.timed_out}")
            if mode == "full":
                if tr.stdout:
                    lines.append(f"Stdout:\n{tr.stdout}")
                if tr.stderr:
                    lines.append(f"Stderr:\n{tr.stderr}")
            else:
                first_out = _first_nonempty_line(tr.stdout) if tr.stdout else None
                last_out = _last_nonempty_line(tr.stdout) if tr.stdout else None
                if first_out:
                    lines.append(
                        f"Stdout (first line): {_preview_line(first_out)}"
                    )
                if last_out and last_out != first_out:
                    lines.append(
                        f"Stdout (last line): {_preview_line(last_out)}"
                    )
                failed = tr.exit_code not in (0, None) or tr.timed_out
                if failed and tr.stderr:
                    first_err = _first_nonempty_line(tr.stderr)
                    if first_err:
                        lines.append(
                            f"Stderr (first line): {_preview_line(first_err)}"
                        )

        return "\n".join(lines)
