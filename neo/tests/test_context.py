"""Tests for session context serialization."""

from agent.context import (
    DEFAULT_RECENT_FULL_TURNS,
    ContextWatermark,
    SessionContext,
)
from agent.types import (
    AgentAction,
    AgentStep,
    PlanItem,
    PlanItemStatus,
    ToolResult,
    TurnRecord,
)


def _shell_turn(turn: int, command: str, stdout: str, *, exit_code: int = 0) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        step=AgentStep(action=AgentAction.RUN_SHELL, command=command),
        tool_result=ToolResult(
            executed=command,
            stdout=stdout,
            exit_code=exit_code,
        ),
    )


def test_to_prompt_includes_task_and_tool_output():
    ctx = SessionContext(user_task="fix the bug")
    ctx.add_turn(
        TurnRecord(
            turn=1,
            step=AgentStep(
                action=AgentAction.RUN_SHELL,
                command="echo ok",
                thought="probe",
            ),
            tool_result=ToolResult(
                executed="echo ok",
                stdout="ok\n",
                exit_code=0,
            ),
        )
    )
    prompt = ctx.to_prompt()
    assert "fix the bug" in prompt
    assert "echo ok" in prompt
    assert "Stdout:" in prompt
    assert "ok" in prompt


def test_user_replies_in_prompt():
    ctx = SessionContext(user_task="deploy")
    ctx.add_user_reply("staging only")
    text = ctx.to_prompt()
    assert "staging only" in text
    assert "User clarifications" in text


def test_conversation_history_in_prompt():
    from agent.context import ConversationExchange

    ctx = SessionContext(
        user_task="and in Sydney",
        conversation_history=[
            ConversationExchange(
                user="weather in London",
                assistant="London: cloudy, 14°C",
            )
        ],
    )
    text = ctx.to_prompt()
    assert "Prior conversation" in text
    assert "weather in London" in text
    assert "London: cloudy" in text
    assert "and in Sydney" in text


def test_older_turns_summarised_drops_full_stdout():
    """Older turns lose their full stdout body; recent turns keep it."""
    big_stdout = "line-old-1\n" + ("noise\n" * 1000) + "line-old-last\n"
    ctx = SessionContext(user_task="t", recent_full_turns=2)
    ctx.add_turn(_shell_turn(1, "ls", big_stdout))
    ctx.add_turn(_shell_turn(2, "echo a", "a\n"))
    ctx.add_turn(_shell_turn(3, "echo b", "b\n"))

    text = ctx.to_prompt()

    # Sliding window header is present.
    assert "Older turns (summarised" in text
    assert "Recent turns (full output)" in text
    # Old turn loses its noisy body but keeps an outcome one-liner.
    assert "noise\nnoise" not in text
    assert "Stdout:\n" not in text.split("Recent turns")[0]
    assert "Shell: ls → exit 0 (PASS)" in text
    # Recent turns still rendered with full Stdout body.
    assert "Stdout:\na" in text
    assert "Stdout:\nb" in text


def test_recent_full_turns_window_keeps_last_n():
    ctx = SessionContext(user_task="t", recent_full_turns=2)
    for i in range(1, 6):
        ctx.add_turn(_shell_turn(i, f"cmd{i}", f"out-{i}\n"))

    text = ctx.to_prompt()

    # Last 2 turns (4 and 5) rendered full.
    assert "Stdout:\nout-4" in text
    assert "Stdout:\nout-5" in text
    # Earlier turns 1..3 are summarised — no full Stdout block.
    for i in (1, 2, 3):
        assert f"Stdout:\nout-{i}" not in text
        assert f"Shell: cmd{i} → exit 0 (PASS)" in text


def test_summary_keeps_message_for_terminal_turns():
    """need_user_input/task_complete carry their signal in `message`; summary mode preserves it."""
    ctx = SessionContext(user_task="t", recent_full_turns=1)
    ctx.add_turn(
        TurnRecord(
            turn=1,
            step=AgentStep(
                action=AgentAction.NEED_USER_INPUT,
                message="Which file should I touch?",
            ),
        )
    )
    ctx.add_turn(_shell_turn(2, "ls", "ok\n"))
    ctx.add_turn(_shell_turn(3, "ls", "ok\n"))

    text = ctx.to_prompt()
    # Turn 1 falls into the summarised bucket but its message survives.
    assert "Older turns (summarised" in text
    assert "Which file should I touch?" in text


def test_summary_includes_first_stderr_on_failure():
    failing_stderr = "ImportError: No module named foo\nTraceback ...\n"
    ctx = SessionContext(user_task="t", recent_full_turns=1)
    ctx.add_turn(
        TurnRecord(
            turn=1,
            step=AgentStep(action=AgentAction.RUN_SHELL, command="python -m foo"),
            tool_result=ToolResult(
                executed="python -m foo",
                stdout="",
                stderr=failing_stderr,
                exit_code=1,
            ),
        )
    )
    ctx.add_turn(_shell_turn(2, "echo ok", "ok\n"))

    text = ctx.to_prompt()
    older_section = text.split("Recent turns")[0]
    # Failed turns are sticky — rendered in full mode with stderr body.
    assert "ImportError: No module named foo" in older_section
    assert "Command: python -m foo" in older_section


def test_default_recent_full_turns_is_eight():
    """Sanity-check the documented default, since the env var defaults here too."""
    ctx = SessionContext(user_task="t")
    assert ctx.recent_full_turns == DEFAULT_RECENT_FULL_TURNS == 8


def test_watermark_snapshot_matches_current_lengths():
    ctx = SessionContext(user_task="t")
    ctx.add_turn(_shell_turn(1, "ls", "ok\n"))
    ctx.add_user_reply("yes")
    ctx.add_parse_error("bad json")
    snap = ctx.watermark()
    assert snap == ContextWatermark(turns=1, parse_errors=1, user_replies=1)


def test_to_prompt_delta_includes_only_new_items():
    ctx = SessionContext(user_task="t")
    ctx.add_turn(_shell_turn(1, "ls", "old-output\n"))
    ctx.add_user_reply("first reply")
    snap = ctx.watermark()
    ctx.add_turn(_shell_turn(2, "echo new", "new-output\n"))
    ctx.add_user_reply("proceed")
    ctx.add_parse_error("turn 2: bad json once")

    delta = ctx.to_prompt_delta(snap)

    # Only new turn data appears.
    assert "new-output" in delta
    assert "old-output" not in delta
    # The first user reply was already on the wire pre-watermark; only the new
    # one shows up, and it's indexed at its absolute position (2 of 2 total).
    assert "first reply" not in delta
    assert "2. proceed" in delta
    # Parse errors since the watermark are surfaced for the model to fix.
    assert "turn 2: bad json once" in delta


def test_to_prompt_delta_when_nothing_new_is_a_continue_marker():
    ctx = SessionContext(user_task="t")
    ctx.add_turn(_shell_turn(1, "ls", "ok\n"))
    snap = ctx.watermark()
    delta = ctx.to_prompt_delta(snap)
    assert "Continue" in delta
    assert "No new" in delta


def test_to_prompt_size_bounded_by_recent_window():
    """5 turns with 50 KB stdout each should not produce a >300 KB prompt."""
    big = "X" * 50_000
    ctx = SessionContext(user_task="t", recent_full_turns=2)
    for i in range(1, 6):
        ctx.add_turn(_shell_turn(i, f"cmd{i}", big))

    text = ctx.to_prompt()
    # With recent_full_turns=2, only 2 turns carry the full 50 KB body. Older 3
    # are summarised to a few hundred bytes each. Cap is generous to absorb
    # headers/footers but well under the naive 5 * 50 KB = 250 KB.
    assert len(text) < 2 * len(big) + 5_000
    # And the older turns' big bodies are gone.
    older_section = text.split("Recent turns")[0]
    assert big not in older_section


def test_format_plan_step_progress_shows_in_progress_index():
    from agent.context import format_plan_step_progress

    line = format_plan_step_progress(
        [
            PlanItem(id="1", text="done step", status=PlanItemStatus.DONE),
            PlanItem(id="2", text="active", status=PlanItemStatus.IN_PROGRESS),
            PlanItem(id="3", text="skipped", status=PlanItemStatus.CANCELLED),
            PlanItem(id="4", text="later", status=PlanItemStatus.PENDING),
        ]
    )
    assert line == '2/3 step "active"'


def test_format_plan_step_progress_falls_back_to_first_pending():
    from agent.context import format_plan_step_progress

    assert (
        format_plan_step_progress(
            [PlanItem(id="1", text="later", status=PlanItemStatus.PENDING)]
        )
        == '1/1 step "later"'
    )


def test_effective_plan_items_prefers_same_response_plan():
    from agent.context import SessionContext, effective_plan_items

    ctx = SessionContext(user_task="t")
    ctx.plan_items = [
        PlanItem(id="old", text="stale", status=PlanItemStatus.PENDING)
    ]
    raw = (
        '{"action":"read_file","path":"main.py"}'
        '{"action":"update_plan","plan_items":[{"id":"1","text":"fresh",'
        '"status":"in_progress"}]}'
    )
    items = effective_plan_items(ctx, raw)
    assert len(items) == 1
    assert items[0].text == "fresh"


def test_plan_items_from_response_reads_trailing_update_plan():
    import json

    from agent.context import plan_items_from_response
    from agent.types import AgentAction, AgentStep

    raw = json.dumps(
        AgentStep(action=AgentAction.RUN_SHELL, command="pwd").model_dump(
            mode="json"
        )
    ) + json.dumps(
        AgentStep(
            action=AgentAction.UPDATE_PLAN,
            plan_items=[
                PlanItem(id="1", text="recon", status=PlanItemStatus.IN_PROGRESS),
            ],
        ).model_dump(mode="json")
    )
    items = plan_items_from_response(raw)
    assert items is not None
    assert items[0].text == "recon"
    assert items[0].status == PlanItemStatus.IN_PROGRESS


def test_should_print_plan_progress():
    from agent.context import should_print_plan_progress
    from agent.types import PlanItem, PlanItemStatus

    pending = [
        PlanItem(id="1", text="a", status=PlanItemStatus.IN_PROGRESS),
        PlanItem(id="2", text="b", status=PlanItemStatus.PENDING),
    ]
    assert should_print_plan_progress([], pending) is True
    assert (
        should_print_plan_progress(
            [],
            [
                PlanItem(id="1", text="a", status=PlanItemStatus.DONE),
            ],
        )
        is False
    )

    done_one = [
        PlanItem(id="1", text="a", status=PlanItemStatus.DONE),
        PlanItem(id="2", text="b", status=PlanItemStatus.IN_PROGRESS),
    ]
    assert should_print_plan_progress(pending, done_one) is True
    assert should_print_plan_progress(done_one, done_one) is False

    only_progress = [
        PlanItem(id="1", text="a", status=PlanItemStatus.IN_PROGRESS),
        PlanItem(id="2", text="b", status=PlanItemStatus.PENDING),
    ]
    assert should_print_plan_progress(pending, only_progress) is False


def test_format_plan_items_marks_status():
    from agent.context import format_plan_items

    text = format_plan_items(
        [
            PlanItem(id="1", text="done step", status=PlanItemStatus.DONE),
            PlanItem(id="2", text="active", status=PlanItemStatus.IN_PROGRESS),
            PlanItem(id="3", text="skipped", status=PlanItemStatus.CANCELLED),
            PlanItem(id="4", text="later", status=PlanItemStatus.PENDING),
        ]
    )
    done_line, active_line, skipped_line, pending_line = text.splitlines()
    assert done_line == "[✓] 1. done step"
    assert active_line == "[→] 2. active"
    assert skipped_line.startswith("[–] 3.")
    assert "\u0336" in skipped_line
    assert pending_line == "[ ] 4. later"


def _read_turn(
    turn: int,
    path: str,
    stdout: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        step=AgentStep(
            action=AgentAction.READ_FILE,
            path=path,
            start_line=start_line,
            end_line=end_line,
        ),
        tool_result=ToolResult(
            executed=f"read_file {path}",
            stdout=stdout,
            exit_code=0,
        ),
    )


def test_read_file_dedup_supersedes_older_reads():
    body_v1 = "version-one\n" * 50
    body_v2 = "version-two\n" * 50
    ctx = SessionContext(user_task="t", recent_full_turns=1)
    ctx.add_turn(_read_turn(1, "foo.py", body_v1))
    ctx.add_turn(_shell_turn(2, "echo x", "x\n"))
    ctx.add_turn(_read_turn(3, "foo.py", body_v2))

    text = ctx.to_prompt()
    assert "superseded by turn 3" in text
    assert body_v1 not in text
    assert body_v2 in text


def test_read_file_dedup_supersedes_when_later_range_covers_earlier():
    """A later read whose range fully contains an earlier one supersedes it."""
    narrow = "narrow-body-line\n" * 20
    wide = "wide-body-line\n" * 200
    ctx = SessionContext(user_task="t", recent_full_turns=5)
    ctx.add_turn(_read_turn(1, "foo.py", narrow, start_line=10, end_line=30))
    ctx.add_turn(_read_turn(2, "foo.py", wide))

    text = ctx.to_prompt()
    assert "superseded by turn 2" in text
    assert "narrow-body-line" not in text
    assert "wide-body-line" in text


def test_read_file_not_superseded_when_ranges_only_overlap():
    """Partial overlap doesn't supersede — model still needs both views."""
    first = "first-body-line\n" * 5
    second = "second-body-line\n" * 5
    ctx = SessionContext(user_task="t", recent_full_turns=5)
    ctx.add_turn(_read_turn(1, "foo.py", first, start_line=1, end_line=50))
    ctx.add_turn(_read_turn(2, "foo.py", second, start_line=30, end_line=80))

    text = ctx.to_prompt()
    assert "superseded by turn" not in text
    assert "first-body-line" in text
    assert "second-body-line" in text


def test_read_file_invalidated_by_later_str_replace():
    """After an edit, the prior read body is stale and is dropped from context."""
    stale = "stale-body-line\n" * 20
    ctx = SessionContext(user_task="t", recent_full_turns=5)
    ctx.add_turn(_read_turn(1, "foo.py", stale))
    ctx.add_turn(
        TurnRecord(
            turn=2,
            step=AgentStep(
                action=AgentAction.STR_REPLACE,
                path="foo.py",
                old_string="x",
                new_string="y",
            ),
            tool_result=ToolResult(
                executed="str_replace foo.py",
                stdout="Updated foo.py\n",
                exit_code=0,
            ),
        )
    )

    text = ctx.to_prompt()
    assert "superseded by turn 2" in text
    assert "stale-body-line" not in text


def test_read_file_invalidated_by_apply_patch():
    stale = "patch-stale-line\n" * 20
    ctx = SessionContext(user_task="t", recent_full_turns=5)
    ctx.add_turn(_read_turn(1, "bar.py", stale))
    ctx.add_turn(
        TurnRecord(
            turn=2,
            step=AgentStep(
                action=AgentAction.APPLY_PATCH,
                path="bar.py",
                patch="--- a/bar.py\n+++ b/bar.py\n@@\n-a\n+b\n",
            ),
            tool_result=ToolResult(
                executed="apply_patch bar.py",
                stdout="Patched bar.py\n",
                exit_code=0,
            ),
        )
    )

    text = ctx.to_prompt()
    assert "superseded by turn 2" in text
    assert "patch-stale-line" not in text


def test_read_file_invalidated_by_move_destination():
    """A read against ``dest_path`` is invalidated by an earlier move of another file there."""
    stale_src = "stale-src-line\n" * 10
    ctx = SessionContext(user_task="t", recent_full_turns=5)
    ctx.add_turn(_read_turn(1, "old.py", stale_src))
    ctx.add_turn(
        TurnRecord(
            turn=2,
            step=AgentStep(
                action=AgentAction.MOVE_FILE,
                path="old.py",
                dest_path="new.py",
            ),
            tool_result=ToolResult(
                executed="move_file old.py -> new.py",
                stdout="Moved old.py -> new.py\n",
                exit_code=0,
            ),
        )
    )

    text = ctx.to_prompt()
    assert "superseded by turn 2" in text
    assert "stale-src-line" not in text


def test_read_file_dedup_only_supersedes_matching_path():
    """Edits to a different file must not invalidate unrelated reads."""
    body = "kept-body-line\n" * 10
    ctx = SessionContext(user_task="t", recent_full_turns=5)
    ctx.add_turn(_read_turn(1, "foo.py", body))
    ctx.add_turn(
        TurnRecord(
            turn=2,
            step=AgentStep(
                action=AgentAction.STR_REPLACE,
                path="bar.py",
                old_string="x",
                new_string="y",
            ),
            tool_result=ToolResult(
                executed="str_replace bar.py",
                stdout="Updated bar.py\n",
                exit_code=0,
            ),
        )
    )

    text = ctx.to_prompt()
    assert "superseded by turn" not in text
    assert "kept-body-line" in text


def test_dedupe_batch_steps_drops_duplicate_reads():
    """Duplicate read-only actions in one parallel batch run only once."""
    from agent.loop import AgentLoop

    steps = [
        AgentStep(action=AgentAction.READ_FILE, path="foo.py"),
        AgentStep(action=AgentAction.READ_FILE, path="foo.py"),
        AgentStep(
            action=AgentAction.READ_FILE,
            path="foo.py",
            start_line=10,
            end_line=20,
        ),
        AgentStep(action=AgentAction.READ_FILE, path="bar.py"),
        AgentStep(action=AgentAction.READ_FILE, path="bar.py"),
        AgentStep(
            action=AgentAction.SEARCH_CODE, pattern="hello", path="src"
        ),
        AgentStep(
            action=AgentAction.SEARCH_CODE, pattern="hello", path="src"
        ),
    ]

    deduped = AgentLoop._dedupe_batch_steps(steps)
    keys = [AgentLoop._batch_dedup_key(s) for s in deduped]
    assert len(deduped) == 4
    assert keys == [
        ("read_file", "foo.py", None, None),
        ("read_file", "foo.py", 10, 20),
        ("read_file", "bar.py", None, None),
        ("search_code", "hello", "src", "", False, None),
    ]


def test_sticky_failed_turn_keeps_full_output_when_old():
    fail_out = "FAIL_MARKER\n" + ("x\n" * 500)
    ctx = SessionContext(user_task="t", recent_full_turns=2)
    ctx.add_turn(
        TurnRecord(
            turn=1,
            step=AgentStep(action=AgentAction.RUN_SHELL, command="false"),
            tool_result=ToolResult(
                executed="false",
                stdout=fail_out,
                stderr="boom",
                exit_code=1,
            ),
        )
    )
    for i in range(2, 22):
        ctx.add_turn(_shell_turn(i, f"echo {i}", f"ok-{i}\n"))

    text = ctx.to_prompt()
    assert "FAIL_MARKER" in text
    assert "Older turns (summarised" in text


def test_token_budget_demotes_recent_turns():
    from agent.context import _estimated_tokens

    big = "Y" * 2000
    ctx = SessionContext(user_task="t", recent_full_turns=10, token_budget=1500)
    for i in range(1, 41):
        ctx.add_turn(_shell_turn(i, f"cmd{i}", big))

    text = ctx.to_prompt()
    assert _estimated_tokens(text) <= 1500


def test_effective_recent_full_turns_widens_complex_tasks():
    from agent.context import effective_recent_full_turns
    from settings import DEFAULT_CONTEXT_RECENT_FULL_TURNS

    base = DEFAULT_CONTEXT_RECENT_FULL_TURNS
    assert (
        effective_recent_full_turns(base, "refactor the entire codebase")
        == min(30, int(base * 2.0))
    )
    assert effective_recent_full_turns(base, "fix typo") == base


def test_loop_applies_complex_window_for_complex_task(tmp_path, monkeypatch):
    from agent.loop import AgentLoop
    from settings import DEFAULT_CONTEXT_RECENT_FULL_TURNS

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-nano")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "settings.ENV_FILE",
        tmp_path / ".env",
        raising=False,
    )
    get_settings = __import__("settings", fromlist=["get_settings"]).get_settings
    get_settings.cache_clear()

    loop = AgentLoop(workspace=tmp_path)
    window = loop._effective_recent_full_turns("refactor the entire auth module")
    assert window == min(30, int(DEFAULT_CONTEXT_RECENT_FULL_TURNS * 2.0))


def test_pinned_plan_and_memory_in_prompt_and_delta():
    ctx = SessionContext(user_task="t")
    ctx.plan_items = [
        PlanItem(id="1", text="run tests", status=PlanItemStatus.PENDING)
    ]
    ctx.working_memory = ["pytest tests/ -q"]
    full = ctx.to_prompt()
    assert "## Active plan" in full
    assert "run tests" in full
    assert "## Working memory" in full
    snap = ctx.watermark()
    delta = ctx.to_prompt_delta(snap)
    assert "## Active plan" in delta
    assert "pytest tests/" in delta
