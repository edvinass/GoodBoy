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
    # Old turn loses its noisy body but keeps command + a first/last preview.
    assert "noise\nnoise" not in text
    assert "Stdout:\n" not in text.split("Recent turns")[0]
    assert "Command: ls" in text
    assert "Stdout (first line): line-old-1" in text
    assert "Stdout (last line): line-old-last" in text
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
        assert f"Stdout (first line): out-{i}" in text


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
    assert "Stderr (first line): ImportError: No module named foo" in older_section
    assert "Traceback ..." not in older_section


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
    assert "[x] (1) done step" in text
    assert "[>] (2) active" in text
    assert "[-] (3) skipped" in text
    assert "[ ] (4) later" in text


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
