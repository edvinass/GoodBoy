"""Tests for session context serialization."""

from agent.context import SessionContext
from agent.types import AgentAction, AgentStep, ToolResult, TurnRecord


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
