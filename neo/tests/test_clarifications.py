"""Tests for clarification loop guards."""

from agent.clarifications import (
    build_stopped_message,
    is_affirmative_reply,
    is_concrete_instruction,
    is_repeat_question,
    proceed_directive,
    should_add_proceed_directive,
)
from agent.types import AgentAction, AgentStep, TurnRecord


def test_affirmative_replies():
    assert is_affirmative_reply("yes")
    assert is_affirmative_reply("You decide yourself")
    assert is_affirmative_reply("add files create message and commit")
    assert is_affirmative_reply("go ahead and do it")
    assert not is_affirmative_reply("maybe tomorrow")


def test_repeat_question():
    q = "Please confirm if you'd like me to create a commit message"
    assert is_repeat_question(q, q)
    assert is_repeat_question(
        q,
        "Please confirm if you want me to create a commit message based on the files",
    )
    assert not is_repeat_question("Which file?", "Which branch?")


def test_proceed_directive_mentions_act():
    assert "run_shell" in proceed_directive("yes")


def test_concrete_instruction_not_bare_affirmative():
    reply = "stage and commit my changes in current directory"
    assert is_affirmative_reply(reply)
    assert is_concrete_instruction(reply)
    assert not should_add_proceed_directive(reply)


def test_build_stopped_message_uses_step_thought():
    turns = [
        TurnRecord(
            turn=1,
            step=AgentStep(
                action=AgentAction.RUN_SHELL,
                command="git checkout -- main.py",
                thought="Reverted main.py to the last committed version.",
            ),
        )
    ]
    message = build_stopped_message(turns=turns)
    assert "Reverted main.py" in message
    assert "Paused — say continue when you're ready." in message
    assert "Stopped after the current step" not in message
