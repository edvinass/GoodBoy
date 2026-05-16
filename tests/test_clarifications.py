"""Tests for clarification loop guards."""

from agent.clarifications import (
    is_affirmative_reply,
    is_repeat_question,
    proceed_directive,
)


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
