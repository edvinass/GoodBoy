"""Detect clarification loops and affirmative user replies."""

from __future__ import annotations

import re

from agent.types import AgentAction

_AFFIRMATIVE = re.compile(
    r"\b("
    r"yes|yep|yeah|y|ok|okay|sure|confirm|confirmed|proceed|go ahead|"
    r"do it|just do|please do|you decide|decide yourself|up to you|"
    r"add files|create message and commit|commit|sounds good|approved|"
    r"no questions|don't ask|stop asking"
    r")\b",
    re.IGNORECASE,
)

_DELEGATION = re.compile(
    r"(you decide|decide yourself|up to you|your call|on you|handle it|"
    r"just (do|run|go)|proceed|no confirm)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def is_affirmative_reply(reply: str) -> bool:
    """True when the user clearly wants the agent to proceed."""
    stripped = reply.strip()
    if not stripped:
        return False
    if _DELEGATION.search(stripped):
        return True
    return bool(_AFFIRMATIVE.search(stripped))


def _token_set(text: str) -> set[str]:
    return {w for w in re.split(r"[^\w']+", normalize_text(text)) if len(w) > 2}


def is_repeat_question(previous: str, current: str) -> bool:
    """True when the agent is asking essentially the same question again."""
    a = normalize_text(previous)
    b = normalize_text(current)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return len(min(a, b, key=len)) > 20

    tokens_a = _token_set(previous)
    tokens_b = _token_set(current)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    if overlap >= 0.55:
        return True
    # Confirmation loops often share "confirm" + "commit" etc.
    confirm_tokens = {"confirm", "commit", "message", "please", "would", "like"}
    shared_confirm = len((tokens_a & tokens_b) & confirm_tokens)
    return shared_confirm >= 3


def count_need_user_input_turns(turns: list) -> int:
    return sum(
        1 for t in turns if t.step.action == AgentAction.NEED_USER_INPUT
    )


def last_need_user_input_message(turns: list) -> str | None:
    for record in reversed(turns):
        if record.step.action == AgentAction.NEED_USER_INPUT:
            return record.step.message
    return None


def proceed_directive(reply: str) -> str:
    return (
        "User has authorized you to proceed. Do not ask for confirmation again. "
        "Use run_shell or run_python to execute the task, then task_complete or failed."
    )


def repeat_question_directive() -> str:
    return (
        "You already asked this question and the user answered. "
        "Do not use need_user_input again for the same topic. "
        "Act with run_shell/run_python now."
    )
