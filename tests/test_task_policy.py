"""Tests for task complexity and harness policy helpers."""

from agent.context import SessionContext
from agent.task_policy import (
    edits_since_last_verify,
    is_complex_task,
    is_verification_command,
    plan_blocks_edit,
    plan_mode_requires_plan,
    plan_required,
    validate_plan_submission,
    verification_blocks_complete,
)
from agent.types import AgentAction, PlanItem, PlanItemStatus


def test_is_complex_task_keywords():
    assert is_complex_task("refactor the auth module") is True
    assert is_complex_task("run pytest -q") is False


def test_plan_mode_auto():
    assert plan_mode_requires_plan("auto", "migrate database") is True
    assert plan_mode_requires_plan("auto", "echo hi") is False
    assert plan_mode_requires_plan("off", "refactor all") is False
    assert plan_mode_requires_plan("always", "echo hi") is True


def test_plan_required_when_empty():
    ctx = SessionContext(user_task="refactor main.py")
    assert plan_required(ctx, plan_mode="auto") is True
    ctx.plan_items = [
        PlanItem(id="1", text="step", status=PlanItemStatus.PENDING)
    ]
    assert plan_required(ctx, plan_mode="auto") is False


def test_plan_blocks_edit():
    ctx = SessionContext(user_task="refactor main.py")
    err = plan_blocks_edit(ctx, AgentAction.STR_REPLACE, plan_mode="auto")
    assert err is not None
    assert "update_plan" in err


def test_validate_plan_submission_complex_first():
    ctx = SessionContext(user_task="refactor the codebase")
    err = validate_plan_submission(
        ctx,
        [PlanItem(id="1", text="only one", status=PlanItemStatus.PENDING)],
    )
    assert err is not None
    assert "2" in err


def test_is_verification_command():
    assert is_verification_command("pytest tests/ -q") is True
    assert is_verification_command("ls -la") is False


def test_edits_since_last_verify():
    ctx = SessionContext(user_task="t", last_edit_turn=5, last_verify_turn=3)
    assert edits_since_last_verify(ctx) is True
    ctx.last_verify_turn = 6
    assert edits_since_last_verify(ctx) is False


def test_verification_blocks_complete():
    ctx = SessionContext(user_task="t", last_edit_turn=2)
    err = verification_blocks_complete(ctx, verify_before_complete=True)
    assert err is not None
    assert verification_blocks_complete(ctx, verify_before_complete=False) is None
