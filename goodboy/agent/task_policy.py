"""Task complexity, plan gates, and verification heuristics."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agent.types import AgentAction, PlanItem

if TYPE_CHECKING:
    from agent.context import SessionContext

_COMPLEX_KEYWORDS = re.compile(
    r"\b("
    r"refactor|architecture|migrate|security|audit|debug|investigate|"
    r"performance|optimi[sz]e|redesign|multi[- ]file|entire|whole|"
    r"codebase|review|analy[sz]e|root cause|stuck|broken build"
    r")\b",
    re.IGNORECASE,
)

_VERIFY_CMD = re.compile(
    r"\b("
    r"pytest|python\s+-m\s+pytest|npm\s+test|pnpm\s+test|yarn\s+test|"
    r"cargo\s+test|go\s+test|make\s+test|ruff\s+check|flake8|mypy|"
    r"eslint|tsc\b"
    r")\b",
    re.IGNORECASE,
)

_EDIT_ACTIONS = frozenset(
    {AgentAction.STR_REPLACE, AgentAction.APPLY_PATCH}
)


def is_complex_task(task: str) -> bool:
    text = task.strip()
    if not text:
        return False
    return bool(_COMPLEX_KEYWORDS.search(text))


def plan_mode_requires_plan(plan_mode: str, task: str) -> bool:
    mode = (plan_mode or "auto").strip().lower()
    if mode == "off":
        return False
    if mode == "always":
        return True
    return is_complex_task(task)


def plan_required(ctx: SessionContext, *, plan_mode: str) -> bool:
    if ctx.plan_items:
        return False
    return plan_mode_requires_plan(plan_mode, ctx.user_task)


def is_edit_action(action: AgentAction) -> bool:
    return action in _EDIT_ACTIONS


def plan_blocks_edit(
    ctx: SessionContext, action: AgentAction, *, plan_mode: str
) -> str | None:
    if not is_edit_action(action):
        return None
    if not plan_required(ctx, plan_mode=plan_mode):
        return None
    return (
        "This task requires a plan before file edits. Use update_plan with "
        "plan_items (at least 2 items for complex tasks), then str_replace or "
        "apply_patch."
    )


def validate_plan_submission(
    ctx: SessionContext, plan_items: list[PlanItem] | None
) -> str | None:
    if not plan_items:
        return "update_plan requires non-empty plan_items"
    if len(ctx.plan_items) == 0 and is_complex_task(ctx.user_task):
        if len(plan_items) < 2:
            return (
                "Complex task: first update_plan must include at least 2 "
                "plan_items"
            )
    return None


def is_verification_command(command: str) -> bool:
    return bool(_VERIFY_CMD.search(command.strip()))


def edits_since_last_verify(ctx: SessionContext) -> bool:
    if ctx.last_edit_turn is None:
        return False
    if ctx.last_verify_turn is None:
        return True
    return ctx.last_edit_turn > ctx.last_verify_turn


def verification_blocks_complete(
    ctx: SessionContext, *, verify_before_complete: bool
) -> str | None:
    if not verify_before_complete:
        return None
    if not edits_since_last_verify(ctx):
        return None
    return (
        "You edited files after the last successful test/lint run. Run the "
        "project verification command (e.g. pytest) and confirm exit 0 before "
        "task_complete."
    )
