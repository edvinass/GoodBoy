"""Model capability, pricing, and OpenAI hosted-tool catalog.

Refresh from https://platform.openai.com/docs/models and /docs/pricing when models change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet

from llm import MODEL_CHOICES, REASONING_EFFORT

# Hosted Responses API tools (documented only in v1; not invoked by harness).
class OpenAITool(str, Enum):
    WEB_SEARCH = "web_search"
    FILE_SEARCH = "file_search"
    IMAGE_GENERATION = "image_generation"
    CODE_INTERPRETER = "code_interpreter"
    HOSTED_SHELL = "hosted_shell"
    APPLY_PATCH = "apply_patch"
    SKILLS = "skills"
    COMPUTER_USE = "computer_use"
    MCP = "mcp"
    TOOL_SEARCH = "tool_search"


class ModelFamily(str, Enum):
    FRONTIER = "frontier"
    REASONING = "reasoning"
    GENERAL = "general"


class CostTier(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


# gpt-5.5 / gpt-5.4 / gpt-5.4-mini: full hosted stack
_FULL_OPENAI_TOOLS: FrozenSet[OpenAITool] = frozenset(OpenAITool)
# gpt-5.4-nano / gpt-5-nano: no computer_use, tool_search
_NANO_OPENAI_TOOLS: FrozenSet[OpenAITool] = _FULL_OPENAI_TOOLS - {
    OpenAITool.COMPUTER_USE,
    OpenAITool.TOOL_SEARCH,
}
# gpt-4.1-nano: text/JSON only (no hosted Responses tools on the API)
_NO_HOSTED_OPENAI_TOOLS: FrozenSet[OpenAITool] = frozenset()
# gpt-4.1-mini / gpt-4.1: partial hosted stack
_GPT41_OPENAI_TOOLS: FrozenSet[OpenAITool] = frozenset(
    {
        OpenAITool.WEB_SEARCH,
        OpenAITool.FILE_SEARCH,
        OpenAITool.IMAGE_GENERATION,
        OpenAITool.CODE_INTERPRETER,
        OpenAITool.MCP,
    }
)

_GPT5_REASONING = ("none", "low", "medium", "high", "xhigh")
_GPT5_LEGACY_REASONING = ("minimal", "low", "medium", "high")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: ModelFamily
    best_for: str
    avoid_when: str
    cost_tier: CostTier
    price_input_per_1m: float
    price_cached_input_per_1m: float
    price_output_per_1m: float
    reasoning: bool
    reasoning_efforts: tuple[str, ...] | None
    openai_tools: FrozenSet[OpenAITool]

    @property
    def cost_index(self) -> float:
        return self.price_input_per_1m + self.price_output_per_1m


def _spec(
    model_id: str,
    *,
    family: ModelFamily,
    best_for: str,
    avoid_when: str,
    cost_tier: CostTier,
    price_in: float,
    price_cached: float,
    price_out: float,
    reasoning: bool = False,
    reasoning_efforts: tuple[str, ...] | None = None,
    openai_tools: FrozenSet[OpenAITool] | None = None,
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        family=family,
        best_for=best_for,
        avoid_when=avoid_when,
        cost_tier=cost_tier,
        price_input_per_1m=price_in,
        price_cached_input_per_1m=price_cached,
        price_output_per_1m=price_out,
        reasoning=reasoning,
        reasoning_efforts=reasoning_efforts,
        openai_tools=(
            _GPT41_OPENAI_TOOLS if openai_tools is None else openai_tools
        ),
    )


MODEL_CATALOG: dict[str, ModelSpec] = {
    "gpt-4.1-nano": _spec(
        "gpt-4.1-nano",
        family=ModelFamily.GENERAL,
        best_for="Cheapest text: classify next step, format JSON, trivial edits.",
        avoid_when="Hosted tools (web_search, etc.) or hard debugging.",
        cost_tier=CostTier.MINIMAL,
        price_in=0.10,
        price_cached=0.025,
        price_out=0.40,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
    "gpt-5-nano": _spec(
        "gpt-5-nano",
        family=ModelFamily.FRONTIER,
        best_for="Fastest, most cost-efficient GPT-5 for simple high-volume tasks.",
        avoid_when="Computer use, tool_search, or deep debugging.",
        cost_tier=CostTier.MINIMAL,
        price_in=0.05,
        price_cached=0.005,
        price_out=0.40,
        reasoning=True,
        reasoning_efforts=_GPT5_LEGACY_REASONING,
        openai_tools=_NANO_OPENAI_TOOLS,
    ),
    "gpt-5.4-nano": _spec(
        "gpt-5.4-nano",
        family=ModelFamily.FRONTIER,
        best_for="Cheapest GPT-5.4-class model for simple high-volume tasks.",
        avoid_when="Computer use, tool_search, or deep debugging.",
        cost_tier=CostTier.LOW,
        price_in=0.20,
        price_cached=0.02,
        price_out=1.25,
        reasoning=True,
        reasoning_efforts=_GPT5_REASONING,
        openai_tools=_NANO_OPENAI_TOOLS,
    ),
    "gpt-5-mini": _spec(
        "gpt-5-mini",
        family=ModelFamily.FRONTIER,
        best_for="Near-frontier intelligence for cost-sensitive, low-latency, high-volume work.",
        avoid_when="Hardest debugging or tasks needing gpt-5.4+ precision.",
        cost_tier=CostTier.LOW,
        price_in=0.25,
        price_cached=0.025,
        price_out=2.00,
        reasoning=True,
        reasoning_efforts=_GPT5_LEGACY_REASONING,
        openai_tools=_FULL_OPENAI_TOOLS,
    ),
    "gpt-4.1-mini": _spec(
        "gpt-4.1-mini",
        family=ModelFamily.GENERAL,
        best_for="Multi-step plans, instruction-following, 1M context at low cost.",
        avoid_when="Deepest reasoning or last-resort debugging.",
        cost_tier=CostTier.LOW,
        price_in=0.40,
        price_cached=0.10,
        price_out=1.60,
    ),
    "gpt-5.4-mini": _spec(
        "gpt-5.4-mini",
        family=ModelFamily.FRONTIER,
        best_for="Strongest mini model for coding, computer use, and subagents.",
        avoid_when="grep, ls, echo, and other trivial commands.",
        cost_tier=CostTier.MEDIUM,
        price_in=0.75,
        price_cached=0.075,
        price_out=4.50,
        reasoning=True,
        reasoning_efforts=_GPT5_REASONING,
        openai_tools=_FULL_OPENAI_TOOLS,
    ),
    "gpt-5": _spec(
        "gpt-5",
        family=ModelFamily.FRONTIER,
        best_for="Previous-generation reasoning model for coding and agentic tasks.",
        avoid_when="When gpt-5.4-mini or gpt-5.4 is available at similar cost.",
        cost_tier=CostTier.MEDIUM,
        price_in=1.25,
        price_cached=0.125,
        price_out=10.00,
        reasoning=True,
        reasoning_efforts=_GPT5_LEGACY_REASONING,
        openai_tools=_FULL_OPENAI_TOOLS,
    ),
    "gpt-5.4": _spec(
        "gpt-5.4",
        family=ModelFamily.FRONTIER,
        best_for="More affordable frontier model for coding and professional work.",
        avoid_when="Trivial one-liners or when gpt-5.4-mini suffices.",
        cost_tier=CostTier.HIGH,
        price_in=2.50,
        price_cached=0.25,
        price_out=15.00,
        reasoning=True,
        reasoning_efforts=_GPT5_REASONING,
        openai_tools=_FULL_OPENAI_TOOLS,
    ),
    "gpt-4.1": _spec(
        "gpt-4.1",
        family=ModelFamily.GENERAL,
        best_for="Smartest non-reasoning model; very long context (huge stdout/history).",
        avoid_when="Simple tasks that fit a minimal-tier model.",
        cost_tier=CostTier.HIGH,
        price_in=2.00,
        price_cached=0.50,
        price_out=8.00,
    ),
    "gpt-5.5": _spec(
        "gpt-5.5",
        family=ModelFamily.FRONTIER,
        best_for="Most advanced model for coding and professional work; use when quality matters most.",
        avoid_when="Trivial commands, first attempt, or cost-sensitive high-volume steps.",
        cost_tier=CostTier.PREMIUM,
        price_in=5.00,
        price_cached=0.50,
        price_out=30.00,
        reasoning=True,
        reasoning_efforts=_GPT5_REASONING,
        openai_tools=_FULL_OPENAI_TOOLS,
    ),
}


@dataclass(frozen=True)
class ReasoningEffortSpec:
    level: str
    best_for: str
    avoid_when: str
    cost_hint: str


REASONING_EFFORT_CATALOG: tuple[ReasoningEffortSpec, ...] = (
    ReasoningEffortSpec(
        "none",
        "Latency-critical steps; classification; simple command formatting.",
        "Debugging or multi-step planning.",
        "low",
    ),
    ReasoningEffortSpec(
        "minimal",
        "Very light planning before a single tool call.",
        "Hard bugs after failed attempts.",
        "low",
    ),
    ReasoningEffortSpec(
        "low",
        "Light tool-use planning, drafting, routine coding.",
        "Deep debugging without trying medium first.",
        "medium",
    ),
    ReasoningEffortSpec(
        "medium",
        "Default max for gpt-5.4-mini on non-trivial agent steps.",
        "Trivial echo/status/ls one-liners.",
        "high",
    ),
    ReasoningEffortSpec(
        "high",
        "Complex debugging, deep planning, repeated tool failures.",
        "Simple grep/cat/ls.",
        "very high",
    ),
    ReasoningEffortSpec(
        "xhigh",
        "Deep research, security review, very hard coding.",
        "Every turn — cost and latency explode.",
        "very high",
    ),
)


def get_model_spec(model_id: str) -> ModelSpec | None:
    return MODEL_CATALOG.get(model_id)


def format_model_select_label(model_id: str, description: str) -> str:
    """Label for interactive model picker (includes output token price)."""
    if model_id.startswith("local:"):
        return description
    spec = get_model_spec(model_id)
    if spec is None:
        return description
    return f"{description} · ${spec.price_output_per_1m:.2f}/1M output"


def is_reasoning_model(model_id: str) -> bool:
    spec = get_model_spec(model_id)
    return spec.reasoning if spec is not None else False


def model_supports_openai_tool(model_id: str, tool: OpenAITool | str) -> bool:
    spec = get_model_spec(model_id)
    if spec is None:
        return False
    key = OpenAITool(tool) if isinstance(tool, str) else tool
    return key in spec.openai_tools


def models_for_prompt(allowed_ids: list[str]) -> list[ModelSpec]:
    """Catalog entries for allowed IDs, cheapest first (unknown IDs omitted)."""
    specs = [MODEL_CATALOG[mid] for mid in allowed_ids if mid in MODEL_CATALOG]
    return sorted(specs, key=lambda s: s.cost_index)


def cheapest_model_with_tools(
    allowed_ids: list[str],
    tools: list[OpenAITool | str],
) -> str | None:
    """Cheapest allowlisted model that supports every requested hosted tool."""
    required = {
        OpenAITool(tool) if isinstance(tool, str) else tool for tool in tools
    }
    for spec in models_for_prompt(allowed_ids):
        if required <= spec.openai_tools:
            return spec.id
    return None


def cheapest_capable_model(
    allowed_ids: list[str],
    *,
    max_tier: CostTier = CostTier.PREMIUM,
) -> str | None:
    tier_order = list(CostTier)
    max_rank = tier_order.index(max_tier)
    candidates = [
        s
        for s in models_for_prompt(allowed_ids)
        if s.id in MODEL_CATALOG and tier_order.index(s.cost_tier) <= max_rank
    ]
    if not candidates:
        return allowed_ids[0] if allowed_ids else None
    return candidates[0].id


def resolve_reasoning_effort(model_id: str, effort: str | None) -> str | None:
    """Return effort for API call, or None if unsupported / omitted."""
    if effort is None or effort == "":
        return None
    if effort not in REASONING_EFFORT:
        return None
    spec = get_model_spec(model_id)
    if spec is None or not spec.reasoning:
        return None
    if spec.reasoning_efforts is None:
        return None
    if effort not in spec.reasoning_efforts:
        return None
    if effort == "none":
        return "none"
    return effort


def validate_reasoning_effort_for_model(model_id: str, effort: str) -> str | None:
    """Return error message if effort is invalid for model, else None."""
    if effort not in REASONING_EFFORT:
        return (
            f"Unknown reasoning_effort '{effort}'. "
            f"Use one of: {', '.join(REASONING_EFFORT)}."
        )
    spec = get_model_spec(model_id)
    target = model_id
    if spec is None or not spec.reasoning:
        return (
            f"Model '{target}' does not support reasoning_effort; omit the field."
        )
    if spec.reasoning_efforts and effort not in spec.reasoning_efforts:
        return (
            f"reasoning_effort '{effort}' not allowed for '{target}'. "
            f"Use one of: {', '.join(spec.reasoning_efforts)}."
        )
    return None


def _format_openai_tools(spec: ModelSpec) -> str:
    if not spec.openai_tools:
        return "function calling + structured outputs only"
    names = sorted(t.value for t in spec.openai_tools)
    return ", ".join(names)


def format_models_section(allowed_ids: list[str]) -> str:
    lines = [
        "## Available models (set `model` for the *next* LLM call; cheapest first)",
        "",
    ]
    for spec in models_for_prompt(allowed_ids):
        price = (
            f"${spec.price_input_per_1m:.2f}/${spec.price_output_per_1m:.2f}/1M"
        )
        header = (
            f"- **{spec.id}** [{spec.cost_tier.value}, {price}, "
            f"{spec.family.value}] — {spec.best_for} Avoid: {spec.avoid_when}"
        )
        lines.append(header)
        extras: list[str] = []
        if spec.reasoning and spec.reasoning_efforts:
            extras.append(
                "reasoning_efforts: " + ", ".join(spec.reasoning_efforts)
            )
        extras.append(f"hosted tools: {_format_openai_tools(spec)}")
        lines.append("  - " + "; ".join(extras))
    return "\n".join(lines)


def format_reasoning_section() -> str:
    lines = [
        "## Reasoning effort catalog",
        "Set `reasoning_effort` on any action except switch_tools (configures the next LLM call).",
        "Valid only when the next model is a reasoning model that lists the level below. Omit on gpt-4.1-*.",
        "Start low; escalate one level only after ambiguity or repeated failure.",
        "",
    ]
    for spec in REASONING_EFFORT_CATALOG:
        lines.append(
            f"- **{spec.level}** (cost {spec.cost_hint}): {spec.best_for}"
        )
    return "\n".join(lines)


def format_hosted_tools_reference() -> str:
    tool_ids = ", ".join(f'"{t.value}"' for t in OpenAITool)
    return f"""## Hosted tool IDs (for switch_tools `tools` array)
Valid: {tool_ids}
- **web_search**: live web pages (news, prices, current events).
- **file_search**: uploaded vector stores (not local files — use run_shell).
- **code_interpreter**: sandboxed Python/charts in OpenAI (not run_python).
Other IDs: see model catalog — not every model supports every tool."""


_FIXED_SESSION_COST_POLICY = """## Cost policy (session model fixed)
Automatic model switching is **off**. Use efficient steps: explore with run_shell, small edits, verify with tests/linters.
The session model and reasoning effort are fixed unless the user changes them with `/model` or `/reasoning` — do not set `model` or `reasoning_effort` in JSON."""

_STRICT_COST_POLICY = """## Cost policy (required)
- Pick the cheapest model + lowest reasoning effort that can succeed.
- Simple steps (ls/cat/echo/single-file edit/running tests/formatting): minimal tier, no reasoning effort.
- Do NOT use gpt-5.5 or high/xhigh reasoning for simple tasks.
- Escalate one tier at a time only after a failed or ambiguous turn.
- Prefer gpt-5.4-nano, gpt-5-nano, or gpt-4.1-nano for turn 1 unless the task is obviously hard.

## Escalation ladder
1. Simple / turn 1: gpt-5.4-nano, gpt-5-nano, or gpt-4.1-nano — no reasoning effort.
2. Mild: gpt-5-mini, gpt-4.1-mini, or gpt-5.4-nano + reasoning.effort=low.
3. Multi-step or one failure: gpt-5.4-mini + medium max.
4. Professional coding / still stuck: gpt-5.4 before gpt-5.5.
5. Messy tool output: gpt-5.4-mini + medium.
6. Stuck 2+ turns: gpt-4.1 (long context if stdout/history is huge).
7. Last resort: gpt-5.5 + high — never for step 1."""

_AUTO_MODEL_SWITCH_POLICY = """## Model routing (automatic switching enabled)
User enabled automatic switching (`/autoswitch` or `GOODBOY_AUTO_MODEL_SWITCH`). Turn 1 of each task runs on the cheapest allowlisted model (routing). On that turn you **must** set **model** (and optional **reasoning_effort**) on the action, or use **switch_model**, to pick the cheapest model that can handle turn 2+.

After turn 1, change **model** and **reasoning_effort** between turns (including via **switch_model**) when complexity, failures, hosted-tool needs, or long context justify it — don't wait for repeated failures. Step down to a cheaper model once the hard part is done.

**Complex tasks** (refactor, migrate, multi-file): on the routing turn pick gpt-5.4-mini (or similar) with reasoning_effort **medium** for turn 2+; step down after implementation and verification are done.

- Prefer setting `model` on run_shell/run_python when you also run a command; use switch_model only when changing model without a tool run.
- Don't use gpt-5.5 or high/xhigh reasoning for trivial steps.
- reasoning_effort on gpt-5.x: default none/low; medium when needed; high/xhigh only when stuck."""


def format_cost_policy_section(*, auto_model_switch: bool = False) -> str:
    if auto_model_switch:
        return _AUTO_MODEL_SWITCH_POLICY
    return _FIXED_SESSION_COST_POLICY


def assert_catalog_covers_model_choices() -> None:
    """Ensure MODEL_CHOICES from llm.py all have catalog entries."""
    missing = [m for m in MODEL_CHOICES if m not in MODEL_CATALOG]
    if missing:
        raise RuntimeError(f"MODEL_CATALOG missing entries for: {missing}")
