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


_DEEPSEEK_REASONING_EFFORTS: tuple[str, ...] = ("none", "medium", "high")
# Adaptive thinking effort hints that map cleanly to Claude's effort scale.
_CLAUDE_REASONING_EFFORTS: tuple[str, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
)


MODEL_CATALOG: dict[str, ModelSpec] = {
    "deepseek-v4-flash": _spec(
        "deepseek-v4-flash",
        family=ModelFamily.GENERAL,
        best_for=(
            "Cost-efficient DeepSeek tier with 1M context; routine coding and "
            "high-volume tasks."
        ),
        avoid_when="Hardest reasoning — prefer deepseek-v4-pro.",
        cost_tier=CostTier.MINIMAL,
        price_in=0.14,
        price_cached=0.0028,
        price_out=0.28,
        reasoning=True,
        reasoning_efforts=_DEEPSEEK_REASONING_EFFORTS,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
    "deepseek-v4-pro": _spec(
        "deepseek-v4-pro",
        family=ModelFamily.REASONING,
        best_for=(
            "Frontier DeepSeek thinking-mode model for complex coding and "
            "multi-step reasoning (1M context)."
        ),
        avoid_when="Trivial one-liners — flash is much cheaper.",
        cost_tier=CostTier.MEDIUM,
        price_in=1.74,
        price_cached=0.0145,
        price_out=3.48,
        reasoning=True,
        reasoning_efforts=_DEEPSEEK_REASONING_EFFORTS,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
    "deepseek-chat": _spec(
        "deepseek-chat",
        family=ModelFamily.GENERAL,
        best_for=(
            "Legacy alias mapping to deepseek-v4-flash non-thinking mode "
            "(deprecated 2026-07-24)."
        ),
        avoid_when="New work — use deepseek-v4-flash directly.",
        cost_tier=CostTier.MINIMAL,
        price_in=0.14,
        price_cached=0.0028,
        price_out=0.28,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
    "deepseek-reasoner": _spec(
        "deepseek-reasoner",
        family=ModelFamily.REASONING,
        best_for=(
            "Legacy alias mapping to deepseek-v4-flash thinking mode "
            "(deprecated 2026-07-24)."
        ),
        avoid_when="New work — use deepseek-v4-flash or deepseek-v4-pro.",
        cost_tier=CostTier.MINIMAL,
        price_in=0.14,
        price_cached=0.0028,
        price_out=0.28,
        reasoning=True,
        reasoning_efforts=_DEEPSEEK_REASONING_EFFORTS,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
    "claude-opus-4-8": _spec(
        "claude-opus-4-8",
        family=ModelFamily.FRONTIER,
        best_for=(
            "Anthropic flagship; hardest reasoning, multi-step agentic work, "
            "1M context with adaptive thinking."
        ),
        avoid_when="Trivial commands or cost-sensitive high-volume steps.",
        cost_tier=CostTier.PREMIUM,
        price_in=5.00,
        price_cached=0.50,
        price_out=25.00,
        reasoning=True,
        reasoning_efforts=_CLAUDE_REASONING_EFFORTS,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
    "claude-opus-4-7": _spec(
        "claude-opus-4-7",
        family=ModelFamily.FRONTIER,
        best_for=(
            "Previous Anthropic flagship; strong reasoning for coding and "
            "long-horizon agents (adaptive thinking only)."
        ),
        avoid_when="When 4.8 is available at the same price.",
        cost_tier=CostTier.PREMIUM,
        price_in=5.00,
        price_cached=0.50,
        price_out=25.00,
        reasoning=True,
        reasoning_efforts=_CLAUDE_REASONING_EFFORTS,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
    "claude-sonnet-4-6": _spec(
        "claude-sonnet-4-6",
        family=ModelFamily.REASONING,
        best_for=(
            "Anthropic balanced default; best price/performance for most "
            "coding and agent workloads (1M context, adaptive thinking)."
        ),
        avoid_when="Single-turn trivial tasks where Haiku is enough.",
        cost_tier=CostTier.MEDIUM,
        price_in=3.00,
        price_cached=0.30,
        price_out=15.00,
        reasoning=True,
        reasoning_efforts=_CLAUDE_REASONING_EFFORTS,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
    "claude-haiku-4-5": _spec(
        "claude-haiku-4-5",
        family=ModelFamily.GENERAL,
        best_for=(
            "Anthropic fastest/cheapest; high-volume extraction, classification, "
            "and simple coding."
        ),
        avoid_when="Hard reasoning or multi-file refactors — use Sonnet/Opus.",
        cost_tier=CostTier.LOW,
        price_in=1.00,
        price_cached=0.10,
        price_out=5.00,
        openai_tools=_NO_HOSTED_OPENAI_TOOLS,
    ),
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


def model_supports_openai_tool(model_id: str, tool: OpenAITool | str) -> bool:
    spec = get_model_spec(model_id)
    if spec is None:
        return False
    key = OpenAITool(tool) if isinstance(tool, str) else tool
    return key in spec.openai_tools


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


def format_hosted_tools_reference() -> str:
    tool_ids = ", ".join(f'"{t.value}"' for t in OpenAITool)
    return f"""## Hosted tool IDs (for switch_tools `tools` array)
Valid: {tool_ids}
- **web_search**: live web pages (news, prices, current events).
- **file_search**: uploaded vector stores (not local files — use run_shell).
- **code_interpreter**: sandboxed Python/charts in OpenAI (not run_python).
Other IDs: see model catalog — not every model supports every tool."""


_FIXED_SESSION_COST_POLICY = """## Cost policy
Use efficient steps: explore with run_shell, small edits, verify with tests/linters.
Session model and reasoning effort are fixed for this run; the user changes them with `/model` or `/reasoning`."""


def format_cost_policy_section() -> str:
    return _FIXED_SESSION_COST_POLICY


def assert_catalog_covers_model_choices() -> None:
    """Ensure every OpenAI MODEL_CHOICES entry has catalog coverage.

    DeepSeek and local models are also catalogued but are tracked outside of
    ``MODEL_CHOICES`` (which is the OpenAI allowlist).
    """
    missing = [m for m in MODEL_CHOICES if m not in MODEL_CATALOG]
    if missing:
        raise RuntimeError(f"MODEL_CATALOG missing entries for: {missing}")
