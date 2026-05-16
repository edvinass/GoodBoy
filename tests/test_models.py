"""Tests for model catalog and cost routing metadata."""

from agent.prompt import build_system_prompt
from agent.models import (
    CostTier,
    MODEL_CATALOG,
    OpenAITool,
    assert_catalog_covers_model_choices,
    cheapest_model_with_tools,
    format_cost_policy_section,
    format_hosted_tools_reference,
    format_models_section,
    get_model_spec,
    model_supports_openai_tool,
    models_for_prompt,
    resolve_reasoning_effort,
    validate_reasoning_effort_for_model,
)
from llm import MODEL_CHOICES


def test_catalog_covers_all_model_choices():
    assert_catalog_covers_model_choices()
    assert set(MODEL_CATALOG.keys()) == set(MODEL_CHOICES)


def test_gpt_5_5_is_premium():
    spec = get_model_spec("gpt-5.5")
    assert spec is not None
    assert spec.cost_tier == CostTier.PREMIUM
    assert spec.cost_index == 35.0


def test_models_sorted_cheapest_first():
    allowed = list(MODEL_CHOICES)
    specs = models_for_prompt(allowed)
    indices = [s.cost_index for s in specs if s.id in MODEL_CATALOG]
    assert indices == sorted(indices)


def test_format_models_section_includes_cost_policy_fields():
    text = format_models_section(["gpt-4o-mini", "gpt-5.5"])
    assert "gpt-4o-mini" in text
    assert "minimal" in text
    assert "gpt-5.5" in text
    assert "premium" in text
    assert "$" in text


def test_format_cost_policy_section():
    text = format_cost_policy_section()
    assert "Cost policy" in text
    assert "Escalation ladder" in text
    assert "gpt-4o-mini" in text


def test_resolve_reasoning_effort_non_reasoning_model():
    assert resolve_reasoning_effort("gpt-4o-mini", "low") is None


def test_resolve_reasoning_effort_reasoning_model():
    assert resolve_reasoning_effort("gpt-5.4-mini", "medium") == "medium"


def test_validate_reasoning_effort_rejects_on_gpt4o_mini():
    err = validate_reasoning_effort_for_model("gpt-4o-mini", "low")
    assert err is not None
    assert "does not support" in err


def test_nano_lacks_computer_use_and_tool_search():
    spec = get_model_spec("gpt-5.4-nano")
    assert spec is not None
    assert not model_supports_openai_tool("gpt-5.4-nano", OpenAITool.COMPUTER_USE)
    assert not model_supports_openai_tool("gpt-5.4-nano", OpenAITool.TOOL_SEARCH)


def test_cheapest_model_with_tools():
    model = cheapest_model_with_tools(
        ["gpt-5.5", "gpt-4o-mini", "gpt-4.1-nano"],
        [OpenAITool.WEB_SEARCH],
    )
    assert model == "gpt-4.1-nano"


def test_format_hosted_tools_reference():
    text = format_hosted_tools_reference()
    assert "web_search" in text
    assert "file_search" in text


def test_build_system_prompt_coding_agent_identity():
    prompt = build_system_prompt(allowed_models=["gpt-4o-mini"])
    assert "powerful autonomous coding agent" in prompt
    assert "Core objective" in prompt
    assert "Inspect the codebase before making changes" in prompt
    assert "rg " in prompt
    assert "perform" in prompt.lower()


def test_build_system_prompt_includes_cost_policy():
    prompt = build_system_prompt(allowed_models=["gpt-4o-mini", "gpt-5.5"])
    assert "Cost policy" in prompt
    assert "Configuration guide" in prompt
    assert "switch_tools" in prompt
    assert "switch_model" in prompt
    assert "reasoning_effort" in prompt
    assert "separate turn" in prompt.lower() or "separate turns" in prompt.lower()
    assert "web_search" in prompt
    assert "run_shell" in prompt
    assert "gpt-4o-mini" in prompt


def test_build_system_prompt_visibility_without_debug():
    prompt = build_system_prompt(allowed_models=["gpt-4o-mini"], debug=False)
    assert "Debug mode is **off**" in prompt
    assert "does **not** see run_shell commands" in prompt
    assert "does **not** see" in prompt and "thought" in prompt
    assert "run_python code" in prompt
    assert "shell command or Python preview" not in prompt


def test_build_system_prompt_visibility_with_thoughts():
    prompt = build_system_prompt(
        allowed_models=["gpt-4o-mini"],
        debug=False,
        show_thoughts=True,
    )
    assert "goodboy -f" in prompt
    assert "optional `thought`" in prompt


def test_build_system_prompt_visibility_with_debug():
    prompt = build_system_prompt(allowed_models=["gpt-4o-mini"], debug=True)
    assert "goodboy -c" in prompt
    assert "run_shell commands" in prompt
    assert "stdout/stderr" in prompt


def test_gpt_5_4_mini_has_full_tools():
    assert model_supports_openai_tool("gpt-5.4-mini", OpenAITool.TOOL_SEARCH)
    assert model_supports_openai_tool("gpt-5.4-mini", OpenAITool.COMPUTER_USE)
