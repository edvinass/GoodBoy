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
    format_model_select_label,
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


def test_format_model_select_label_includes_output_price():
    label = format_model_select_label("gpt-4.1-nano", "GPT-4.1 nano")
    assert label == "GPT-4.1 nano · $0.40/1M output"


def test_format_model_select_label_unknown_model():
    assert format_model_select_label("unknown", "unknown") == "unknown"


def test_format_models_section_includes_cost_policy_fields():
    text = format_models_section(["gpt-5.4-nano", "gpt-5.5"])
    assert "gpt-5.4-nano" in text
    assert "low" in text
    assert "gpt-5.5" in text
    assert "premium" in text
    assert "$" in text


def test_format_cost_policy_section_fixed_session():
    text = format_cost_policy_section()
    assert "session model fixed" in text.lower()
    assert "Escalation ladder" not in text


def test_format_cost_policy_section_autoswitch():
    text = format_cost_policy_section(auto_model_switch=True)
    assert "automatic switching enabled" in text
    assert "Escalation ladder" not in text


def test_resolve_reasoning_effort_non_reasoning_model():
    assert resolve_reasoning_effort("gpt-4.1-nano", "low") is None


def test_resolve_reasoning_effort_reasoning_model():
    assert resolve_reasoning_effort("gpt-5.4-mini", "medium") == "medium"


def test_validate_reasoning_effort_rejects_on_gpt41_nano():
    err = validate_reasoning_effort_for_model("gpt-4.1-nano", "low")
    assert err is not None
    assert "does not support" in err


def test_nano_lacks_computer_use_and_tool_search():
    spec = get_model_spec("gpt-5.4-nano")
    assert spec is not None
    assert not model_supports_openai_tool("gpt-5.4-nano", OpenAITool.COMPUTER_USE)
    assert not model_supports_openai_tool("gpt-5.4-nano", OpenAITool.TOOL_SEARCH)


def test_gpt41_nano_has_no_hosted_tools():
    assert not model_supports_openai_tool("gpt-4.1-nano", OpenAITool.WEB_SEARCH)
    spec = get_model_spec("gpt-4.1-nano")
    assert spec is not None
    assert spec.openai_tools == frozenset()


def test_cheapest_model_with_tools():
    model = cheapest_model_with_tools(
        ["gpt-5.5", "gpt-5.4-nano", "gpt-4.1-nano"],
        [OpenAITool.WEB_SEARCH],
    )
    assert model == "gpt-5.4-nano"


def test_format_hosted_tools_reference():
    text = format_hosted_tools_reference()
    assert "web_search" in text
    assert "file_search" in text


def test_build_system_prompt_coding_agent_identity():
    prompt = build_system_prompt(allowed_models=["gpt-5.4-nano"])
    assert "powerful autonomous coding agent" in prompt
    assert "Core objective" in prompt
    assert "Inspect the codebase before making changes" in prompt
    assert "run_shell" in prompt and "rg" in prompt
    assert "perform" in prompt.lower()


def test_build_system_prompt_includes_cost_policy():
    prompt = build_system_prompt(allowed_models=["gpt-5.4-nano", "gpt-5.5"])
    assert "Cost policy" in prompt
    assert "session model fixed" in prompt.lower()
    assert "Configuration (hosted tools only)" in prompt
    assert "switch_tools" in prompt
    assert "Configuration guide" not in prompt
    assert "Do not set `model`" in prompt
    assert "reasoning_effort" in prompt
    assert "web_search" in prompt
    assert "run_shell" in prompt
    assert "gpt-5.4-nano" not in prompt
    assert "Reasoning effort catalog" not in prompt
    assert "Available models" not in prompt


def test_build_system_prompt_autoswitch_includes_model_catalogs():
    prompt = build_system_prompt(
        allowed_models=["gpt-5.4-nano", "gpt-5.5"],
        auto_model_switch=True,
    )
    assert "Configuration guide" in prompt
    assert "switch_model" in prompt
    assert "Escalate one tier at a time only after a failed" not in prompt
    assert "automatic switching enabled" in prompt
    assert "gpt-5.4-nano" in prompt
    assert "Reasoning effort catalog" in prompt


def test_build_system_prompt_visibility_without_debug():
    prompt = build_system_prompt(allowed_models=["gpt-5.4-nano"], debug=False)
    assert "Thought visibility" in prompt
    assert "optional `thought`" in prompt
    assert "do **not** see run_shell commands" in prompt
    assert "do **not** see" in prompt and "tool stdout/stderr" in prompt
    assert "run_python code" in prompt
    assert "goodboy -f" in prompt
    assert "shell command or Python preview" not in prompt


def test_build_system_prompt_visibility_with_thoughts():
    prompt = build_system_prompt(
        allowed_models=["gpt-5.4-nano"],
        debug=False,
        show_thoughts=True,
    )
    assert "goodboy -f" in prompt
    assert "optional `thought`" in prompt


def test_build_system_prompt_visibility_with_show_commands():
    prompt = build_system_prompt(
        allowed_models=["gpt-5.4-nano"],
        show_commands=True,
    )
    assert "goodboy -c" in prompt
    assert "run_shell commands" in prompt
    assert "do **not** see tool stdout/stderr" in prompt


def test_build_system_prompt_auto_model_switch_policy():
    prompt = build_system_prompt(
        allowed_models=["gpt-5.4-nano"],
        auto_model_switch=True,
    )
    assert "automatic switching enabled" in prompt
    assert "GOODBOY_AUTO_MODEL_SWITCH" in prompt
    assert "Turn 1" in prompt
    assert "routing" in prompt.lower()
    assert "session model fixed" not in prompt.lower()


def test_build_system_prompt_visibility_with_debug():
    prompt = build_system_prompt(allowed_models=["gpt-5.4-nano"], debug=True)
    assert "goodboy -d" in prompt
    assert "run_shell commands" in prompt
    assert "stdout/stderr" in prompt


def test_gpt_5_4_mini_has_full_tools():
    assert model_supports_openai_tool("gpt-5.4-mini", OpenAITool.TOOL_SEARCH)
    assert model_supports_openai_tool("gpt-5.4-mini", OpenAITool.COMPUTER_USE)
