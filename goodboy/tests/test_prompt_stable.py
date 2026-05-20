"""Tests for cache-friendly prompt composition."""

from agent.prompt import (
    build_session_prompt_suffix,
    build_stable_system_prompt,
    build_system_prompt,
)


def test_visibility_suffix_not_in_stable_core():
    stable = build_stable_system_prompt(
        allowed_models=["gpt-5.4-nano"],
    )
    suffix = build_session_prompt_suffix(debug=True, show_commands=True)
    full = build_system_prompt(
        allowed_models=["gpt-5.4-nano"],
        debug=True,
        show_commands=True,
    )
    assert "User visibility" not in stable
    assert "User visibility" in suffix
    assert stable in full
    assert suffix in full
