"""Agent loop behavior with local models."""

from unittest.mock import MagicMock

from agent.loop import AgentLoop
from agent.types import AgentAction, AgentStep


def _ALLOWED_LOCAL():
    return ["local:qwen2.5-coder-7b-q4"]


def _llm_ok(**_kwargs):
    return '{"action":"task_complete","message":"done","status":"Done"}', None, None


def test_local_session_disables_response_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "local:qwen2.5-coder-7b-q4")
    from settings import get_settings

    get_settings.cache_clear()

    loop = AgentLoop(
        workspace=tmp_path,
        model="local:qwen2.5-coder-7b-q4",
        allowed_models=_ALLOWED_LOCAL(),
        llm_call=_llm_ok,
    )
    assert loop._chain_enabled is False


def test_local_llm_call_omits_reasoning_and_tools(tmp_path):
    captured: list[dict] = []

    def tracking_llm(**kwargs):
        captured.append(dict(kwargs))
        return (
            '{"action":"task_complete","message":"ok","status":"Done"}',
            None,
            None,
        )

    loop = AgentLoop(
        workspace=tmp_path,
        model="local:qwen2.5-coder-7b-q4",
        allowed_models=_ALLOWED_LOCAL(),
        llm_call=tracking_llm,
    )
    loop.run("List files in the workspace")
    assert captured
    call = captured[0]
    assert call["reasoning_effort"] is None
    assert "tools" not in call
    assert "previous_response_id" not in call


def test_local_session_uses_small_recent_full_turns(tmp_path, monkeypatch):
    """Local models default to a tighter prompt window to fit 8k contexts."""
    monkeypatch.delenv("GOODBOY_LOCAL_RECENT_FULL_TURNS", raising=False)
    monkeypatch.delenv("GOODBOY_CONTEXT_RECENT_FULL_TURNS", raising=False)
    from settings import DEFAULT_LOCAL_RECENT_FULL_TURNS, get_settings

    get_settings.cache_clear()
    loop = AgentLoop(
        workspace=tmp_path,
        model="local:qwen2.5-coder-7b-q4",
        allowed_models=_ALLOWED_LOCAL(),
        llm_call=_llm_ok,
    )
    assert loop.recent_full_turns == DEFAULT_LOCAL_RECENT_FULL_TURNS


def test_cloud_session_keeps_default_recent_full_turns(tmp_path, monkeypatch):
    monkeypatch.delenv("GOODBOY_LOCAL_RECENT_FULL_TURNS", raising=False)
    monkeypatch.delenv("GOODBOY_CONTEXT_RECENT_FULL_TURNS", raising=False)
    from settings import DEFAULT_CONTEXT_RECENT_FULL_TURNS, get_settings

    get_settings.cache_clear()
    loop = AgentLoop(
        workspace=tmp_path,
        model="gpt-5.4-nano",
        allowed_models=["gpt-5.4-nano"],
        llm_call=_llm_ok,
    )
    assert loop.recent_full_turns == DEFAULT_CONTEXT_RECENT_FULL_TURNS


def test_set_session_model_resizes_window(tmp_path, monkeypatch):
    """Switching between cloud and local models adjusts the prompt window."""
    monkeypatch.delenv("GOODBOY_LOCAL_RECENT_FULL_TURNS", raising=False)
    monkeypatch.delenv("GOODBOY_CONTEXT_RECENT_FULL_TURNS", raising=False)
    from settings import (
        DEFAULT_CONTEXT_RECENT_FULL_TURNS,
        DEFAULT_LOCAL_RECENT_FULL_TURNS,
        get_settings,
    )

    get_settings.cache_clear()
    loop = AgentLoop(
        workspace=tmp_path,
        model="gpt-5.4-nano",
        allowed_models=["gpt-5.4-nano", "local:qwen2.5-coder-7b-q4"],
        llm_call=_llm_ok,
    )
    assert loop.recent_full_turns == DEFAULT_CONTEXT_RECENT_FULL_TURNS

    loop.set_session_model("local:qwen2.5-coder-7b-q4")
    assert loop.recent_full_turns == DEFAULT_LOCAL_RECENT_FULL_TURNS

    loop.set_session_model("gpt-5.4-nano")
    assert loop.recent_full_turns == DEFAULT_CONTEXT_RECENT_FULL_TURNS


def test_explicit_recent_full_turns_overrides_local_default(tmp_path):
    """An explicit recent_full_turns kwarg wins over model-aware defaults."""
    loop = AgentLoop(
        workspace=tmp_path,
        model="local:qwen2.5-coder-7b-q4",
        allowed_models=_ALLOWED_LOCAL(),
        llm_call=_llm_ok,
        recent_full_turns=7,
    )
    assert loop.recent_full_turns == 7


def test_loop_run_resets_local_context_notice(tmp_path, monkeypatch):
    """Each task re-arms the one-shot context-full warning."""
    monkeypatch.setenv("OPENAI_MODEL", "local:qwen2.5-coder-7b-q4")
    from settings import get_settings

    get_settings.cache_clear()
    import agent.local_llm as local_llm

    local_llm._truncation_state["events"] = 4
    local_llm._truncation_state["warned_at"] = 1

    loop = AgentLoop(
        workspace=tmp_path,
        model="local:qwen2.5-coder-7b-q4",
        allowed_models=_ALLOWED_LOCAL(),
        llm_call=_llm_ok,
    )
    loop.run("anything")
    assert local_llm.context_full_events() == 0
    assert local_llm._truncation_state["warned_at"] == 0


def test_switch_tools_rejected_on_local_model(tmp_path):
    loop = AgentLoop(
        workspace=tmp_path,
        model="local:qwen2.5-coder-7b-q4",
        allowed_models=_ALLOWED_LOCAL(),
        llm_call=_llm_ok,
    )
    ui = MagicMock()
    ui.auto_model_switch = False
    loop._ui = ui
    step = AgentStep(
        action=AgentAction.SWITCH_TOOLS,
        tools=["web_search"],
        status="Enabling tools",
    )
    err = loop._validate_routing(step, "local:qwen2.5-coder-7b-q4")
    assert err is not None
    assert "switch_tools" in err
    assert "local" in err.lower()
