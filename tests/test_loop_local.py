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
