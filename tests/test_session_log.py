"""Tests for session JSONL logging."""

import json
from pathlib import Path

import pytest

from agent.loop import AgentLoop, LoopOutcome
from agent.session_log import SessionLog, open_session_log
from agent.types import AgentAction, AgentStep

_ALLOWED = ["gpt-4o-mini", "gpt-5.4-mini", "gpt-5.5", "o4-mini"]


def _llm_responses(responses: list[AgentStep]):
    payloads = [json.dumps(step.model_dump(mode="json")) for step in responses]
    index = {"i": 0}

    def fake_llm(**_kwargs):
        i = index["i"]
        index["i"] += 1
        return payloads[i]

    return fake_llm


def test_session_log_writes_jsonl(tmp_path: Path):
    path = tmp_path / "session-test.jsonl"
    log = SessionLog(path)
    log.event("test", foo="bar")
    log.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["event"] == "test"
    assert record["foo"] == "bar"
    assert "ts" in record


def test_open_session_log_writes_start_and_end(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOODBOY_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("GOODBOY_SESSION_LOG", "1")
    from settings import get_settings

    get_settings.cache_clear()

    with open_session_log(workspace=tmp_path) as log:
        assert log is not None
        log.event("ping")

    files = list(tmp_path.glob("session-*.jsonl"))
    assert len(files) == 1
    events = [json.loads(line)["event"] for line in files[0].read_text().splitlines()]
    assert events[0] == "session_start"
    assert "ping" in events
    assert events[-1] == "session_end"


def test_loop_logs_llm_and_tool(tmp_path: Path):
    path = tmp_path / "task.jsonl"
    log = SessionLog(path)
    llm = _llm_responses(
        [
            AgentStep(action=AgentAction.RUN_SHELL, command="echo hi"),
            AgentStep(action=AgentAction.TASK_COMPLETE, message="done"),
        ]
    )
    loop = AgentLoop(
        workspace=tmp_path,
        max_turns=5,
        allowed_models=_ALLOWED,
        llm_call=llm,
    )
    result = loop.run("say hi", session_log=log)
    log.close()

    assert result.outcome == LoopOutcome.TASK_COMPLETE
    events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    assert "task_start" in events
    assert "llm_request" in events
    assert "llm_response" in events
    assert "agent_step" in events
    assert "tool_result" in events
    assert "task_end" in events


def test_session_log_disabled(monkeypatch):
    monkeypatch.setenv("GOODBOY_SESSION_LOG", "0")
    from settings import get_settings

    get_settings.cache_clear()
    with open_session_log() as log:
        assert log is None
