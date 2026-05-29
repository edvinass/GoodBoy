"""Tests for session JSONL logging."""

import json
from pathlib import Path

import pytest

from agent.loop import AgentLoop, LoopOutcome
from agent.session_log import SessionLog, instructions_digest, open_session_log
from agent.types import AgentAction, AgentStep

_ALLOWED = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.5"]


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
    monkeypatch.setenv("NEO_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("NEO_SESSION_LOG", "1")
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


def test_log_llm_request_dedupes_instructions(tmp_path: Path):
    path = tmp_path / "dedup.jsonl"
    log = SessionLog(path)
    long_instructions = "system prompt " * 500
    log.log_llm_request(
        turn=1,
        model="gpt-5.4-nano",
        reasoning_effort=None,
        instructions=long_instructions,
        input_text="turn 1",
    )
    log.log_llm_request(
        turn=2,
        model="gpt-5.4-nano",
        reasoning_effort=None,
        instructions=long_instructions,
        input_text="turn 2",
    )
    log.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    requests = [r for r in records if r["event"] == "llm_request"]
    assert len(requests) == 2
    assert "instructions" in requests[0]
    assert requests[0]["instructions_id"] == instructions_digest(long_instructions)
    assert "instructions" not in requests[1]
    assert requests[1]["instructions_id"] == requests[0]["instructions_id"]


def test_log_llm_request_logs_instructions_when_prompt_changes(tmp_path: Path):
    path = tmp_path / "change.jsonl"
    log = SessionLog(path)
    log.log_llm_request(
        turn=1,
        model="gpt-5.4-nano",
        reasoning_effort=None,
        instructions="v1",
        input_text="a",
    )
    log.log_llm_request(
        turn=2,
        model="gpt-5.4-nano",
        reasoning_effort=None,
        instructions="v2",
        input_text="b",
    )
    log.close()

    requests = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line)["event"] == "llm_request"
    ]
    assert requests[0]["instructions"] == "v1"
    assert requests[1]["instructions"] == "v2"


def test_session_log_disabled(monkeypatch):
    monkeypatch.setenv("NEO_SESSION_LOG", "0")
    from settings import get_settings

    get_settings.cache_clear()
    with open_session_log() as log:
        assert log is None
