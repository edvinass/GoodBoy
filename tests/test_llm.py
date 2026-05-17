"""Tests for LLM client wiring."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from llm import complete_structured


def test_complete_structured_passes_reasoning_effort():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = '{"action":"task_complete","message":"ok"}'
    mock_client.responses.create.return_value = mock_response

    with patch("llm.get_client", return_value=mock_client), patch(
        "llm.get_settings"
    ) as mock_settings:
        mock_settings.return_value.default_model = "gpt-5.4-mini"
        complete_structured(
            input="task",
            instructions="sys",
            json_schema={"type": "object"},
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )

    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "low"}


def test_complete_structured_passes_hosted_tools():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = '{"action":"task_complete","message":"ok"}'
    mock_client.responses.create.return_value = mock_response

    with patch("llm.get_client", return_value=mock_client), patch(
        "llm.get_settings"
    ) as mock_settings:
        mock_settings.return_value.default_model = "gpt-5.4-mini"
        complete_structured(
            input="task",
            instructions="sys",
            json_schema={"type": "object"},
            model="gpt-5.4-mini",
            tools=["web_search"],
        )

    kwargs = mock_client.responses.create.call_args.kwargs
    assert kwargs["tools"] == [{"type": "web_search"}]


def test_complete_structured_streams_output_text_deltas():
    mock_client = MagicMock()
    mock_final = MagicMock()
    mock_final.output_text = '{"action":"task_complete","message":"ok"}'

    delta_event = MagicMock()
    delta_event.type = "response.output_text.delta"
    delta_event.delta = '{"action":'

    @contextmanager
    def fake_stream(**_kwargs):
        stream = MagicMock()
        stream.__iter__ = lambda self: iter([delta_event])
        stream.get_final_response.return_value = mock_final
        yield stream

    mock_client.responses.stream = fake_stream
    deltas: list[str] = []

    with patch("llm.get_client", return_value=mock_client), patch(
        "llm.get_settings"
    ) as mock_settings:
        mock_settings.return_value.default_model = "gpt-5.4-mini"
        text = complete_structured(
            input="task",
            instructions="sys",
            json_schema={"type": "object"},
            model="gpt-5.4-mini",
            stream=True,
            on_text_delta=deltas.append,
        )

    assert deltas == ['{"action":']
    assert "task_complete" in text
    mock_client.responses.create.assert_not_called()
