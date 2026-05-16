"""Tests for LLM client wiring."""

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
