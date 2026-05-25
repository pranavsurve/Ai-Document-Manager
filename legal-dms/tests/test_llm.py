from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from legal_dms.common.llm import LlmSchemaError, OpenAiClient


class SimpleSchema(BaseModel):
    name: str


def test_openai_complete_returns_string():
    client = OpenAiClient("gpt-4o", "test-key")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Hello world"))]

    with patch("legal_dms.common.llm.openai.ChatCompletion.create", return_value=mock_response):
        output = client.complete("system", "user")

    assert output == "Hello world"


def test_openai_complete_with_schema_parses():
    client = OpenAiClient("gpt-4o", "test-key")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"name": "Alice"}'))]

    with patch("legal_dms.common.llm.openai.ChatCompletion.create", return_value=mock_response):
        result = client.complete("system", "user", response_schema=SimpleSchema)

    assert isinstance(result, SimpleSchema)
    assert result.name == "Alice"


def test_openai_complete_with_schema_retries_then_raises():
    client = OpenAiClient("gpt-4o", "test-key")
    invalid_response = Mock()
    invalid_response.choices = [Mock(message=Mock(content="not valid json"))]
    second_invalid_response = Mock()
    second_invalid_response.choices = [Mock(message=Mock(content="still invalid"))]

    with patch(
        "legal_dms.common.llm.openai.ChatCompletion.create",
        side_effect=[invalid_response, second_invalid_response],
    ):
        with pytest.raises(LlmSchemaError) as exc_info:
            client.complete("system", "user", response_schema=SimpleSchema)

    assert len(exc_info.value.attempts) == 2
    assert exc_info.value.attempts[0]["output"] == "not valid json"
    assert exc_info.value.attempts[1]["output"] == "still invalid"


def test_openai_stream_yields_tokens_in_order():
    client = OpenAiClient("gpt-4o", "test-key")

    def mock_stream(*args, **kwargs):
        yield Mock(choices=[Mock(delta={"content": "Hello"})])
        yield Mock(choices=[Mock(delta={"content": " "})])
        yield Mock(choices=[Mock(delta={"content": "world"})])

    with patch("legal_dms.common.llm.openai.ChatCompletion.create", side_effect=mock_stream):
        tokens = list(client.stream("system", "user"))

    assert tokens == ["Hello", " ", "world"]