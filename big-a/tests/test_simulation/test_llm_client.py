from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

import anthropic
import pytest

from big_a.llm.client import LLMClient, LLMError


class _TestOutput(BaseModel):
    answer: str
    score: float


class TestLLMClient:
    def test_chat_success(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="AAPL looks bullish")]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_instance = mock_anthropic.return_value
            mock_instance.messages.create.return_value = mock_response

            client = LLMClient(api_key="test-key")
            result = client.chat("You are a helpful assistant.", "Is AAPL bullish?")

            assert result == "AAPL looks bullish"
            mock_instance.messages.create.assert_called_once()

    def test_chat_structured_success(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"answer": "Buy", "score": 0.85}')]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_instance = mock_anthropic.return_value
            mock_instance.messages.create.return_value = mock_response

            client = LLMClient(api_key="test-key")
            result = client.chat_structured(
                "Analyze stocks.",
                "What is the recommendation for AAPL?",
                _TestOutput,
            )

            assert isinstance(result, _TestOutput)
            assert result.answer == "Buy"
            assert result.score == 0.85

    def test_timeout_retry_success(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Success after retry")]

        with patch("anthropic.Anthropic") as mock_anthropic, patch("big_a.llm.client.time.sleep"):
            mock_instance = mock_anthropic.return_value
            mock_instance.messages.create.side_effect = [
                anthropic.APITimeoutError("timeout"),
                anthropic.APITimeoutError("timeout"),
                mock_response,
            ]

            client = LLMClient(api_key="test-key", max_retries=3)
            result = client.chat("System prompt", "User message")

            assert result == "Success after retry"
            assert mock_instance.messages.create.call_count == 3

    def test_all_retries_failed(self):
        with patch("anthropic.Anthropic") as mock_anthropic, patch("big_a.llm.client.time.sleep"):
            mock_instance = mock_anthropic.return_value
            mock_instance.messages.create.side_effect = anthropic.APITimeoutError("timeout")

            client = LLMClient(api_key="test-key", max_retries=3)

            with pytest.raises(LLMError) as exc_info:
                client.chat("System prompt", "User message")

            assert "failed after 3 retries" in str(exc_info.value)
            assert mock_instance.messages.create.call_count == 3

    def test_api_error_retry(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Recovered after API error")]

        with patch("anthropic.Anthropic") as mock_anthropic, patch("big_a.llm.client.time.sleep"):
            mock_instance = mock_anthropic.return_value
            mock_instance.messages.create.side_effect = [
                anthropic.APIError(message="Server error", request=MagicMock(), body=None),
                mock_response,
            ]

            client = LLMClient(api_key="test-key", max_retries=3)
            result = client.chat("System prompt", "User message")

            assert result == "Recovered after API error"
            assert mock_instance.messages.create.call_count == 2

    def test_chat_structured_json_in_markdown(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"answer": "Hold", "score": 0.5}\n```')]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_instance = mock_anthropic.return_value
            mock_instance.messages.create.return_value = mock_response

            client = LLMClient(api_key="test-key")
            result = client.chat_structured(
                "Analyze.",
                "What is the score for AAPL?",
                _TestOutput,
            )

            assert isinstance(result, _TestOutput)
            assert result.answer == "Hold"
            assert result.score == 0.5

    def test_chat_structured_invalid_json(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not JSON at all")]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_instance = mock_anthropic.return_value
            mock_instance.messages.create.return_value = mock_response

            client = LLMClient(api_key="test-key")

            with pytest.raises(LLMError) as exc_info:
                client.chat_structured(
                    "Analyze.",
                    "Give me JSON.",
                    _TestOutput,
                )

            assert "Failed to parse LLM response" in str(exc_info.value)
            assert "This is not JSON" in str(exc_info.value)