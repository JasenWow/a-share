"""Tests for LLM abstraction layer for ZhipuAI."""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from big_a.models.hedge_fund.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, call_llm, create_llm


class SampleResponse(BaseModel):
    """Sample response model for testing."""

    value: str
    count: int


class TestCreateLLM:
    """Tests for create_llm function."""

    def test_create_llm_with_config(self) -> None:
        """Test that ChatOpenAI is created with correct config."""
        config = {
            "llm": {
                "model": "glm-4-plus",
                "base_url": "https://custom.api.com/v1/",
                "api_key": "test-key-123",
                "temperature": 0.5,
                "max_tokens": 4000,
            }
        }

        with patch("big_a.models.hedge_fund.llm.ChatOpenAI") as mock_chat_openai:
            mock_instance = MagicMock(spec=BaseChatModel)
            mock_chat_openai.return_value = mock_instance

            result = create_llm(config)

            mock_chat_openai.assert_called_once_with(
                model="glm-4-plus",
                base_url="https://custom.api.com/v1/",
                api_key="test-key-123",
                temperature=0.5,
                max_tokens=4000,
                timeout=60,
                request_timeout=60,
            )
            assert result is mock_instance

    def test_create_llm_no_api_key_raises(self) -> None:
        """Test that ValueError is raised when no API key is provided."""
        # Ensure no env var and no config key
        with patch.dict(os.environ, {"ZHIPU_API_KEY": ""}, clear=True):
            config: dict[str, Any] = {"llm": {}}

            with pytest.raises(ValueError, match="ZHIPU_API_KEY environment variable or llm.api_key config is required"):
                create_llm(config)

    def test_create_llm_from_env_var(self) -> None:
        """Test that API key is read from environment variable."""
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "env-api-key"}, clear=True):
            config: dict[str, Any] = {"llm": {}}

            with patch("big_a.models.hedge_fund.llm.ChatOpenAI") as mock_chat_openai:
                mock_instance = MagicMock(spec=BaseChatModel)
                mock_chat_openai.return_value = mock_instance

                result = create_llm(config)

                # Check that the API key from env var is used
                call_kwargs = mock_chat_openai.call_args.kwargs
                assert call_kwargs["api_key"] == "env-api-key"

    def test_create_llm_default_values(self) -> None:
        """Test that default model and base_url are used when not specified."""
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "test-key"}, clear=True):
            config: dict[str, Any] = {"llm": {"api_key": "config-key"}}

            with patch("big_a.models.hedge_fund.llm.ChatOpenAI") as mock_chat_openai:
                mock_instance = MagicMock(spec=BaseChatModel)
                mock_chat_openai.return_value = mock_instance

                result = create_llm(config)

                # Check that defaults are used
                call_kwargs = mock_chat_openai.call_args.kwargs
                assert call_kwargs["model"] == DEFAULT_MODEL
                assert call_kwargs["base_url"] == DEFAULT_BASE_URL
                assert call_kwargs["temperature"] == 0.1
                assert call_kwargs["max_tokens"] == 2000

    def test_create_llm_env_var_takes_precedence(self) -> None:
        """Test that API key from env var takes precedence over config."""
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "env-key"}, clear=True):
            config = {"llm": {"api_key": "config-key"}}

            with patch("big_a.models.hedge_fund.llm.ChatOpenAI") as mock_chat_openai:
                mock_instance = MagicMock(spec=BaseChatModel)
                mock_chat_openai.return_value = mock_instance

                result = create_llm(config)

                # Check that env var key is used (takes precedence)
                call_kwargs = mock_chat_openai.call_args.kwargs
                assert call_kwargs["api_key"] == "env-key"


class TestCallLLM:
    """Tests for call_llm function."""

    def test_call_llm_structured_output(self) -> None:
        """Test that call_llm uses function_calling method for structured output."""
        config = {"llm": {"api_key": "test-key"}}
        prompt = "Extract test data"
        expected_response = SampleResponse(value="test", count=42)

        with patch("big_a.models.hedge_fund.llm.ChatOpenAI") as mock_chat_openai:
            # Setup mock LLM instance
            mock_llm = MagicMock(spec=BaseChatModel)
            mock_chat_openai.return_value = mock_llm

            # Setup mock structured output
            mock_structured = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured
            mock_structured.invoke.return_value = expected_response

            result = call_llm(prompt, SampleResponse, config)

            # Verify function_calling method is used
            mock_llm.with_structured_output.assert_called_once_with(SampleResponse, method="function_calling")
            mock_structured.invoke.assert_called_once_with(prompt)
            assert result == expected_response

    def test_call_llm_passes_config_to_create_llm(self) -> None:
        """Test that config is properly passed to create_llm."""
        config = {"llm": {"api_key": "test-key", "model": "custom-model"}}
        prompt = "Test prompt"

        with patch("big_a.models.hedge_fund.llm.create_llm") as mock_create_llm:
            mock_llm = MagicMock(spec=BaseChatModel)
            mock_create_llm.return_value = mock_llm

            mock_structured = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured
            mock_structured.invoke.return_value = SampleResponse(value="test", count=1)

            call_llm(prompt, SampleResponse, config)

            # Verify config is passed to create_llm
            mock_create_llm.assert_called_once_with(config)


class TestIntegration:
    """Integration tests (require real API key)."""

    @pytest.mark.integration
    def test_call_llm_real_api(self) -> None:
        """Integration test with real ZhipuAI API."""
        # This test requires ZHIPU_API_KEY to be set
        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            pytest.skip("ZHIPU_API_KEY not set for integration test")

        config = {"llm": {"api_key": api_key}}
        prompt = "Return a simple test response"

        result = call_llm(prompt, SampleResponse, config)

        assert isinstance(result, SampleResponse)
        assert result.value
        assert isinstance(result.count, int)


class TestOpenRouter:
    """Tests for OpenRouter LLM provider support."""

    def test_create_llm_openrouter_with_config(self) -> None:
        """Test that OpenRouter ChatOpenAI is created with correct config."""
        config = {
            "llm": {
                "provider": "openrouter",
                "api_key": "sk-or-test-key",
                "model": "google/gemini-2.0-flash-exp:free",
            }
        }
        with patch("big_a.models.hedge_fund.llm.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock(spec=BaseChatModel)
            mock_chat.return_value = mock_instance
            result = create_llm(config)
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1/"
            assert call_kwargs["model"] == "google/gemini-2.0-flash-exp:free"
            assert call_kwargs["api_key"] == "sk-or-test-key"

    def test_create_llm_openrouter_from_env(self) -> None:
        """Test that OpenRouter API key is read from OPENROUTER_API_KEY env var."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-or-key"}, clear=True):
            config = {"llm": {"provider": "openrouter"}}
            with patch("big_a.models.hedge_fund.llm.ChatOpenAI") as mock_chat:
                mock_chat.return_value = MagicMock(spec=BaseChatModel)
                create_llm(config)
                call_kwargs = mock_chat.call_args.kwargs
                assert call_kwargs["api_key"] == "env-or-key"

    def test_create_llm_openrouter_no_key_raises(self) -> None:
        """Test that missing OpenRouter API key raises ValueError."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=True):
            config = {"llm": {"provider": "openrouter"}}
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                create_llm(config)

    def test_create_llm_openrouter_custom_model(self) -> None:
        """Test that custom model overrides default."""
        config = {"llm": {"provider": "openrouter", "api_key": "test", "model": "meta-llama/llama-3-8b:free"}}
        with patch("big_a.models.hedge_fund.llm.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock(spec=BaseChatModel)
            create_llm(config)
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["model"] == "meta-llama/llama-3-8b:free"
