from __future__ import annotations

import json
import os
import time
from typing import TypeVar

import anthropic
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when LLM API call fails after all retries."""

    pass


class LLMClient:
    """LLM client using Anthropic SDK to connect to MiniMax API."""

    def __init__(
        self,
        api_base: str = "https://api.minimaxi.com/anthropic",
        api_key: str | None = None,
        model: str = "MiniMax-M2.7",
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            base_url=api_base,
            timeout=timeout,
        )

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a chat message and return the text response."""
        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text  # type: ignore[reportAttributeAccessIssue]
            except anthropic.APITimeoutError as e:
                last_error = e
                logger.warning(f"LLM timeout attempt {attempt + 1}/{self._max_retries}")
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt)
            except anthropic.APIError as e:
                last_error = e
                logger.warning(f"LLM API error attempt {attempt + 1}: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt)

        raise LLMError(f"LLM API call failed after {self._max_retries} retries: {last_error}")

    def chat_structured(self, system_prompt: str, user_message: str, response_model: type[T]) -> T:
        """Send a chat message and parse the response as a Pydantic model.

        Instructs the LLM to return JSON, then parses it with the Pydantic model.
        Raises LLMError if parsing fails.
        """
        json_instruction = f"\n\nYou MUST respond with valid JSON that matches this schema: {response_model.model_json_schema()}"
        enhanced_prompt = system_prompt + json_instruction

        raw_response = self.chat(enhanced_prompt, user_message)

        json_str = raw_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_str.strip())
            return response_model.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            raise LLMError(f"Failed to parse LLM response as {response_model.__name__}: {e}\nRaw: {raw_response[:500]}")