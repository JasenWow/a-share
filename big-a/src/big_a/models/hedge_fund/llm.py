"""LLM abstraction layer for ZhipuAI (智谱 GLM) via OpenAI-compatible API."""
from __future__ import annotations

import os
from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "glm-4-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/"
OPENROUTER_DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"


def create_llm(config: dict[str, Any] | None = None) -> BaseChatModel:
    """Create a ChatOpenAI instance configured for ZhipuAI or OpenRouter."""
    cfg = config or {}
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "zhipu")

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY") or llm_cfg.get("api_key", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable or llm.api_key config is required for OpenRouter provider")
        return ChatOpenAI(
            model=llm_cfg.get("model", OPENROUTER_DEFAULT_MODEL),
            base_url=llm_cfg.get("base_url", OPENROUTER_BASE_URL),
            api_key=api_key,
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 2000),
        )
    else:
        api_key = os.environ.get("ZHIPU_API_KEY") or llm_cfg.get("api_key", "")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY environment variable or llm.api_key config is required")
        return ChatOpenAI(
            model=llm_cfg.get("model", DEFAULT_MODEL),
            base_url=llm_cfg.get("base_url", DEFAULT_BASE_URL),
            api_key=api_key,
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 2000),
        )


def call_llm(prompt: str, schema: type[T], config: dict[str, Any] | None = None) -> T:
    """Call LLM and parse response as structured Pydantic model.

    Uses function_calling method (required by ZhipuAI).
    """
    llm = create_llm(config)
    structured = llm.with_structured_output(schema, method="function_calling")
    result = structured.invoke(prompt)
    return result
