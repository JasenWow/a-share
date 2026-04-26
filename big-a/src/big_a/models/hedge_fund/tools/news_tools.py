"""News fetching tools for hedge fund agents — replaces financial-datasets API."""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from loguru import logger


@tool
def zhipu_web_search(query: str) -> str:
    """Use ZhipuAI's web search tool to search for information.

    This is a placeholder implementation. In the future, this would use
    ZhipuAI's actual web search capability via tool calling.

    Args:
        query: Search query string

    Returns:
        Search results as a string
    """
    logger.warning(f"ZhipuAI web search called with query: {query} (not yet implemented)")
    return ""


def get_stock_news(
    ticker: str,
    start_date: str,
    end_date: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Fetch stock-related news for analysis.

    Args:
        ticker: Stock ticker symbol (e.g., "SH600000" for A-shares)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Configuration dict containing news settings

    Returns:
        List of news articles, each with title, content, date, source.
        Returns empty list if news is unavailable.
    """
    cfg = config or {}
    news_cfg = cfg.get("news", {})
    source = news_cfg.get("source", "placeholder")

    logger.info(f"Fetching news for {ticker} from {start_date} to {end_date} using source: {source}")

    try:
        if source == "zhipu_websearch":
            # TODO: Implement ZhipuAI web search integration using zhipu_web_search tool
            logger.warning("ZhipuAI web search not yet implemented, returning empty list")
            return []
        else:
            logger.warning(f"News source '{source}' not implemented, returning empty list")
            return []
    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        # Must NOT crash — return empty list
        return []
