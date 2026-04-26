"""Sentiment analysis agent for hedge fund workflow."""
from __future__ import annotations

from typing import Any

from loguru import logger

from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.news_tools import get_stock_news
from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState


def sentiment_agent(state: HedgeFundState, config: dict[str, Any] | None = None) -> HedgeFundState:
    """Analyze news sentiment for stock trading signal.

    Args:
        state: HedgeFundState containing current workflow state
        config: Configuration dict (optional)

    Returns:
        Updated HedgeFundState with sentiment signal in data.analyst_signals.sentiment_agent
    """
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    start_date = metadata.get("start_date")
    end_date = metadata.get("end_date")

    if not ticker or not start_date or not end_date:
        logger.warning("Missing ticker or dates in state, returning neutral signal")
        signal = AgentSignal(
            agent_name="sentiment_agent",
            signal="neutral",
            confidence=0.0,
            reasoning="Insufficient data: missing ticker or date range",
        )
        state.setdefault("data", {}).setdefault("analyst_signals", {})["sentiment_agent"] = signal
        return state

    logger.info(f"Sentiment agent analyzing news for {ticker} from {start_date} to {end_date}")

    news_list = get_stock_news(ticker, start_date, end_date, config)

    if not news_list:
        logger.info(f"No news available for {ticker}, returning neutral signal")
        signal = AgentSignal(
            agent_name="sentiment_agent",
            signal="neutral",
            confidence=0.0,
            reasoning="No news available for analysis",
        )
        state.setdefault("data", {}).setdefault("analyst_signals", {})["sentiment_agent"] = signal
        return state

    news_summary = "\n".join(
        [f"- {item.get('title', 'N/A')} ({item.get('date', 'N/A')}): {item.get('content', 'N/A')[:200]}" for item in news_list]
    )

    prompt = f"""You are an expert in Chinese stock market sentiment analysis (A股).

Analyze the following news articles for stock {ticker} and provide a trading signal.

News Articles:
{news_summary}

Consider:
1. Overall sentiment of the news (positive, negative, or neutral)
2. Key themes and factors mentioned
3. Impact expectations on stock price
4. Confidence level in your assessment

Provide your response as a structured trading signal with:
- signal: "bullish" for positive sentiment, "bearish" for negative sentiment, "neutral" for mixed or unclear
- confidence: 0.0 to 1.0 (higher confidence when sentiment is clear and consistent)
- reasoning: brief explanation of your analysis
"""

    try:
        signal = call_llm(prompt, AgentSignal, config)
        signal.agent_name = "sentiment_agent"
        logger.info(f"Sentiment agent generated signal: {signal.signal} (confidence: {signal.confidence})")
    except Exception as e:
        logger.error(f"Error calling LLM for sentiment analysis: {e}")
        signal = AgentSignal(
            agent_name="sentiment_agent",
            signal="neutral",
            confidence=0.0,
            reasoning=f"Error in sentiment analysis: {e}",
        )

    state.setdefault("data", {}).setdefault("analyst_signals", {})["sentiment_agent"] = signal
    return state
