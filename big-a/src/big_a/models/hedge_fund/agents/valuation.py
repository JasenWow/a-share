"""Valuation analysis agent for relative valuation metrics."""
from __future__ import annotations

from loguru import logger
import pandas as pd
from typing import Any

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_prices, get_technical_indicators


def valuation_agent(state: HedgeFundState) -> HedgeFundState:
    """Analyze stock valuation using relative metrics based on price history.

    Focuses on:
    - Price position relative to moving averages (MA5, MA20, MA60)
    - Price percentile rank in historical range
    - Volume-weighted price analysis
    - Price momentum relative to own history

    Args:
        state: HedgeFundState containing ticker, start_date, end_date in data field

    Returns:
        Updated HedgeFundState with valuation_agent signal in data.analyst_signals
    """
    ticker = state["data"].get("ticker")
    start_date = state["data"].get("start_date")
    end_date = state["data"].get("end_date")

    if not ticker or not start_date or not end_date:
        logger.warning("Missing required data (ticker, start_date, end_date) for valuation analysis")
        signal = AgentSignal(
            agent_name="valuation_agent",
            signal="neutral",
            confidence=0.0,
            reasoning="Missing required data: ticker, start_date, or end_date not provided",
        )
        _store_signal(state, signal)
        return state

    try:
        prices_df = get_prices([ticker], start_date, end_date)
        technical_df = get_technical_indicators([ticker], start_date, end_date)

        if prices_df.empty or technical_df.empty:
            logger.warning(f"No data available for {ticker} in the specified date range")
            signal = AgentSignal(
                agent_name="valuation_agent",
                signal="neutral",
                confidence=0.0,
                reasoning=f"No price or technical data available for {ticker} from {start_date} to {end_date}",
            )
            _store_signal(state, signal)
            return state

        metrics = _compute_valuation_metrics(prices_df, technical_df)

        analysis_summary = _build_analysis_summary(ticker, metrics)

        signal = call_llm(
            prompt=analysis_summary,
            schema=AgentSignal,
            config=None,
        )
        signal.agent_name = "valuation_agent"

        _store_signal(state, signal)

        logger.info(f"Valuation analysis completed for {ticker}: {signal.signal} (confidence: {signal.confidence:.2f})")

    except Exception as e:
        logger.error(f"Error in valuation analysis for {ticker}: {e}")
        signal = AgentSignal(
            agent_name="valuation_agent",
            signal="neutral",
            confidence=0.0,
            reasoning=f"Error during valuation analysis: {str(e)}",
        )
        _store_signal(state, signal)

    return state


def _compute_valuation_metrics(
    prices_df: pd.DataFrame,
    technical_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute relative valuation metrics from price and technical data.

    Args:
        prices_df: DataFrame with OHLCV data (MultiIndex: instrument, datetime)
        technical_df: DataFrame with technical indicators (MultiIndex: instrument, datetime)

    Returns:
        Dictionary of computed metrics
    """
    metrics: dict[str, Any] = {}

    try:
        instrument = prices_df.index.get_level_values(0).unique()[0]
        price_data = prices_df.loc[instrument]
        tech_data = technical_df.loc[instrument]

        if isinstance(price_data, pd.DataFrame):
            price_data = price_data.squeeze()
        if isinstance(tech_data, pd.DataFrame):
            tech_data = tech_data.squeeze()

        latest_close = price_data["close"].iloc[-1] if hasattr(price_data["close"], "iloc") else price_data["close"]
        latest_volume = price_data["volume"].iloc[-1] if hasattr(price_data["volume"], "iloc") else price_data["volume"]

        ma5 = tech_data["Mean($close, 5)"].iloc[-1] if "Mean($close, 5)" in tech_data else None
        ma20 = tech_data["Mean($close, 20)"].iloc[-1] if "Mean($close, 20)" in tech_data else None

        if len(price_data) >= 60:
            ma60 = price_data["close"].rolling(60).mean().iloc[-1]
        else:
            ma60 = None

        if ma5 and ma5 > 0:
            metrics["price_vs_ma5"] = (latest_close - ma5) / ma5 * 100
        else:
            metrics["price_vs_ma5"] = None

        if ma20 and ma20 > 0:
            metrics["price_vs_ma20"] = (latest_close - ma20) / ma20 * 100
        else:
            metrics["price_vs_ma20"] = None

        if ma60 and ma60 > 0:
            metrics["price_vs_ma60"] = (latest_close - ma60) / ma60 * 100
        else:
            metrics["price_vs_ma60"] = None

        if len(price_data) >= 20:
            min_price = price_data["close"].min()
            max_price = price_data["close"].max()
            if max_price > min_price:
                metrics["price_percentile"] = (latest_close - min_price) / (max_price - min_price) * 100
            else:
                metrics["price_percentile"] = 50.0
        else:
            metrics["price_percentile"] = None

        if latest_volume and latest_volume > 0:
            volume_weighted_price = (price_data["close"] * price_data["volume"]).sum() / price_data["volume"].sum()
            metrics["vwap"] = volume_weighted_price
            metrics["price_vs_vwap"] = (latest_close - volume_weighted_price) / volume_weighted_price * 100
        else:
            metrics["vwap"] = None
            metrics["price_vs_vwap"] = None

        if len(price_data) >= 20:
            short_term_avg = price_data["close"].tail(5).mean()
            long_term_avg = price_data["close"].head(len(price_data) - 5).mean()
            if long_term_avg and long_term_avg > 0:
                metrics["momentum_ratio"] = (short_term_avg - long_term_avg) / long_term_avg * 100
            else:
                metrics["momentum_ratio"] = None
        else:
            metrics["momentum_ratio"] = None

        if len(price_data) >= 20:
            returns = price_data["close"].pct_change().dropna()
            metrics["volatility_20d"] = returns.tail(20).std() * 100
        else:
            metrics["volatility_20d"] = None

        if len(price_data) >= 20:
            price_5d_ago = price_data["close"].iloc[-6] if len(price_data) >= 6 else price_data["close"].iloc[0]
            price_20d_ago = price_data["close"].iloc[0]
            if price_5d_ago and price_5d_ago > 0:
                metrics["change_5d"] = (latest_close - price_5d_ago) / price_5d_ago * 100
            else:
                metrics["change_5d"] = None
            if price_20d_ago and price_20d_ago > 0:
                metrics["change_20d"] = (latest_close - price_20d_ago) / price_20d_ago * 100
            else:
                metrics["change_20d"] = None
        else:
            metrics["change_5d"] = None
            metrics["change_20d"] = None

        metrics["current_price"] = latest_close
        metrics["volume"] = latest_volume
        metrics["data_points"] = len(price_data)

    except Exception as e:
        logger.error(f"Error computing valuation metrics: {e}")
        metrics["error"] = str(e)

    return metrics


def _build_analysis_summary(ticker: str, metrics: dict[str, Any]) -> str:
    """Build analysis summary string for LLM input.

    Args:
        ticker: Stock ticker symbol
        metrics: Dictionary of computed valuation metrics

    Returns:
        Formatted summary string for LLM analysis
    """

    def _format_float(value: Any, fmt: str = ".2f") -> str:
        """Safely format a float value, returning 'N/A' for None."""
        if value is None:
            return "N/A"
        try:
            return f"{value:{fmt}}"
        except (TypeError, ValueError):
            return str(value)

    price_vs_ma5 = _format_float(metrics.get("price_vs_ma5"))
    price_vs_ma20 = _format_float(metrics.get("price_vs_ma20"))
    price_vs_ma60 = _format_float(metrics.get("price_vs_ma60"))
    price_percentile = _format_float(metrics.get("price_percentile"))
    vwap = _format_float(metrics.get("vwap"))
    price_vs_vwap = _format_float(metrics.get("price_vs_vwap"))
    momentum_ratio = _format_float(metrics.get("momentum_ratio"))
    change_5d = _format_float(metrics.get("change_5d"))
    change_20d = _format_float(metrics.get("change_20d"))
    volatility_20d = _format_float(metrics.get("volatility_20d"))

    summary = f"""You are a valuation analysis expert for A-share (China) stocks. Analyze the relative valuation metrics for {ticker} and provide a trading signal.

**Current Price**: {metrics.get('current_price', 'N/A')}
**Volume**: {metrics.get('volume', 'N/A')}
**Data Points**: {metrics.get('data_points', 'N/A')}

**Price vs Moving Averages**:
- Price vs MA5: {price_vs_ma5}% (positive = above MA, negative = below)
- Price vs MA20: {price_vs_ma20}%
- Price vs MA60: {price_vs_ma60}%

**Historical Position**:
- Price Percentile: {price_percentile}% (0 = lowest historical price, 100 = highest)

**Volume-Weighted Analysis**:
- VWAP: {vwap}
- Price vs VWAP: {price_vs_vwap}%

**Momentum**:
- Momentum Ratio (recent vs historical): {momentum_ratio}%
- 5-Day Change: {change_5d}%
- 20-Day Change: {change_20d}%

**Volatility**: {volatility_20d}% (20-day)

**Analysis Guidelines**:
1. **Bullish Signal**: Consider when price is well below moving averages (e.g., price_vs_ma20 < -5%), price percentile is low (< 30%), and showing upward momentum
2. **Bearish Signal**: Consider when price is well above moving averages (e.g., price_vs_ma20 > 5%), price percentile is high (> 70%), or showing declining momentum
3. **Neutral Signal**: When metrics are mixed or data is insufficient

**Confidence Scoring**:
- 0.9-1.0: Strong agreement across multiple metrics with sufficient data (20+ data points)
- 0.7-0.9: Good agreement but some conflicting signals
- 0.5-0.7: Mixed signals or limited data (10-20 data points)
- 0.3-0.5: Weak signals or very limited data (< 10 data points)
- 0.0-0.3: Insufficient data for reliable analysis

Please provide a trading signal (bullish/bearish/neutral) with confidence and reasoning based on these relative valuation metrics for A-share market context."""

    return summary


def _store_signal(state: HedgeFundState, signal: AgentSignal) -> None:
    """Store the agent signal in the state.

    Args:
        state: HedgeFundState to update
        signal: AgentSignal to store
    """
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["valuation_agent"] = signal.model_dump()
