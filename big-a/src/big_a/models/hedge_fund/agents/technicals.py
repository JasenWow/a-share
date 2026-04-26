"""Technicals agent for technical analysis of A-share stocks."""
from __future__ import annotations

from typing import Any

import pandas as pd
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_prices
from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState


def technicals_agent(state: HedgeFundState) -> HedgeFundState:
    """Analyze technical indicators and generate trading signal.

    Computes trend, momentum, mean reversion, and volatility signals
    using local pandas calculations, then uses LLM to generate final signal.

    Args:
        state: HedgeFundState containing ticker, start_date, end_date

    Returns:
        Updated HedgeFundState with technicals_agent signal in
        state["data"]["analyst_signals"]["technicals_agent"]
    """
    data = state.get("data", {})
    ticker = data.get("ticker")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not ticker or not start_date or not end_date:
        neutral_signal = AgentSignal(
            agent_name="technicals_agent",
            signal="neutral",
            confidence=0.0,
            reasoning="Missing required data fields (ticker, start_date, or end_date)",
        )
        state.setdefault("data", {}).setdefault("analyst_signals", {})["technicals_agent"] = neutral_signal.model_dump()
        return state

    try:
        prices_df = get_prices([ticker], start_date, end_date)
        if prices_df.empty:
            neutral_signal = AgentSignal(
                agent_name="technicals_agent",
                signal="neutral",
                confidence=0.0,
                reasoning=f"No price data available for {ticker} between {start_date} and {end_date}",
            )
            state.setdefault("data", {}).setdefault("analyst_signals", {})["technicals_agent"] = neutral_signal.model_dump()
            return state

        close = prices_df["close"].values
        high = prices_df["high"].values
        low = prices_df["low"].values
        volume = prices_df["volume"].values

        if len(close) < 55:
            neutral_signal = AgentSignal(
                agent_name="technicals_agent",
                signal="neutral",
                confidence=0.0,
                reasoning=f"Insufficient data points ({len(close)}) for technical analysis (need at least 55)",
            )
            state.setdefault("data", {}).setdefault("analyst_signals", {})["technicals_agent"] = neutral_signal.model_dump()
            return state

        ema_8 = _calculate_ema(close, 8)
        ema_21 = _calculate_ema(close, 21)
        ema_55 = _calculate_ema(close, 55)
        rsi_14 = _calculate_rsi(close, 14)
        macd, macd_signal, macd_hist = _calculate_macd(close)
        bb_upper, bb_middle, bb_lower = _calculate_bollinger_bands(close, 20, 2)
        adx, plus_di, minus_di = _calculate_adx(high, low, close, 14)
        atr = _calculate_atr(high, low, close, 14)

        latest = {
            "close": float(close[-1]),
            "ema_8": float(ema_8.iloc[-1]),
            "ema_21": float(ema_21.iloc[-1]),
            "ema_55": float(ema_55.iloc[-1]),
            "rsi_14": float(rsi_14.iloc[-1]),
            "macd": float(macd.iloc[-1]),
            "macd_signal": float(macd_signal.iloc[-1]),
            "macd_hist": float(macd_hist.iloc[-1]),
            "bb_upper": float(bb_upper.iloc[-1]),
            "bb_middle": float(bb_middle.iloc[-1]),
            "bb_lower": float(bb_lower.iloc[-1]),
            "adx": float(adx.iloc[-1]),
            "plus_di": float(plus_di.iloc[-1]),
            "minus_di": float(minus_di.iloc[-1]),
            "atr": float(atr.iloc[-1]),
            "volume": float(volume[-1]) if len(volume) > 0 else 0.0,
        }

        trend_signal = "bullish" if ema_8.iloc[-1] > ema_21.iloc[-1] > ema_55.iloc[-1] else "bearish" if ema_8.iloc[-1] < ema_21.iloc[-1] < ema_55.iloc[-1] else "neutral"
        momentum_signal = "bullish" if macd_hist.iloc[-1] > 0 else "bearish" if macd_hist.iloc[-1] < 0 else "neutral"
        mean_reversion_signal = "bullish" if rsi_14.iloc[-1] < 30 else "bearish" if rsi_14.iloc[-1] > 70 else "neutral"
        volatility_signal = "high" if atr.iloc[-1] > close[-1] * 0.03 else "normal"

        prompt = f"""你是一名专业的A股技术分析专家。请分析以下技术指标，为股票{ticker}（{start_date}至{end_date}期间）生成交易信号。

当前价格数据:
- 收盘价: {latest['close']:.2f}
- 成交量: {latest['volume']:.0f}

趋势指标:
- EMA8: {latest['ema_8']:.2f}
- EMA21: {latest['ema_21']:.2f}
- EMA55: {latest['ema_55']:.2f}
- 趋势判断: {trend_signal} (EMA8 {'>' if ema_8.iloc[-1] > ema_21.iloc[-1] else '<'} EMA21 {'>' if ema_21.iloc[-1] > ema_55.iloc[-1] else '<'} EMA55)

动量指标:
- MACD: {latest['macd']:.4f}
- MACD信号线: {latest['macd_signal']:.4f}
- MACD柱状图: {latest['macd_hist']:.4f}
- 动量判断: {momentum_signal}

均值回归指标:
- RSI(14): {latest['rsi_14']:.2f}
- 布林带上轨: {latest['bb_upper']:.2f}
- 布林带中轨: {latest['bb_middle']:.2f}
- 布林带下轨: {latest['bb_lower']:.2f}
- 价格位置: {((latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100):.1f}% (布林带内)
- 均值回归判断: {mean_reversion_signal}

波动率指标:
- ADX(14): {latest['adx']:.2f} (趋势强度，>25为强趋势)
- +DI: {latest['plus_di']:.2f}
- -DI: {latest['minus_di']:.2f}
- ATR(14): {latest['atr']:.2f}
- 波动率判断: {volatility_signal}

请基于沪深300指数和A股市场的特点，综合以上指标给出:
1. 交易信号 (bullish/bearish/neutral)
2. 置信度 (0.0-1.0)
3. 详细分析理由

注意事项:
- A股市场受政策影响较大，需结合当前技术面和市场情绪
- 沪深300指数可作为大盘参考，个股需对比指数表现
- 上交所/深交所股票的交易规则和流动性特点需考虑
- 成交量配合度很重要，无量上涨需谨慎
"""

        config = state.get("metadata", {}).get("config")
        signal = call_llm(prompt, AgentSignal, config)

        state.setdefault("data", {}).setdefault("analyst_signals", {})["technicals_agent"] = signal.model_dump()
        return state

    except Exception as e:
        neutral_signal = AgentSignal(
            agent_name="technicals_agent",
            signal="neutral",
            confidence=0.0,
            reasoning=f"Error during technical analysis: {str(e)}",
        )
        state.setdefault("data", {}).setdefault("analyst_signals", {})["technicals_agent"] = neutral_signal.model_dump()
        return state


def _calculate_ema(data: pd.Series | list[float] | Any, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return pd.Series(data).ewm(span=period, adjust=False).mean()


def _calculate_rsi(data: pd.Series | list[float] | Any, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = pd.Series(data).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _calculate_macd(data: pd.Series | list[float] | Any, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = _calculate_ema(data, fast)
    ema_slow = _calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calculate_bollinger_bands(data: pd.Series | list[float] | Any, period: int = 20, std_dev: float = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = pd.Series(data).rolling(window=period).mean()
    std = pd.Series(data).rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def _calculate_atr(high: pd.Series | list[float] | Any, low: pd.Series | list[float] | Any, close: pd.Series | list[float] | Any, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.fillna(0)


def _calculate_adx(high: pd.Series | list[float] | Any, low: pd.Series | list[float] | Any, close: pd.Series | list[float] | Any, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Average Directional Index (ADX)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    prev_high = high_s.shift(1)
    prev_low = low_s.shift(1)
    prev_close = close_s.shift(1)
    
    plus_dm = high_s - prev_high
    minus_dm = prev_low - low_s
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
