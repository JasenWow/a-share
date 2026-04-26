"""Tests for technicals agent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from big_a.models.hedge_fund.agents.technicals import technicals_agent
from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState


class TestTechnicalsAgent:
    """Tests for technicals_agent function."""

    @patch("big_a.models.hedge_fund.agents.technicals.call_llm")
    @patch("big_a.models.hedge_fund.agents.technicals.get_prices")
    def test_bullish_signal_generation(self, mock_get_prices: MagicMock, mock_call_llm: MagicMock) -> None:
        """Test that bullish indicators generate bullish signal."""
        prices = pd.DataFrame(
            {
                "open": [100.0 + i * 0.5 for i in range(60)],
                "high": [101.0 + i * 0.5 for i in range(60)],
                "low": [99.0 + i * 0.5 for i in range(60)],
                "close": [100.0 + i * 0.5 for i in range(60)],
                "volume": [1000000 + i * 10000 for i in range(60)],
            },
            index=pd.MultiIndex.from_tuples(
                [("SH600000", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(60)],
                names=["instrument", "datetime"],
            ),
        )
        mock_get_prices.return_value = prices

        expected_signal = AgentSignal(
            agent_name="technicals_agent",
            signal="bullish",
            confidence=0.85,
            reasoning="Strong uptrend with EMA8 > EMA21 > EMA55, RSI above 50, MACD histogram positive, indicating bullish momentum in A-share market.",
        )
        mock_call_llm.return_value = expected_signal

        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-03-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        assert "analyst_signals" in result["data"]
        assert "technicals_agent" in result["data"]["analyst_signals"]
        signal_data = result["data"]["analyst_signals"]["technicals_agent"]
        assert signal_data["signal"] == "bullish"
        assert signal_data["confidence"] == 0.85
        assert "bullish" in signal_data["reasoning"].lower()

    @patch("big_a.models.hedge_fund.agents.technicals.call_llm")
    @patch("big_a.models.hedge_fund.agents.technicals.get_prices")
    def test_bearish_signal_generation(self, mock_get_prices: MagicMock, mock_call_llm: MagicMock) -> None:
        """Test that bearish indicators generate bearish signal."""
        prices = pd.DataFrame(
            {
                "open": [130.0 - i * 0.5 for i in range(60)],
                "high": [131.0 - i * 0.5 for i in range(60)],
                "low": [129.0 - i * 0.5 for i in range(60)],
                "close": [130.0 - i * 0.5 for i in range(60)],
                "volume": [1000000 + i * 5000 for i in range(60)],
            },
            index=pd.MultiIndex.from_tuples(
                [("SH600000", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(60)],
                names=["instrument", "datetime"],
            ),
        )
        mock_get_prices.return_value = prices

        expected_signal = AgentSignal(
            agent_name="technicals_agent",
            signal="bearish",
            confidence=0.78,
            reasoning="Downtrend with EMA8 < EMA21 < EMA55, RSI declining below 50, MACD histogram negative, indicating bearish pressure in A-share market.",
        )
        mock_call_llm.return_value = expected_signal

        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-03-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        signal_data = result["data"]["analyst_signals"]["technicals_agent"]
        assert signal_data["signal"] == "bearish"
        assert signal_data["confidence"] == 0.78

    @patch("big_a.models.hedge_fund.agents.technicals.get_prices")
    def test_empty_data_returns_neutral(self, mock_get_prices: MagicMock) -> None:
        """Test that empty price data returns neutral signal."""
        mock_get_prices.return_value = pd.DataFrame()

        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-03-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        signal_data = result["data"]["analyst_signals"]["technicals_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0
        assert "No price data available" in signal_data["reasoning"]

    @patch("big_a.models.hedge_fund.agents.technicals.get_prices")
    def test_insufficient_data_returns_neutral(self, mock_get_prices: MagicMock) -> None:
        """Test that insufficient data points returns neutral signal."""
        prices = pd.DataFrame(
            {
                "open": [100.0 + i * 0.5 for i in range(30)],
                "high": [101.0 + i * 0.5 for i in range(30)],
                "low": [99.0 + i * 0.5 for i in range(30)],
                "close": [100.0 + i * 0.5 for i in range(30)],
                "volume": [1000000 + i * 10000 for i in range(30)],
            },
            index=pd.MultiIndex.from_tuples(
                [("SH600000", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(30)],
                names=["instrument", "datetime"],
            ),
        )
        mock_get_prices.return_value = prices

        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-02-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        signal_data = result["data"]["analyst_signals"]["technicals_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0
        assert "Insufficient data points" in signal_data["reasoning"]

    def test_missing_ticker_returns_neutral(self) -> None:
        """Test that missing ticker returns neutral signal."""
        state: HedgeFundState = {
            "messages": [],
            "data": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        signal_data = result["data"]["analyst_signals"]["technicals_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0
        assert "Missing required data fields" in signal_data["reasoning"]

    def test_missing_start_date_returns_neutral(self) -> None:
        """Test that missing start_date returns neutral signal."""
        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "end_date": "2024-03-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        signal_data = result["data"]["analyst_signals"]["technicals_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0

    def test_missing_end_date_returns_neutral(self) -> None:
        """Test that missing end_date returns neutral signal."""
        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "start_date": "2024-01-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        signal_data = result["data"]["analyst_signals"]["technicals_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0

    @patch("big_a.models.hedge_fund.agents.technicals.get_prices")
    def test_exception_handling_returns_neutral(self, mock_get_prices: MagicMock) -> None:
        """Test that exceptions are caught and return neutral signal."""
        mock_get_prices.side_effect = Exception("Qlib connection error")

        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-03-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        signal_data = result["data"]["analyst_signals"]["technicals_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0
        assert "Error during technical analysis" in signal_data["reasoning"]

    @patch("big_a.models.hedge_fund.agents.technicals.call_llm")
    @patch("big_a.models.hedge_fund.agents.technicals.get_prices")
    def test_signal_stored_in_correct_location(self, mock_get_prices: MagicMock, mock_call_llm: MagicMock) -> None:
        """Test that signal is stored in the correct location in state."""
        prices = pd.DataFrame(
            {
                "open": [100.0 + i * 0.5 for i in range(60)],
                "high": [101.0 + i * 0.5 for i in range(60)],
                "low": [99.0 + i * 0.5 for i in range(60)],
                "close": [100.0 + i * 0.5 for i in range(60)],
                "volume": [1000000 + i * 10000 for i in range(60)],
            },
            index=pd.MultiIndex.from_tuples(
                [("SH600000", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(60)],
                names=["instrument", "datetime"],
            ),
        )
        mock_get_prices.return_value = prices

        expected_signal = AgentSignal(
            agent_name="technicals_agent",
            signal="neutral",
            confidence=0.5,
            reasoning="Mixed signals with no clear direction",
        )
        mock_call_llm.return_value = expected_signal

        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-03-01"},
            "metadata": {},
        }

        result = technicals_agent(state)

        assert "data" in result
        assert "analyst_signals" in result["data"]
        assert "technicals_agent" in result["data"]["analyst_signals"]
        assert result["data"]["analyst_signals"]["technicals_agent"]["agent_name"] == "technicals_agent"

    @patch("big_a.models.hedge_fund.agents.technicals.call_llm")
    @patch("big_a.models.hedge_fund.agents.technicals.get_prices")
    def test_passes_config_to_llm(self, mock_get_prices: MagicMock, mock_call_llm: MagicMock) -> None:
        """Test that config from metadata is passed to LLM."""
        prices = pd.DataFrame(
            {
                "open": [100.0 + i * 0.5 for i in range(60)],
                "high": [101.0 + i * 0.5 for i in range(60)],
                "low": [99.0 + i * 0.5 for i in range(60)],
                "close": [100.0 + i * 0.5 for i in range(60)],
                "volume": [1000000 + i * 10000 for i in range(60)],
            },
            index=pd.MultiIndex.from_tuples(
                [("SH600000", pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)) for i in range(60)],
                names=["instrument", "datetime"],
            ),
        )
        mock_get_prices.return_value = prices

        expected_signal = AgentSignal(
            agent_name="technicals_agent",
            signal="neutral",
            confidence=0.5,
            reasoning="Test",
        )
        mock_call_llm.return_value = expected_signal

        config = {"llm": {"model": "custom-model"}}
        state: HedgeFundState = {
            "messages": [],
            "data": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-03-01"},
            "metadata": {"config": config},
        }

        result = technicals_agent(state)

        mock_call_llm.assert_called_once()
        call_args = mock_call_llm.call_args
        assert call_args[0][2] == config


class TestCalculateEMA:
    """Tests for _calculate_ema function."""

    def test_calculate_ema_returns_series(self) -> None:
        """Test that _calculate_ema returns a pandas Series."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_ema

        data = [100.0, 101.0, 102.0, 103.0, 104.0]
        result = _calculate_ema(data, 3)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_ema_values_decrease_with_smoothing(self) -> None:
        """Test that EMA smooths data appropriately."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_ema

        data = [100.0, 110.0, 120.0, 130.0, 140.0]
        result = _calculate_ema(data, 3)

        assert result.iloc[-1] > result.iloc[0]
        assert result.iloc[-1] < data[-1]


class TestCalculateRSI:
    """Tests for _calculate_rsi function."""

    def test_calculate_rsi_returns_series(self) -> None:
        """Test that _calculate_rsi returns a pandas Series."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_rsi

        data = [100.0 + i * 0.5 for i in range(30)]
        result = _calculate_rsi(data, 14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(data)

    def test_calculate_rsi_values_in_range(self) -> None:
        """Test that RSI values are between 0 and 100."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_rsi

        data = [100.0 + i * 0.5 + (i % 3 - 1) for i in range(50)]
        result = _calculate_rsi(data, 14)

        valid_values = result.dropna()
        assert all(0 <= v <= 100 for v in valid_values)


class TestCalculateMACD:
    """Tests for _calculate_macd function."""

    def test_calculate_macd_returns_three_series(self) -> None:
        """Test that _calculate_macd returns three series."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_macd

        data = [100.0 + i * 0.5 for i in range(30)]
        macd, signal, hist = _calculate_macd(data)

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(data)
        assert len(signal) == len(data)
        assert len(hist) == len(data)

    def test_calculate_macd_histogram_equals_macd_minus_signal(self) -> None:
        """Test that MACD histogram equals MACD line minus signal line."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_macd

        data = [100.0 + i * 0.5 for i in range(50)]
        macd, signal, hist = _calculate_macd(data)

        pd.testing.assert_series_equal(hist, macd - signal)


class TestCalculateBollingerBands:
    """Tests for _calculate_bollinger_bands function."""

    def test_calculate_bollinger_bands_returns_three_series(self) -> None:
        """Test that _calculate_bollinger_bands returns three series."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_bollinger_bands

        data = [100.0 + i * 0.5 + (i % 5 - 2) for i in range(30)]
        upper, middle, lower = _calculate_bollinger_bands(data, 20, 2)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_calculate_bollinger_bands_upper_above_middle_above_lower(self) -> None:
        """Test that upper band > middle band > lower band."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_bollinger_bands

        data = [100.0 + i * 0.5 + (i % 5 - 2) for i in range(50)]
        upper, middle, lower = _calculate_bollinger_bands(data, 20, 2)

        valid_idx = ~upper.isna() & ~middle.isna() & ~lower.isna()
        assert all(upper[valid_idx] >= middle[valid_idx])
        assert all(middle[valid_idx] >= lower[valid_idx])


class TestCalculateATR:
    """Tests for _calculate_atr function."""

    def test_calculate_atr_returns_series(self) -> None:
        """Test that _calculate_atr returns a pandas Series."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_atr

        high = [101.0 + i * 0.5 for i in range(30)]
        low = [99.0 + i * 0.5 for i in range(30)]
        close = [100.0 + i * 0.5 for i in range(30)]

        result = _calculate_atr(high, low, close, 14)

        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_calculate_atr_values_non_negative(self) -> None:
        """Test that ATR values are non-negative."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_atr

        high = [101.0 + i * 0.5 for i in range(50)]
        low = [99.0 + i * 0.5 for i in range(50)]
        close = [100.0 + i * 0.5 for i in range(50)]

        result = _calculate_atr(high, low, close, 14)

        valid_values = result.dropna()
        assert all(v >= 0 for v in valid_values)


class TestCalculateADX:
    """Tests for _calculate_adx function."""

    def test_calculate_adx_returns_three_series(self) -> None:
        """Test that _calculate_adx returns three series."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_adx

        high = [101.0 + i * 0.5 + (i % 3) for i in range(30)]
        low = [99.0 + i * 0.5 - (i % 3) for i in range(30)]
        close = [100.0 + i * 0.5 for i in range(30)]

        adx, plus_di, minus_di = _calculate_adx(high, low, close, 14)

        assert isinstance(adx, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)

    def test_calculate_adx_values_non_negative(self) -> None:
        """Test that ADX values are non-negative."""
        from big_a.models.hedge_fund.agents.technicals import _calculate_adx

        high = [101.0 + i * 0.5 + (i % 3) for i in range(50)]
        low = [99.0 + i * 0.5 - (i % 3) for i in range(50)]
        close = [100.0 + i * 0.5 for i in range(50)]

        adx, plus_di, minus_di = _calculate_adx(high, low, close, 14)

        valid_adx = adx.dropna()
        valid_plus = plus_di.dropna()
        valid_minus = minus_di.dropna()

        assert all(v >= 0 for v in valid_adx)
        assert all(v >= 0 for v in valid_plus)
        assert all(v >= 0 for v in valid_minus)
