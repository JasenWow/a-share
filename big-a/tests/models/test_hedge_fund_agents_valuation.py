"""Tests for valuation agent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from big_a.models.hedge_fund.agents.valuation import valuation_agent
from big_a.models.hedge_fund.types import AgentSignal


class TestValuationAgent:
    """Tests for valuation_agent function."""

    @patch("big_a.models.hedge_fund.agents.valuation.get_prices")
    @patch("big_a.models.hedge_fund.agents.valuation.get_technical_indicators")
    @patch("big_a.models.hedge_fund.agents.valuation.call_llm")
    def test_valuation_agent_normal_data(self, mock_call_llm, mock_get_tech, mock_get_prices) -> None:
        """Test valuation agent with normal price data."""
        from big_a.models.hedge_fund.types import HedgeFundState

        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = [10.0 + i * 0.1 for i in range(30)]
        volumes = [1000000 + i * 10000 for i in range(30)]

        price_df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": prices,
                "volume": volumes,
            },
            index=pd.MultiIndex.from_tuples([("SH600000", d) for d in dates], names=["instrument", "datetime"]),
        )

        tech_df = pd.DataFrame(
            {
                "Mean($close, 5)": [p + 0.2 for p in prices],
                "Mean($close, 10)": [p + 0.5 for p in prices],
                "Mean($close, 20)": [p + 1.0 for p in prices],
                "Std($change, 20)": [0.02] * 30,
                "Rank($volume)": [50] * 30,
                "Mean($volume, 5)": volumes,
                "Mean($volume, 20)": volumes,
                "$volume / Mean($volume, 20)": [1.0] * 30,
            },
            index=pd.MultiIndex.from_tuples([("SH600000", d) for d in dates], names=["instrument", "datetime"]),
        )

        mock_get_prices.return_value = price_df
        mock_get_tech.return_value = tech_df

        mock_signal = AgentSignal(
            agent_name="valuation_agent",
            signal="bullish",
            confidence=0.85,
            reasoning="Price is below all major moving averages, historical percentile is low (25%), showing upward momentum.",
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "ticker": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
            },
            "metadata": {},
        }

        result = valuation_agent(state)

        assert "analyst_signals" in result["data"]
        assert "valuation_agent" in result["data"]["analyst_signals"]

        signal_dict = result["data"]["analyst_signals"]["valuation_agent"]
        assert signal_dict["agent_name"] == "valuation_agent"
        assert signal_dict["signal"] == "bullish"
        assert signal_dict["confidence"] == 0.85
        assert "below all major moving averages" in signal_dict["reasoning"]

        mock_get_prices.assert_called_once_with(["SH600000"], "2024-01-01", "2024-01-30")
        mock_get_tech.assert_called_once_with(["SH600000"], "2024-01-01", "2024-01-30")
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.valuation.get_prices")
    @patch("big_a.models.hedge_fund.agents.valuation.get_technical_indicators")
    @patch("big_a.models.hedge_fund.agents.valuation.call_llm")
    def test_valuation_agent_undervalued_signal(self, mock_call_llm, mock_get_tech, mock_get_prices) -> None:
        """Test valuation agent detects undervalued condition (low percentile)."""
        from big_a.models.hedge_fund.types import HedgeFundState

        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = [10.0 + i * 0.2 for i in range(30)]
        volumes = [1000000 + i * 10000 for i in range(30)]

        price_df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": prices,
                "volume": volumes,
            },
            index=pd.MultiIndex.from_tuples([("SH600000", d) for d in dates], names=["instrument", "datetime"]),
        )

        tech_df = pd.DataFrame(
            {
                "Mean($close, 5)": [p - 0.2 for p in prices],
                "Mean($close, 10)": [p - 0.5 for p in prices],
                "Mean($close, 20)": [p - 1.0 for p in prices],
                "Std($change, 20)": [0.02] * 30,
                "Rank($volume)": [50] * 30,
                "Mean($volume, 5)": volumes,
                "Mean($volume, 20)": volumes,
                "$volume / Mean($volume, 20)": [1.0] * 30,
            },
            index=pd.MultiIndex.from_tuples([("SH600000", d) for d in dates], names=["instrument", "datetime"]),
        )

        mock_get_prices.return_value = price_df
        mock_get_tech.return_value = tech_df

        mock_signal = AgentSignal(
            agent_name="valuation_agent",
            signal="bearish",
            confidence=0.9,
            reasoning="Price is well above all moving averages, at historical high percentile (95%), showing signs of overvaluation.",
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "ticker": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
            },
            "metadata": {},
        }

        result = valuation_agent(state)

        signal_dict = result["data"]["analyst_signals"]["valuation_agent"]
        assert signal_dict["signal"] == "bearish"
        assert signal_dict["confidence"] == 0.9

    @patch("big_a.models.hedge_fund.agents.valuation.get_prices")
    @patch("big_a.models.hedge_fund.agents.valuation.get_technical_indicators")
    @patch("big_a.models.hedge_fund.agents.valuation.call_llm")
    def test_valuation_agent_overvalued_signal(self, mock_call_llm, mock_get_tech, mock_get_prices) -> None:
        """Test valuation agent detects overvalued condition (high percentile)."""
        from big_a.models.hedge_fund.types import HedgeFundState

        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = [10.0 + (i % 5) * 0.1 for i in range(30)]
        volumes = [1000000 + i * 10000 for i in range(30)]

        price_df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": prices,
                "volume": volumes,
            },
            index=pd.MultiIndex.from_tuples([("SH600000", d) for d in dates], names=["instrument", "datetime"]),
        )

        tech_df = pd.DataFrame(
            {
                "Mean($close, 5)": prices,
                "Mean($close, 10)": [p + 0.1 for p in prices],
                "Mean($close, 20)": [p - 0.1 for p in prices],
                "Std($change, 20)": [0.02] * 30,
                "Rank($volume)": [50] * 30,
                "Mean($volume, 5)": volumes,
                "Mean($volume, 20)": volumes,
                "$volume / Mean($volume, 20)": [1.0] * 30,
            },
            index=pd.MultiIndex.from_tuples([("SH600000", d) for d in dates], names=["instrument", "datetime"]),
        )

        mock_get_prices.return_value = price_df
        mock_get_tech.return_value = tech_df

        mock_signal = AgentSignal(
            agent_name="valuation_agent",
            signal="neutral",
            confidence=0.6,
            reasoning="Mixed signals: price is near MA5, historical percentile is moderate (55%), momentum is flat.",
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "ticker": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
            },
            "metadata": {},
        }

        result = valuation_agent(state)

        signal_dict = result["data"]["analyst_signals"]["valuation_agent"]
        assert signal_dict["signal"] == "neutral"
        assert signal_dict["confidence"] == 0.6

    def test_valuation_agent_missing_data(self) -> None:
        """Test valuation agent with missing required data."""
        from big_a.models.hedge_fund.types import HedgeFundState

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
            },
            "metadata": {},
        }

        result = valuation_agent(state)

        assert "analyst_signals" in result["data"]
        signal_dict = result["data"]["analyst_signals"]["valuation_agent"]
        assert signal_dict["agent_name"] == "valuation_agent"
        assert signal_dict["signal"] == "neutral"
        assert signal_dict["confidence"] == 0.0
        assert "Missing required data" in signal_dict["reasoning"]

    @patch("big_a.models.hedge_fund.agents.valuation.get_prices")
    @patch("big_a.models.hedge_fund.agents.valuation.get_technical_indicators")
    def test_valuation_agent_empty_data(self, mock_get_tech, mock_get_prices) -> None:
        """Test valuation agent with empty price data."""
        from big_a.models.hedge_fund.types import HedgeFundState

        mock_get_prices.return_value = pd.DataFrame()
        mock_get_tech.return_value = pd.DataFrame()

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "ticker": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
            },
            "metadata": {},
        }

        result = valuation_agent(state)

        signal_dict = result["data"]["analyst_signals"]["valuation_agent"]
        assert signal_dict["signal"] == "neutral"
        assert signal_dict["confidence"] == 0.0
        assert "No price or technical data available" in signal_dict["reasoning"]

    @patch("big_a.models.hedge_fund.agents.valuation.get_prices")
    @patch("big_a.models.hedge_fund.agents.valuation.get_technical_indicators")
    @patch("big_a.models.hedge_fund.agents.valuation.call_llm")
    def test_valuation_agent_error_handling(self, mock_call_llm, mock_get_tech, mock_get_prices) -> None:
        """Test valuation agent handles errors gracefully."""
        from big_a.models.hedge_fund.types import HedgeFundState

        mock_get_prices.side_effect = Exception("Qlib connection error")

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "ticker": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
            },
            "metadata": {},
        }

        result = valuation_agent(state)

        signal_dict = result["data"]["analyst_signals"]["valuation_agent"]
        assert signal_dict["signal"] == "neutral"
        assert signal_dict["confidence"] == 0.0
        assert "Error during valuation analysis" in signal_dict["reasoning"]
        assert "Qlib connection error" in signal_dict["reasoning"]

    @patch("big_a.models.hedge_fund.agents.valuation.get_prices")
    @patch("big_a.models.hedge_fund.agents.valuation.get_technical_indicators")
    @patch("big_a.models.hedge_fund.agents.valuation.call_llm")
    def test_valuation_agent_preserves_existing_signals(self, mock_call_llm, mock_get_tech, mock_get_prices) -> None:
        """Test valuation agent preserves existing analyst signals."""
        from big_a.models.hedge_fund.types import HedgeFundState

        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = [10.0 + i * 0.1 for i in range(30)]
        volumes = [1000000 + i * 10000 for i in range(30)]

        price_df = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": prices,
                "volume": volumes,
            },
            index=pd.MultiIndex.from_tuples([("SH600000", d) for d in dates], names=["instrument", "datetime"]),
        )

        tech_df = pd.DataFrame(
            {
                "Mean($close, 5)": [p + 0.2 for p in prices],
                "Mean($close, 10)": [p + 0.5 for p in prices],
                "Mean($close, 20)": [p + 1.0 for p in prices],
                "Std($change, 20)": [0.02] * 30,
                "Rank($volume)": [50] * 30,
                "Mean($volume, 5)": volumes,
                "Mean($volume, 20)": volumes,
                "$volume / Mean($volume, 20)": [1.0] * 30,
            },
            index=pd.MultiIndex.from_tuples([("SH600000", d) for d in dates], names=["instrument", "datetime"]),
        )

        mock_get_prices.return_value = price_df
        mock_get_tech.return_value = tech_df

        mock_signal = AgentSignal(
            agent_name="valuation_agent",
            signal="bullish",
            confidence=0.85,
            reasoning="Test reasoning",
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "ticker": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2024-01-30",
                "analyst_signals": {
                    "technical_agent": {
                        "agent_name": "technical_agent",
                        "signal": "bullish",
                        "confidence": 0.75,
                        "reasoning": "Technical analysis bullish",
                    }
                },
            },
            "metadata": {},
        }

        result = valuation_agent(state)

        assert "technical_agent" in result["data"]["analyst_signals"]
        assert result["data"]["analyst_signals"]["technical_agent"]["agent_name"] == "technical_agent"
        assert "valuation_agent" in result["data"]["analyst_signals"]
