"""Tests for risk manager agent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from big_a.models.hedge_fund.agents.risk_manager import risk_manager_agent
from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState, RiskAssessment


class TestRiskManagerAgent:
    """Tests for risk_manager_agent function."""

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_all_bullish_signals_produces_bullish_assessment(self, mock_call_llm: MagicMock) -> None:
        """Test that all bullish signals produce bullish risk assessment with high confidence."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="bullish",
            confidence=0.9,
            max_position_weight=0.8,
            reasoning="Strong bullish consensus across all analysts with high confidence",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.85,
                        reasoning="Strong uptrend indicators",
                    ).model_dump(),
                    "valuation_agent": AgentSignal(
                        agent_name="valuation_agent",
                        signal="bullish",
                        confidence=0.8,
                        reasoning="Undervalued relative to historical averages",
                    ).model_dump(),
                    "sentiment_agent": AgentSignal(
                        agent_name="sentiment_agent",
                        signal="bullish",
                        confidence=0.9,
                        reasoning="Positive news flow",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assert "risk_assessment" in result["data"]
        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "bullish"
        assert assessment["confidence"] == 0.9
        assert assessment["max_position_weight"] == 0.8
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_all_bearish_signals_produces_bearish_assessment(self, mock_call_llm: MagicMock) -> None:
        """Test that all bearish signals produce bearish risk assessment with high confidence."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="bearish",
            confidence=0.88,
            max_position_weight=0.1,
            reasoning="Strong bearish consensus across all analysts",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bearish",
                        confidence=0.85,
                        reasoning="Downtrend indicators",
                    ).model_dump(),
                    "valuation_agent": AgentSignal(
                        agent_name="valuation_agent",
                        signal="bearish",
                        confidence=0.8,
                        reasoning="Overvalued",
                    ).model_dump(),
                    "sentiment_agent": AgentSignal(
                        agent_name="sentiment_agent",
                        signal="bearish",
                        confidence=0.9,
                        reasoning="Negative news flow",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "bearish"
        assert assessment["confidence"] == 0.88
        assert assessment["max_position_weight"] == 0.1

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_mixed_signals_produces_measured_assessment(self, mock_call_llm: MagicMock) -> None:
        """Test that mixed signals produce measured/neutral assessment."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.5,
            max_position_weight=0.2,
            reasoning="Mixed signals with conflicting views, recommend cautious approach",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.7,
                        reasoning="Uptrend indicators",
                    ).model_dump(),
                    "valuation_agent": AgentSignal(
                        agent_name="valuation_agent",
                        signal="bearish",
                        confidence=0.75,
                        reasoning="Overvalued",
                    ).model_dump(),
                    "sentiment_agent": AgentSignal(
                        agent_name="sentiment_agent",
                        signal="neutral",
                        confidence=0.5,
                        reasoning="Mixed news",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "neutral"
        assert assessment["confidence"] == 0.5
        assert assessment["max_position_weight"] == 0.2

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_all_neutral_signals_produces_conservative_assessment(self, mock_call_llm: MagicMock) -> None:
        """Test that all neutral signals produce conservative assessment with low confidence."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.3,
            max_position_weight=0.15,
            reasoning="All analysts are neutral, insufficient conviction for directional position",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="neutral",
                        confidence=0.5,
                        reasoning="No clear trend",
                    ).model_dump(),
                    "valuation_agent": AgentSignal(
                        agent_name="valuation_agent",
                        signal="neutral",
                        confidence=0.4,
                        reasoning="Fairly valued",
                    ).model_dump(),
                    "sentiment_agent": AgentSignal(
                        agent_name="sentiment_agent",
                        signal="neutral",
                        confidence=0.5,
                        reasoning="No significant news",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "neutral"
        assert assessment["confidence"] == 0.3
        assert assessment["max_position_weight"] == 0.15

    def test_no_signals_returns_neutral_fallback(self) -> None:
        """Test that no signals returns neutral fallback with zero confidence."""
        state: HedgeFundState = {
            "messages": [],
            "data": {"analyst_signals": {}},
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "neutral"
        assert assessment["confidence"] == 0.0
        assert assessment["max_position_weight"] == 0.0
        assert "No analyst signals available" in assessment["reasoning"]

    def test_missing_analyst_signals_returns_neutral_fallback(self) -> None:
        """Test that missing analyst_signals field returns neutral fallback."""
        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "neutral"
        assert assessment["confidence"] == 0.0
        assert assessment["max_position_weight"] == 0.0

    def test_all_signals_are_none_returns_neutral_fallback(self) -> None:
        """Test that all None signals returns neutral fallback."""
        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": None,
                    "valuation_agent": None,
                    "sentiment_agent": None,
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "neutral"
        assert assessment["confidence"] == 0.0
        assert "No valid analyst signals" in assessment["reasoning"]

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_handles_dict_signals(self, mock_call_llm: MagicMock) -> None:
        """Test that dict signals are handled correctly (not just AgentSignal objects)."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="bullish",
            confidence=0.75,
            max_position_weight=0.6,
            reasoning="Good bullish signals",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": {
                        "agent_name": "technicals_agent",
                        "signal": "bullish",
                        "confidence": 0.8,
                        "reasoning": "Uptrend",
                    },
                    "valuation_agent": {
                        "agent_name": "valuation_agent",
                        "signal": "bullish",
                        "confidence": 0.7,
                        "reasoning": "Undervalued",
                    },
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "bullish"
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_handles_agent_signal_objects(self, mock_call_llm: MagicMock) -> None:
        """Test that AgentSignal objects are handled correctly."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="bearish",
            confidence=0.8,
            max_position_weight=0.2,
            reasoning="Bearish consensus",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bearish",
                        confidence=0.85,
                        reasoning="Downtrend",
                    ),
                    "valuation_agent": AgentSignal(
                        agent_name="valuation_agent",
                        signal="bearish",
                        confidence=0.75,
                        reasoning="Overvalued",
                    ),
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "bearish"
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_config_passed_to_llm(self, mock_call_llm: MagicMock) -> None:
        """Test that config from metadata is passed to LLM."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.5,
            max_position_weight=0.3,
            reasoning="Test",
        )

        config = {"llm": {"model": "custom-model", "temperature": 0.2}}
        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="neutral",
                        confidence=0.5,
                        reasoning="Test",
                    ).model_dump(),
                },
            },
            "metadata": {"config": config},
        }

        result = risk_manager_agent(state)

        mock_call_llm.assert_called_once()
        call_args = mock_call_llm.call_args
        assert call_args[0][2] == config

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_llm_error_returns_neutral_fallback(self, mock_call_llm: MagicMock) -> None:
        """Test that LLM errors are caught and return neutral fallback."""
        mock_call_llm.side_effect = Exception("LLM API error")

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.8,
                        reasoning="Uptrend",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "neutral"
        assert assessment["confidence"] == 0.0
        assert assessment["max_position_weight"] == 0.0
        assert "Error during risk assessment" in assessment["reasoning"]

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_signals_missing_required_fields_are_ignored(self, mock_call_llm: MagicMock) -> None:
        """Test that signals missing required fields are filtered out."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="bullish",
            confidence=0.8,
            max_position_weight=0.6,
            reasoning="Based on valid signals",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": {
                        "agent_name": "technicals_agent",
                        "signal": "bullish",
                        "confidence": 0.8,
                        "reasoning": "Valid signal",
                    },
                    "invalid_agent": {
                        "agent_name": "invalid_agent",
                        "reasoning": "Missing signal and confidence",
                    },
                    "another_invalid": {
                        "agent_name": "another_invalid",
                        "signal": "bearish",
                        "reasoning": "Missing confidence",
                    },
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "bullish"
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_multiple_investor_signals_aggregated(self, mock_call_llm: MagicMock) -> None:
        """Test that signals from multiple investor agents are properly aggregated."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="bullish",
            confidence=0.85,
            max_position_weight=0.7,
            reasoning="Strong consensus across multiple investor personas",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.8,
                        reasoning="Technical bullish",
                    ).model_dump(),
                    "valuation_agent": AgentSignal(
                        agent_name="valuation_agent",
                        signal="bullish",
                        confidence=0.75,
                        reasoning="Valuation bullish",
                    ).model_dump(),
                    "sentiment_agent": AgentSignal(
                        agent_name="sentiment_agent",
                        signal="neutral",
                        confidence=0.5,
                        reasoning="Sentiment neutral",
                    ).model_dump(),
                    "warren_buffett_agent": AgentSignal(
                        agent_name="warren_buffett_agent",
                        signal="bullish",
                        confidence=0.9,
                        reasoning="Value investing bullish",
                    ).model_dump(),
                    "charlie_munger_agent": AgentSignal(
                        agent_name="charlie_munger_agent",
                        signal="bullish",
                        confidence=0.85,
                        reasoning="Rational bullish",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        assessment = result["data"]["risk_assessment"]
        assert assessment["adjusted_signal"] == "bullish"
        mock_call_llm.assert_called_once()
        prompt = mock_call_llm.call_args[0][0]
        assert "warren_buffett_agent" in prompt
        assert "charlie_munger_agent" in prompt

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_prompt_includes_risk_metrics(self, mock_call_llm: MagicMock) -> None:
        """Test that the LLM prompt includes computed risk metrics."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.5,
            max_position_weight=0.3,
            reasoning="Test",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.8,
                        reasoning="Test",
                    ).model_dump(),
                    "valuation_agent": AgentSignal(
                        agent_name="valuation_agent",
                        signal="bearish",
                        confidence=0.7,
                        reasoning="Test",
                    ).model_dump(),
                    "sentiment_agent": AgentSignal(
                        agent_name="sentiment_agent",
                        signal="neutral",
                        confidence=0.5,
                        reasoning="Test",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = risk_manager_agent(state)

        prompt = mock_call_llm.call_args[0][0]
        assert "看涨信号数量: 1" in prompt
        assert "看跌信号数量: 1" in prompt
        assert "中性信号数量: 1" in prompt
        assert "平均置信度" in prompt
        assert "信号一致性比例" in prompt

    @patch("big_a.models.hedge_fund.agents.risk_manager.call_llm")
    def test_preserves_other_data_fields(self, mock_call_llm: MagicMock) -> None:
        """Test that other data fields are preserved in the returned state."""
        mock_call_llm.return_value = RiskAssessment(
            adjusted_signal="bullish",
            confidence=0.8,
            max_position_weight=0.6,
            reasoning="Test",
        )

        state: HedgeFundState = {
            "messages": ["test message"],
            "data": {
                "ticker": "SH600000",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.8,
                        reasoning="Test",
                    ).model_dump(),
                },
            },
            "metadata": {"config": {"test": "value"}},
        }

        result = risk_manager_agent(state)

        assert result["messages"] == ["test message"]
        assert result["data"]["ticker"] == "SH600000"
        assert result["data"]["start_date"] == "2024-01-01"
        assert result["data"]["end_date"] == "2024-12-31"
        assert result["metadata"]["config"] == {"test": "value"}
