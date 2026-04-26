"""Tests for portfolio manager agent."""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from big_a.models.hedge_fund.agents.portfolio_manager import portfolio_manager_agent
from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState, PortfolioDecision, RiskAssessment


class TestPortfolioManagerAgent:
    """Tests for portfolio_manager_agent function."""

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_risk_bullish_high_confidence_produces_positive_score(self, mock_call_llm: MagicMock) -> None:
        """Test that bullish risk with high confidence produces positive score."""
        mock_call_llm.return_value = PortfolioDecision(
            action="buy",
            score=0.8,
            reasoning="Strong bullish risk assessment with high confidence",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bullish",
                    confidence=0.9,
                    max_position_weight=0.8,
                    reasoning="Strong bullish consensus with high confidence",
                ).model_dump(),
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.85,
                        reasoning="Strong uptrend indicators",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        assert "portfolio_decision" in result["data"]
        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "buy"
        assert decision["score"] == 0.8
        assert decision["score"] > 0
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_risk_bearish_high_confidence_produces_negative_score(self, mock_call_llm: MagicMock) -> None:
        """Test that bearish risk with high confidence produces negative score."""
        mock_call_llm.return_value = PortfolioDecision(
            action="sell",
            score=-0.85,
            reasoning="Strong bearish risk assessment with high confidence",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bearish",
                    confidence=0.88,
                    max_position_weight=0.9,
                    reasoning="Strong bearish consensus with high confidence",
                ).model_dump(),
                "analyst_signals": {
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

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "sell"
        assert decision["score"] == -0.85
        assert decision["score"] < 0

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_risk_neutral_produces_score_near_zero(self, mock_call_llm: MagicMock) -> None:
        """Test that neutral risk assessment produces score near zero."""
        mock_call_llm.return_value = PortfolioDecision(
            action="hold",
            score=0.1,
            reasoning="Neutral risk assessment with mixed signals",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="neutral",
                    confidence=0.5,
                    max_position_weight=0.2,
                    reasoning="Mixed signals, neutral stance",
                ).model_dump(),
                "analyst_signals": {},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "hold"
        assert abs(decision["score"]) < 0.5

    def test_no_risk_assessment_produces_neutral_fallback(self) -> None:
        """Test that missing risk assessment produces neutral fallback."""
        state: HedgeFundState = {
            "messages": [],
            "data": {
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.9,
                        reasoning="Strong signals",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "hold"
        assert decision["score"] == 0.0
        assert "No risk assessment available" in decision["reasoning"]

    def test_missing_risk_assessment_field_produces_neutral_fallback(self) -> None:
        """Test that risk assessment missing required fields produces neutral fallback."""
        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": {
                    "adjusted_signal": "bullish",
                    "confidence": 0.9,
                },
                "analyst_signals": {},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "hold"
        assert decision["score"] == 0.0
        assert "Invalid risk assessment" in decision["reasoning"]

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_multiple_analysts_agree_bullish_produces_stronger_positive_score(
        self, mock_call_llm: MagicMock
    ) -> None:
        """Test that multiple bullish analysts produce stronger positive score."""
        mock_call_llm.return_value = PortfolioDecision(
            action="buy",
            score=0.9,
            reasoning="Strong bullish consensus across multiple analysts",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bullish",
                    confidence=0.95,
                    max_position_weight=1.0,
                    reasoning="Strong bullish consensus",
                ).model_dump(),
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="bullish",
                        confidence=0.9,
                        reasoning="Strong uptrend",
                    ).model_dump(),
                    "valuation_agent": AgentSignal(
                        agent_name="valuation_agent",
                        signal="bullish",
                        confidence=0.85,
                        reasoning="Undervalued",
                    ).model_dump(),
                    "sentiment_agent": AgentSignal(
                        agent_name="sentiment_agent",
                        signal="bullish",
                        confidence=0.88,
                        reasoning="Positive sentiment",
                    ).model_dump(),
                },
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "buy"
        assert decision["score"] == 0.9
        assert decision["score"] > 0

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_config_passed_to_llm(self, mock_call_llm: MagicMock) -> None:
        """Test that config from metadata is passed to LLM."""
        mock_call_llm.return_value = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning="Test decision",
        )

        test_config = {"llm": {"model": "test-model", "temperature": 0.5}}

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="neutral",
                    confidence=0.5,
                    max_position_weight=0.2,
                    reasoning="Neutral",
                ).model_dump(),
                "analyst_signals": {},
            },
            "metadata": {"config": test_config},
        }

        result = portfolio_manager_agent(state)

        assert "portfolio_decision" in result["data"]
        mock_call_llm.assert_called_once()
        call_args = mock_call_llm.call_args
        assert call_args[0][2] == test_config

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_score_is_in_valid_range(self, mock_call_llm: MagicMock) -> None:
        """Test that score is always in [-1, 1] range."""
        mock_call_llm.return_value = PortfolioDecision(
            action="buy",
            score=0.75,
            reasoning="Valid score",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bullish",
                    confidence=0.8,
                    max_position_weight=0.75,
                    reasoning="Bullish",
                ).model_dump(),
                "analyst_signals": {},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert -1.0 <= decision["score"] <= 1.0

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_score_is_float_not_nan_or_inf(self, mock_call_llm: MagicMock) -> None:
        """Test that score is a valid float, not NaN or inf."""
        mock_call_llm.return_value = PortfolioDecision(
            action="buy",
            score=0.5,
            reasoning="Valid float score",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bullish",
                    confidence=0.7,
                    max_position_weight=0.5,
                    reasoning="Bullish",
                ).model_dump(),
                "analyst_signals": {},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert isinstance(decision["score"], float)
        assert not pd.isna(decision["score"])
        assert not math.isinf(decision["score"])

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_portfolio_decision_can_be_converted_to_qlib_dataframe(self, mock_call_llm: MagicMock) -> None:
        """Test that PortfolioDecision can be converted to Qlib DataFrame format."""
        mock_call_llm.return_value = PortfolioDecision(
            action="buy",
            score=0.6,
            reasoning="Test decision for Qlib",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bullish",
                    confidence=0.8,
                    max_position_weight=0.6,
                    reasoning="Bullish",
                ).model_dump(),
                "analyst_signals": {},
            },
            "metadata": {"ticker": "SH600000", "date": "2024-01-01"},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        ticker = result.get("metadata", {}).get("ticker", "SH600000")
        date = result.get("metadata", {}).get("date", "2024-01-01")

        df = pd.DataFrame(
            {"score": [decision["score"]]},
            index=pd.MultiIndex.from_tuples(
                [(ticker, pd.Timestamp(date))],
                names=["instrument", "datetime"],
            ),
        )

        assert df.shape == (1, 1)
        assert df.iloc[0, 0] == 0.6
        assert df.index.names == ["instrument", "datetime"]

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_handles_risk_assessment_object(self, mock_call_llm: MagicMock) -> None:
        """Test that RiskAssessment object (not dict) is handled correctly."""
        mock_call_llm.return_value = PortfolioDecision(
            action="buy",
            score=0.7,
            reasoning="Test with RiskAssessment object",
        )

        risk_obj = RiskAssessment(
            adjusted_signal="bullish",
            confidence=0.8,
            max_position_weight=0.7,
            reasoning="Bullish assessment",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": risk_obj,
                "analyst_signals": {},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "buy"
        assert decision["score"] == 0.7

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_handles_agent_signal_objects(self, mock_call_llm: MagicMock) -> None:
        """Test that AgentSignal objects (not dicts) are handled correctly."""
        mock_call_llm.return_value = PortfolioDecision(
            action="buy",
            score=0.65,
            reasoning="Test with AgentSignal objects",
        )

        signal_obj = AgentSignal(
            agent_name="test_agent",
            signal="bullish",
            confidence=0.9,
            reasoning="Test signal",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bullish",
                    confidence=0.8,
                    max_position_weight=0.65,
                    reasoning="Bullish",
                ).model_dump(),
                "analyst_signals": {"test_agent": signal_obj},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "buy"
        assert decision["score"] == 0.65

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_llm_error_returns_neutral_fallback(self, mock_call_llm: MagicMock) -> None:
        """Test that LLM errors return neutral fallback decision."""
        mock_call_llm.side_effect = Exception("LLM API error")

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bullish",
                    confidence=0.9,
                    max_position_weight=0.8,
                    reasoning="Bullish",
                ).model_dump(),
                "analyst_signals": {},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "hold"
        assert decision["score"] == 0.0
        assert "Error during portfolio decision" in decision["reasoning"]

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_preserves_other_data_fields(self, mock_call_llm: MagicMock) -> None:
        """Test that other data fields are preserved in returned state."""
        mock_call_llm.return_value = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning="Test preservation",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="neutral",
                    confidence=0.5,
                    max_position_weight=0.2,
                    reasoning="Neutral",
                ).model_dump(),
                "analyst_signals": {
                    "technicals_agent": AgentSignal(
                        agent_name="technicals_agent",
                        signal="neutral",
                        confidence=0.5,
                        reasoning="Neutral",
                    ).model_dump(),
                },
                "ticker": "SH600000",
                "start_date": "2024-01-01",
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        assert "ticker" in result["data"]
        assert result["data"]["ticker"] == "SH600000"
        assert "start_date" in result["data"]
        assert result["data"]["start_date"] == "2024-01-01"
        assert "analyst_signals" in result["data"]
        assert "risk_assessment" in result["data"]

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_handles_none_signals(self, mock_call_llm: MagicMock) -> None:
        """Test that None signal values are handled gracefully."""
        mock_call_llm.return_value = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning="Test with None signals",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="neutral",
                    confidence=0.5,
                    max_position_weight=0.2,
                    reasoning="Neutral",
                ).model_dump(),
                "analyst_signals": {
                    "test_agent": None,
                    "another_agent": None,
                },
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "hold"
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_score_respects_max_position_weight(self, mock_call_llm: MagicMock) -> None:
        """Test that score absolute value respects max_position_weight."""
        mock_call_llm.return_value = PortfolioDecision(
            action="buy",
            score=0.5,
            reasoning="Score limited by max_position_weight",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="bullish",
                    confidence=0.9,
                    max_position_weight=0.5,
                    reasoning="Bullish with weight limit",
                ).model_dump(),
                "analyst_signals": {},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert abs(decision["score"]) <= 0.5

    @patch("big_a.models.hedge_fund.agents.portfolio_manager.call_llm")
    def test_empty_analyst_signals(self, mock_call_llm: MagicMock) -> None:
        """Test that empty analyst_signals is handled correctly."""
        mock_call_llm.return_value = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning="No analyst signals",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {
                "risk_assessment": RiskAssessment(
                    adjusted_signal="neutral",
                    confidence=0.5,
                    max_position_weight=0.2,
                    reasoning="Neutral",
                ).model_dump(),
                "analyst_signals": {},
            },
            "metadata": {},
        }

        result = portfolio_manager_agent(state)

        decision = result["data"]["portfolio_decision"]
        assert decision["action"] == "hold"
        mock_call_llm.assert_called_once()
        prompt = mock_call_llm.call_args[0][0]
        assert "无分析师信号" in prompt
