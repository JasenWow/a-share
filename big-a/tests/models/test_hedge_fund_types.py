"""Tests for hedge fund type definitions."""
import pytest
from pydantic import ValidationError

from big_a.models.hedge_fund.types import (
    AgentSignal,
    RiskAssessment,
    PortfolioDecision,
    MarketData,
    HedgeFundState,
    merge_dicts,
)


class TestAgentSignal:

    def test_create_valid_signal(self):
        signal = AgentSignal(
            agent_name="technical_analyst",
            signal="bullish",
            confidence=0.85,
            reasoning="Strong momentum breakout"
        )
        assert signal.agent_name == "technical_analyst"
        assert signal.signal == "bullish"
        assert signal.confidence == 0.85
        assert signal.reasoning == "Strong momentum breakout"

    def test_valid_signals(self):
        valid_signals = ["bullish", "bearish", "neutral"]
        for sig in valid_signals:
            signal = AgentSignal(
                agent_name="test_agent",
                signal=sig,
                confidence=0.5,
                reasoning="test"
            )
            assert signal.signal == sig

    def test_invalid_signal_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            AgentSignal(
                agent_name="test_agent",
                signal="invalid_signal",
                confidence=0.5,
                reasoning="test"
            )
        assert "signal" in str(exc_info.value).lower()

    def test_confidence_out_of_range_high(self):
        with pytest.raises(ValidationError) as exc_info:
            AgentSignal(
                agent_name="test_agent",
                signal="bullish",
                confidence=1.5,
                reasoning="test"
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_confidence_out_of_range_low(self):
        with pytest.raises(ValidationError) as exc_info:
            AgentSignal(
                agent_name="test_agent",
                signal="bullish",
                confidence=-0.1,
                reasoning="test"
            )
        assert "confidence" in str(exc_info.value).lower()

    def test_confidence_boundary_values(self):
        signal_high = AgentSignal(
            agent_name="test",
            signal="neutral",
            confidence=1.0,
            reasoning="test"
        )
        assert signal_high.confidence == 1.0

        signal_low = AgentSignal(
            agent_name="test",
            signal="neutral",
            confidence=0.0,
            reasoning="test"
        )
        assert signal_low.confidence == 0.0


class TestRiskAssessment:

    def test_create_valid_risk_assessment(self):
        risk = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.7,
            max_position_weight=0.3,
            reasoning="High volatility detected"
        )
        assert risk.adjusted_signal == "neutral"
        assert risk.confidence == 0.7
        assert risk.max_position_weight == 0.3
        assert risk.reasoning == "High volatility detected"

    def test_valid_adjusted_signals(self):
        valid_signals = ["bullish", "bearish", "neutral"]
        for sig in valid_signals:
            risk = RiskAssessment(
                adjusted_signal=sig,
                confidence=0.5,
                max_position_weight=0.5,
                reasoning="test"
            )
            assert risk.adjusted_signal == sig

    def test_max_position_weight_boundary(self):
        risk_max = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.5,
            max_position_weight=1.0,
            reasoning="test"
        )
        assert risk_max.max_position_weight == 1.0

        risk_min = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.5,
            max_position_weight=0.0,
            reasoning="test"
        )
        assert risk_min.max_position_weight == 0.0


class TestPortfolioDecision:

    def test_create_valid_decision(self):
        decision = PortfolioDecision(
            action="buy",
            score=0.8,
            reasoning="Strong bullish signal with low risk"
        )
        assert decision.action == "buy"
        assert decision.score == 0.8
        assert decision.reasoning == "Strong bullish signal with low risk"

    def test_valid_actions(self):
        valid_actions = ["buy", "sell", "hold"]
        for action in valid_actions:
            decision = PortfolioDecision(
                action=action,
                score=0.0,
                reasoning="test"
            )
            assert decision.action == action

    def test_score_out_of_range_high(self):
        with pytest.raises(ValidationError) as exc_info:
            PortfolioDecision(
                action="buy",
                score=1.5,
                reasoning="test"
            )
        assert "score" in str(exc_info.value).lower()

    def test_score_out_of_range_low(self):
        with pytest.raises(ValidationError) as exc_info:
            PortfolioDecision(
                action="buy",
                score=-1.5,
                reasoning="test"
            )
        assert "score" in str(exc_info.value).lower()

    def test_score_boundary_values(self):
        decision_max = PortfolioDecision(
            action="buy",
            score=1.0,
            reasoning="test"
        )
        assert decision_max.score == 1.0

        decision_min = PortfolioDecision(
            action="sell",
            score=-1.0,
            reasoning="test"
        )
        assert decision_min.score == -1.0

    def test_qlib_score_mapping_bullish(self):
        decision = PortfolioDecision(
            action="buy",
            score=0.75,
            reasoning="test"
        )
        assert 0.5 <= decision.score <= 1.0

    def test_qlib_score_mapping_bearish(self):
        decision = PortfolioDecision(
            action="sell",
            score=-0.75,
            reasoning="test"
        )
        assert -1.0 <= decision.score <= -0.5

    def test_qlib_score_mapping_neutral(self):
        decision = PortfolioDecision(
            action="hold",
            score=0.1,
            reasoning="test"
        )
        assert -0.5 < decision.score < 0.5


class TestMarketData:

    def test_create_valid_market_data(self):
        data = MarketData(
            instrument="SH600000",
            date="2026-04-26",
            open=10.5,
            high=11.0,
            low=10.3,
            close=10.8,
            volume=1000000.0
        )
        assert data.instrument == "SH600000"
        assert data.date == "2026-04-26"
        assert data.open == 10.5
        assert data.high == 11.0
        assert data.low == 10.3
        assert data.close == 10.8
        assert data.volume == 1000000.0

    def test_market_data_with_optional_fields(self):
        data = MarketData(
            instrument="SH600001",
            date="2026-04-26"
        )
        assert data.instrument == "SH600001"
        assert data.date == "2026-04-26"
        assert data.open is None
        assert data.high is None
        assert data.low is None
        assert data.close is None
        assert data.volume is None


class TestHedgeFundState:

    def test_merge_dicts_basic(self):
        dict_a = {"key1": "value1", "key2": "value2"}
        dict_b = {"key2": "new_value2", "key3": "value3"}
        result = merge_dicts(dict_a, dict_b)
        assert result == {"key1": "value1", "key2": "new_value2", "key3": "value3"}

    def test_merge_dicts_empty(self):
        assert merge_dicts({}, {}) == {}
        assert merge_dicts({"a": 1}, {}) == {"a": 1}
        assert merge_dicts({}, {"b": 2}) == {"b": 2}

    def test_merge_dicts_nested(self):
        dict_a = {"outer": {"inner": "value"}}
        dict_b = {"other": "data"}
        result = merge_dicts(dict_a, dict_b)
        assert result == {"outer": {"inner": "value"}, "other": "data"}

    def test_hedge_fund_state_typed_dict(self):
        state: HedgeFundState = {
            "messages": ["msg1", "msg2"],
            "data": {"stock": "SH600000"},
            "metadata": {"timestamp": "2026-04-26"}
        }
        assert state["messages"] == ["msg1", "msg2"]
        assert state["data"]["stock"] == "SH600000"
        assert state["metadata"]["timestamp"] == "2026-04-26"


class TestSerialization:

    def test_agent_signal_serialize_deserialize(self):
        original = AgentSignal(
            agent_name="test",
            signal="bullish",
            confidence=0.9,
            reasoning="test"
        )
        data = original.model_dump()
        restored = AgentSignal.model_validate(data)
        assert restored.agent_name == original.agent_name
        assert restored.signal == original.signal
        assert restored.confidence == original.confidence
        assert restored.reasoning == original.reasoning

    def test_risk_assessment_serialize_deserialize(self):
        original = RiskAssessment(
            adjusted_signal="bearish",
            confidence=0.6,
            max_position_weight=0.2,
            reasoning="test"
        )
        data = original.model_dump()
        restored = RiskAssessment.model_validate(data)
        assert restored.adjusted_signal == original.adjusted_signal
        assert restored.confidence == original.confidence
        assert restored.max_position_weight == original.max_position_weight
        assert restored.reasoning == original.reasoning

    def test_portfolio_decision_serialize_deserialize(self):
        original = PortfolioDecision(
            action="hold",
            score=0.1,
            reasoning="test"
        )
        data = original.model_dump()
        restored = PortfolioDecision.model_validate(data)
        assert restored.action == original.action
        assert restored.score == original.score
        assert restored.reasoning == original.reasoning

    def test_market_data_serialize_deserialize(self):
        original = MarketData(
            instrument="SH600000",
            date="2026-04-26",
            close=10.5
        )
        data = original.model_dump()
        restored = MarketData.model_validate(data)
        assert restored.instrument == original.instrument
        assert restored.date == original.date
        assert restored.close == original.close
