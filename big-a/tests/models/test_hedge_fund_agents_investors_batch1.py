from unittest.mock import Mock, patch

import pytest

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState


class TestWarrenBuffettAgent:
    @patch("big_a.models.hedge_fund.agents.warren_buffett.get_market_data")
    @patch("big_a.models.hedge_fund.agents.warren_buffett.call_llm")
    def test_warren_buffett_agent_normal(self, mock_call_llm, mock_get_market_data):
        mock_get_market_data.return_value = {
            "SH600000": {"close": 10.0, "volume": 1000000, "change": 0.05}
        }
        mock_signal = AgentSignal(
            agent_name="warren_buffett",
            signal="bullish",
            confidence=0.8,
            reasoning="公司具有持久竞争优势，盈利能力稳定"
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "metadata": {"ticker": "SH600000", "date": "2024-01-01"}
        }
        result = __import__(
            "big_a.models.hedge_fund.agents.warren_buffett",
            fromlist=["warren_buffett_agent"],
        ).warren_buffett_agent(state)

        assert "analyst_signals" in result["data"]
        assert "warren_buffett_agent" in result["data"]["analyst_signals"]
        assert result["data"]["analyst_signals"]["warren_buffett_agent"]["signal"] == "bullish"
        assert result["data"]["analyst_signals"]["warren_buffett_agent"]["confidence"] == 0.8
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.warren_buffett.call_llm")
    def test_warren_buffett_agent_missing_params(self, mock_call_llm):
        state: HedgeFundState = {"metadata": {}}
        result = __import__(
            "big_a.models.hedge_fund.agents.warren_buffett",
            fromlist=["warren_buffett_agent"],
        ).warren_buffett_agent(state)

        assert "analyst_signals" in result["data"]
        assert result["data"]["analyst_signals"]["warren_buffett_agent"]["signal"] == "neutral"
        assert result["data"]["analyst_signals"]["warren_buffett_agent"]["confidence"] == 0.0
        mock_call_llm.assert_not_called()


class TestCharlieMungerAgent:
    @patch("big_a.models.hedge_fund.agents.charlie_munger.get_market_data")
    @patch("big_a.models.hedge_fund.agents.charlie_munger.call_llm")
    def test_charlie_munger_agent_normal(self, mock_call_llm, mock_get_market_data):
        mock_get_market_data.return_value = {
            "SH600000": {"close": 10.0, "volume": 1000000, "change": 0.05}
        }
        mock_signal = AgentSignal(
            agent_name="charlie_munger",
            signal="bullish",
            confidence=0.75,
            reasoning="伟大的企业，价格公平"
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "metadata": {"ticker": "SH600000", "date": "2024-01-01"}
        }
        result = __import__(
            "big_a.models.hedge_fund.agents.charlie_munger",
            fromlist=["charlie_munger_agent"],
        ).charlie_munger_agent(state)

        assert "analyst_signals" in result["data"]
        assert "charlie_munger_agent" in result["data"]["analyst_signals"]
        assert result["data"]["analyst_signals"]["charlie_munger_agent"]["signal"] == "bullish"
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.charlie_munger.call_llm")
    def test_charlie_munger_agent_missing_params(self, mock_call_llm):
        state: HedgeFundState = {"metadata": {}}
        result = __import__(
            "big_a.models.hedge_fund.agents.charlie_munger",
            fromlist=["charlie_munger_agent"],
        ).charlie_munger_agent(state)

        assert result["data"]["analyst_signals"]["charlie_munger_agent"]["signal"] == "neutral"
        mock_call_llm.assert_not_called()


class TestPeterLynchAgent:
    @patch("big_a.models.hedge_fund.agents.peter_lynch.get_market_data")
    @patch("big_a.models.hedge_fund.agents.peter_lynch.call_llm")
    def test_peter_lynch_agent_normal(self, mock_call_llm, mock_get_market_data):
        mock_get_market_data.return_value = {
            "SH600000": {"close": 10.0, "volume": 1000000, "change": 0.05}
        }
        mock_signal = AgentSignal(
            agent_name="peter_lynch",
            signal="bullish",
            confidence=0.85,
            reasoning="快速成长型，PEG比率合理"
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "metadata": {"ticker": "SH600000", "date": "2024-01-01"}
        }
        result = __import__(
            "big_a.models.hedge_fund.agents.peter_lynch",
            fromlist=["peter_lynch_agent"],
        ).peter_lynch_agent(state)

        assert "analyst_signals" in result["data"]
        assert "peter_lynch_agent" in result["data"]["analyst_signals"]
        assert result["data"]["analyst_signals"]["peter_lynch_agent"]["signal"] == "bullish"
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.peter_lynch.call_llm")
    def test_peter_lynch_agent_missing_params(self, mock_call_llm):
        state: HedgeFundState = {"metadata": {}}
        result = __import__(
            "big_a.models.hedge_fund.agents.peter_lynch",
            fromlist=["peter_lynch_agent"],
        ).peter_lynch_agent(state)

        assert result["data"]["analyst_signals"]["peter_lynch_agent"]["signal"] == "neutral"
        mock_call_llm.assert_not_called()


class TestBenGrahamAgent:
    @patch("big_a.models.hedge_fund.agents.ben_graham.get_market_data")
    @patch("big_a.models.hedge_fund.agents.ben_graham.call_llm")
    def test_ben_graham_agent_normal(self, mock_call_llm, mock_get_market_data):
        mock_get_market_data.return_value = {
            "SH600000": {"close": 10.0, "volume": 1000000, "change": 0.05}
        }
        mock_signal = AgentSignal(
            agent_name="ben_graham",
            signal="bullish",
            confidence=0.9,
            reasoning="价格远低于内在价值，安全边际充足"
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "metadata": {"ticker": "SH600000", "date": "2024-01-01"}
        }
        result = __import__(
            "big_a.models.hedge_fund.agents.ben_graham",
            fromlist=["ben_graham_agent"],
        ).ben_graham_agent(state)

        assert "analyst_signals" in result["data"]
        assert "ben_graham_agent" in result["data"]["analyst_signals"]
        assert result["data"]["analyst_signals"]["ben_graham_agent"]["signal"] == "bullish"
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.ben_graham.call_llm")
    def test_ben_graham_agent_missing_params(self, mock_call_llm):
        state: HedgeFundState = {"metadata": {}}
        result = __import__(
            "big_a.models.hedge_fund.agents.ben_graham",
            fromlist=["ben_graham_agent"],
        ).ben_graham_agent(state)

        assert result["data"]["analyst_signals"]["ben_graham_agent"]["signal"] == "neutral"
        mock_call_llm.assert_not_called()


class TestPhilFisherAgent:
    @patch("big_a.models.hedge_fund.agents.phil_fisher.get_market_data")
    @patch("big_a.models.hedge_fund.agents.phil_fisher.call_llm")
    def test_phil_fisher_agent_normal(self, mock_call_llm, mock_get_market_data):
        mock_get_market_data.return_value = {
            "SH600000": {"close": 10.0, "volume": 1000000, "change": 0.05}
        }
        mock_signal = AgentSignal(
            agent_name="phil_fisher",
            signal="bullish",
            confidence=0.8,
            reasoning="长期成长潜力巨大，管理层有远见"
        )
        mock_call_llm.return_value = mock_signal

        state: HedgeFundState = {
            "metadata": {"ticker": "SH600000", "date": "2024-01-01"}
        }
        result = __import__(
            "big_a.models.hedge_fund.agents.phil_fisher",
            fromlist=["phil_fisher_agent"],
        ).phil_fisher_agent(state)

        assert "analyst_signals" in result["data"]
        assert "phil_fisher_agent" in result["data"]["analyst_signals"]
        assert result["data"]["analyst_signals"]["phil_fisher_agent"]["signal"] == "bullish"
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.phil_fisher.call_llm")
    def test_phil_fisher_agent_missing_params(self, mock_call_llm):
        state: HedgeFundState = {"metadata": {}}
        result = __import__(
            "big_a.models.hedge_fund.agents.phil_fisher",
            fromlist=["phil_fisher_agent"],
        ).phil_fisher_agent(state)

        assert result["data"]["analyst_signals"]["phil_fisher_agent"]["signal"] == "neutral"
        mock_call_llm.assert_not_called()


class TestNoUSMarketReferences:
    def test_no_forbidden_us_market_terms(self):
        import os

        agent_files = [
            "src/big_a/models/hedge_fund/agents/warren_buffett.py",
            "src/big_a/models/hedge_fund/agents/charlie_munger.py",
            "src/big_a/models/hedge_fund/agents/peter_lynch.py",
            "src/big_a/models/hedge_fund/agents/ben_graham.py",
            "src/big_a/models/hedge_fund/agents/phil_fisher.py",
        ]

        forbidden_terms = ["SEC", "10-K", "NYSE", "NASDAQ", "S&P"]

        for file_path in agent_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                for term in forbidden_terms:
                    assert term not in content, f"{term} found in {file_path}"
