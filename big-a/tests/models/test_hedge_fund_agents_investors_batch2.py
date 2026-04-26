"""Tests for hedge fund investor agents - Batch 2 (Ackman, Burry, Taleb, Wood, Damodaran)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from big_a.models.hedge_fund.agents.bill_ackman import bill_ackman_agent
from big_a.models.hedge_fund.agents.michael_burry import michael_burry_agent
from big_a.models.hedge_fund.agents.nassim_taleb import nassim_taleb_agent
from big_a.models.hedge_fund.agents.cathie_wood import cathie_wood_agent
from big_a.models.hedge_fund.agents.aswath_damodaran import aswath_damodaran_agent
from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState


class TestBillAckmanAgent:
    """Tests for Bill Ackman agent."""

    def test_bill_ackman_agent_normal(self) -> None:
        """Test Bill Ackman agent with normal data."""
        state: HedgeFundState = {
            "metadata": {
                "ticker": "SH600000",
                "date": "2024-01-15",
                "config": {"llm": {"api_key": "test-key"}}
            }
        }

        mock_signal = AgentSignal(
            agent_name="bill_ackman",
            signal="bullish",
            confidence=0.8,
            reasoning="Strong brand moat and potential for activist change"
        )

        with patch("big_a.models.hedge_fund.agents.bill_ackman.get_market_data") as mock_get_data, \
             patch("big_a.models.hedge_fund.agents.bill_ackman.call_llm") as mock_call_llm:
            mock_get_data.return_value = {
                "SH600000": {"close": 10.5, "volume": 1000000, "change": 0.02}
            }
            mock_call_llm.return_value = mock_signal

            result = bill_ackman_agent(state)

            assert "data" in result
            assert "analyst_signals" in result["data"]
            assert "bill_ackman_agent" in result["data"]["analyst_signals"]
            signal_data = result["data"]["analyst_signals"]["bill_ackman_agent"]
            assert signal_data["agent_name"] == "bill_ackman"
            assert signal_data["signal"] == "bullish"
            assert signal_data["confidence"] == 0.8

    def test_bill_ackman_agent_missing_params(self) -> None:
        """Test Bill Ackman agent with missing parameters."""
        state: HedgeFundState = {
            "metadata": {}
        }

        result = bill_ackman_agent(state)

        assert "data" in result
        assert "analyst_signals" in result["data"]
        assert "bill_ackman_agent" in result["data"]["analyst_signals"]
        signal_data = result["data"]["analyst_signals"]["bill_ackman_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0


class TestMichaelBurryAgent:
    """Tests for Michael Burry agent."""

    def test_michael_burry_agent_normal(self) -> None:
        """Test Michael Burry agent with normal data."""
        state: HedgeFundState = {
            "metadata": {
                "ticker": "SZ000001",
                "date": "2024-01-15",
                "config": {"llm": {"api_key": "test-key"}}
            }
        }

        mock_signal = AgentSignal(
            agent_name="michael_burry",
            signal="bullish",
            confidence=0.9,
            reasoning="Deep value opportunity with catalyst for reversion"
        )

        with patch("big_a.models.hedge_fund.agents.michael_burry.get_market_data") as mock_get_data, \
             patch("big_a.models.hedge_fund.agents.michael_burry.call_llm") as mock_call_llm:
            mock_get_data.return_value = {
                "SZ000001": {"close": 5.2, "volume": 500000, "change": -0.05}
            }
            mock_call_llm.return_value = mock_signal

            result = michael_burry_agent(state)

            assert "data" in result
            assert "analyst_signals" in result["data"]
            assert "michael_burry_agent" in result["data"]["analyst_signals"]
            signal_data = result["data"]["analyst_signals"]["michael_burry_agent"]
            assert signal_data["agent_name"] == "michael_burry"
            assert signal_data["signal"] == "bullish"
            assert signal_data["confidence"] == 0.9

    def test_michael_burry_agent_missing_params(self) -> None:
        """Test Michael Burry agent with missing parameters."""
        state: HedgeFundState = {
            "metadata": {}
        }

        result = michael_burry_agent(state)

        assert "data" in result
        assert "analyst_signals" in result["data"]
        assert "michael_burry_agent" in result["data"]["analyst_signals"]
        signal_data = result["data"]["analyst_signals"]["michael_burry_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0


class TestNassimTalebAgent:
    """Tests for Nassim Taleb agent."""

    def test_nassim_taleb_agent_normal(self) -> None:
        """Test Nassim Taleb agent with normal data."""
        state: HedgeFundState = {
            "metadata": {
                "ticker": "SH600519",
                "date": "2024-01-15",
                "config": {"llm": {"api_key": "test-key"}}
            }
        }

        mock_signal = AgentSignal(
            agent_name="nassim_taleb",
            signal="bearish",
            confidence=0.7,
            reasoning="High tail risk with limited convexity potential"
        )

        with patch("big_a.models.hedge_fund.agents.nassim_taleb.get_market_data") as mock_get_data, \
             patch("big_a.models.hedge_fund.agents.nassim_taleb.call_llm") as mock_call_llm:
            mock_get_data.return_value = {
                "SH600519": {"close": 1500.0, "volume": 200000, "change": 0.01}
            }
            mock_call_llm.return_value = mock_signal

            result = nassim_taleb_agent(state)

            assert "data" in result
            assert "analyst_signals" in result["data"]
            assert "nassim_taleb_agent" in result["data"]["analyst_signals"]
            signal_data = result["data"]["analyst_signals"]["nassim_taleb_agent"]
            assert signal_data["agent_name"] == "nassim_taleb"
            assert signal_data["signal"] == "bearish"
            assert signal_data["confidence"] == 0.7

    def test_nassim_taleb_agent_missing_params(self) -> None:
        """Test Nassim Taleb agent with missing parameters."""
        state: HedgeFundState = {
            "metadata": {}
        }

        result = nassim_taleb_agent(state)

        assert "data" in result
        assert "analyst_signals" in result["data"]
        assert "nassim_taleb_agent" in result["data"]["analyst_signals"]
        signal_data = result["data"]["analyst_signals"]["nassim_taleb_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0


class TestCathieWoodAgent:
    """Tests for Cathie Wood agent."""

    def test_cathie_wood_agent_normal(self) -> None:
        """Test Cathie Wood agent with normal data."""
        state: HedgeFundState = {
            "metadata": {
                "ticker": "SH688981",
                "date": "2024-01-15",
                "config": {"llm": {"api_key": "test-key"}}
            }
        }

        mock_signal = AgentSignal(
            agent_name="cathie_wood",
            signal="bullish",
            confidence=0.85,
            reasoning="Disruptive innovation with exponential growth potential"
        )

        with patch("big_a.models.hedge_fund.agents.cathie_wood.get_market_data") as mock_get_data, \
             patch("big_a.models.hedge_fund.agents.cathie_wood.call_llm") as mock_call_llm:
            mock_get_data.return_value = {
                "SH688981": {"close": 50.0, "volume": 300000, "change": 0.03}
            }
            mock_call_llm.return_value = mock_signal

            result = cathie_wood_agent(state)

            assert "data" in result
            assert "analyst_signals" in result["data"]
            assert "cathie_wood_agent" in result["data"]["analyst_signals"]
            signal_data = result["data"]["analyst_signals"]["cathie_wood_agent"]
            assert signal_data["agent_name"] == "cathie_wood"
            assert signal_data["signal"] == "bullish"
            assert signal_data["confidence"] == 0.85

    def test_cathie_wood_agent_missing_params(self) -> None:
        """Test Cathie Wood agent with missing parameters."""
        state: HedgeFundState = {
            "metadata": {}
        }

        result = cathie_wood_agent(state)

        assert "data" in result
        assert "analyst_signals" in result["data"]
        assert "cathie_wood_agent" in result["data"]["analyst_signals"]
        signal_data = result["data"]["analyst_signals"]["cathie_wood_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0


class TestAswathDamodaranAgent:
    """Tests for Aswath Damodaran agent."""

    def test_aswath_damodaran_agent_normal(self) -> None:
        """Test Aswath Damodaran agent with normal data."""
        state: HedgeFundState = {
            "metadata": {
                "ticker": "SZ300750",
                "date": "2024-01-15",
                "config": {"llm": {"api_key": "test-key"}}
            }
        }

        mock_signal = AgentSignal(
            agent_name="aswath_damodaran",
            signal="neutral",
            confidence=0.6,
            reasoning="Fair valuation based on DCF with reasonable assumptions"
        )

        with patch("big_a.models.hedge_fund.agents.aswath_damodaran.get_market_data") as mock_get_data, \
             patch("big_a.models.hedge_fund.agents.aswath_damodaran.call_llm") as mock_call_llm:
            mock_get_data.return_value = {
                "SZ300750": {"close": 180.0, "volume": 150000, "change": 0.0}
            }
            mock_call_llm.return_value = mock_signal

            result = aswath_damodaran_agent(state)

            assert "data" in result
            assert "analyst_signals" in result["data"]
            assert "aswath_damodaran_agent" in result["data"]["analyst_signals"]
            signal_data = result["data"]["analyst_signals"]["aswath_damodaran_agent"]
            assert signal_data["agent_name"] == "aswath_damodaran"
            assert signal_data["signal"] == "neutral"
            assert signal_data["confidence"] == 0.6

    def test_aswath_damodaran_agent_missing_params(self) -> None:
        """Test Aswath Damodaran agent with missing parameters."""
        state: HedgeFundState = {
            "metadata": {}
        }

        result = aswath_damodaran_agent(state)

        assert "data" in result
        assert "analyst_signals" in result["data"]
        assert "aswath_damodaran_agent" in result["data"]["analyst_signals"]
        signal_data = result["data"]["analyst_signals"]["aswath_damodaran_agent"]
        assert signal_data["signal"] == "neutral"
        assert signal_data["confidence"] == 0.0


class TestForbiddenUSMarketTerms:
    """Test that no forbidden US-market terms appear in agent files."""

    def test_no_forbidden_terms(self) -> None:
        """Ensure no SEC, 10-K, NYSE, NASDAQ, S&P references in agent files."""
        import subprocess

        agent_files = [
            "big-a/src/big_a/models/hedge_fund/agents/bill_ackman.py",
            "big-a/src/big_a/models/hedge_fund/agents/michael_burry.py",
            "big-a/src/big_a/models/hedge_fund/agents/nassim_taleb.py",
            "big-a/src/big_a/models/hedge_fund/agents/cathie_wood.py",
            "big-a/src/big_a/models/hedge_fund/agents/aswath_damodaran.py",
        ]

        forbidden_terms = ["SEC", "10-K", "NYSE", "NASDAQ", "S&P"]

        for agent_file in agent_files:
            for term in forbidden_terms:
                result = subprocess.run(
                    ["grep", term, agent_file],
                    capture_output=True,
                    text=True
                )
                assert result.returncode != 0, f"Found forbidden term '{term}' in {agent_file}"
