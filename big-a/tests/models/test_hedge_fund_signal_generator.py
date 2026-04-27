"""Tests for HedgeFundSignalGenerator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from big_a.models.hedge_fund import HedgeFundSignalGenerator
from big_a.models.hedge_fund.types import PortfolioDecision


class TestHedgeFundSignalGenerator:
    """Tests for HedgeFundSignalGenerator class."""

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = {"llm": {"model": "test-model"}}
        gen = HedgeFundSignalGenerator(config)
        assert gen.config == config

    def test_init_without_config(self) -> None:
        """Test initialization without config."""
        gen = HedgeFundSignalGenerator()
        assert gen.config == {}

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_single_instrument_generates_signal(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that single instrument returns correct DataFrame."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        mock_decision = PortfolioDecision(
            action="buy",
            score=0.75,
            reasoning="Strong bullish signal from multiple agents",
        )
        mock_run_workflow.return_value = {
            "data": {"portfolio_decision": mock_decision.model_dump()},
        }

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert isinstance(result, pd.DataFrame)
        assert result.index.names == ["datetime", "instrument"]
        assert list(result.columns) == ["score"]
        assert len(result) == 1
        assert result.iloc[0]["score"] == 0.75
        assert result.index[0][0] == pd.Timestamp("2024-12-31")
        assert result.index[0][1] == "SH600000"

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_multiple_instruments_generate_signals(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that multiple instruments return DataFrame with all instruments."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        def side_effect(ticker: str, **kwargs: object) -> dict[str, object]:
            if ticker == "SH600000":
                return {
                    "data": {
                        "portfolio_decision": PortfolioDecision(
                            action="buy",
                            score=0.8,
                            reasoning="Bullish",
                        ).model_dump()
                    },
                }
            else:
                return {
                    "data": {
                        "portfolio_decision": PortfolioDecision(
                            action="sell",
                            score=-0.6,
                            reasoning="Bearish",
                        ).model_dump()
                    },
                }

        mock_run_workflow.side_effect = side_effect

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments=["SH600000", "SZ000001"],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert len(result) == 2
        scores = result["score"].tolist()
        assert 0.8 in scores
        assert -0.6 in scores
        instruments = result.index.get_level_values(1).tolist()
        assert "SH600000" in instruments
        assert "SZ000001" in instruments

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_score_in_valid_range(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that score is clamped to [-1, 1] range when returned from dict."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        mock_run_workflow.return_value = {
            "data": {"portfolio_decision": {"score": 1.5, "action": "buy", "reasoning": "Test"}},
        }

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert result.iloc[0]["score"] == 1.0

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_dataframe_has_correct_multiindex_structure(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that DataFrame has correct MultiIndex structure."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        mock_decision = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning="Neutral",
        )
        mock_run_workflow.return_value = {
            "data": {"portfolio_decision": mock_decision.model_dump()},
        }

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["datetime", "instrument"]
        assert list(result.columns) == ["score"]

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_workflow_error_returns_neutral_score(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that workflow errors return neutral score (0.0)."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph
        mock_run_workflow.side_effect = Exception("Workflow error")

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert len(result) == 1
        assert result.iloc[0]["score"] == 0.0

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_config_passed_through_correctly(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that config is passed through to workflow."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        mock_decision = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning="Test",
        )
        mock_run_workflow.return_value = {
            "data": {"portfolio_decision": mock_decision.model_dump()},
        }

        config = {"llm": {"model": "custom-model"}}
        gen = HedgeFundSignalGenerator(config)
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        mock_create_workflow.assert_called_once_with(config=config)
        mock_run_workflow.assert_called_once()
        call_kwargs = mock_run_workflow.call_args.kwargs
        assert call_kwargs["config"] == config

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_extract_score_from_dict(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that score is extracted from dict decision."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        mock_run_workflow.return_value = {
            "data": {"portfolio_decision": {"score": 0.5, "action": "buy", "reasoning": "Test"}},
        }

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert result.iloc[0]["score"] == 0.5

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_extract_score_from_portfolio_decision_object(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that score is extracted from PortfolioDecision object."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        mock_decision = PortfolioDecision(
            action="buy",
            score=0.7,
            reasoning="Test",
        )
        mock_run_workflow.return_value = {
            "data": {"portfolio_decision": mock_decision},
        }

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert result.iloc[0]["score"] == 0.7

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_empty_instruments_returns_empty_dataframe(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that empty instruments list returns empty DataFrame."""
        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments=[],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert result.index.names == ["datetime", "instrument"]
        assert list(result.columns) == ["score"]

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_missing_portfolio_decision_returns_zero(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that missing portfolio_decision returns score 0.0."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        mock_run_workflow.return_value = {"data": {}}

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert result.iloc[0]["score"] == 0.0

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_invalid_score_returns_zero(
        self,
        mock_run_workflow: MagicMock,
        mock_create_workflow: MagicMock,
    ) -> None:
        """Test that invalid score returns 0.0."""
        mock_graph = MagicMock()
        mock_create_workflow.return_value = mock_graph

        mock_run_workflow.return_value = {
            "data": {"portfolio_decision": {"score": "invalid", "action": "hold", "reasoning": "Test"}},
        }

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(
            instruments="SH600000",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert result.iloc[0]["score"] == 0.0


class TestHedgeFundSignalGeneratorImport:
    """Tests for HedgeFundSignalGenerator import."""

    def test_import_from_package(self) -> None:
        """Test that HedgeFundSignalGenerator can be imported from package."""
        from big_a.models.hedge_fund import HedgeFundSignalGenerator as HFSG

        assert HFSG is HedgeFundSignalGenerator

    def test_direct_import(self) -> None:
        """Test that HedgeFundSignalGenerator can be imported directly."""
        from big_a.models.hedge_fund.signal_generator import HedgeFundSignalGenerator as HFSG

        assert HFSG is HedgeFundSignalGenerator


class TestReturnDetails:
    """Tests for return_details parameter in generate_signals."""

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_return_details_false_returns_dataframe(self, mock_run, mock_create) -> None:
        """return_details=False (default) returns plain DataFrame as before."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {
            "data": {
                "portfolio_decision": {"score": 0.5, "action": "buy", "reasoning": "test"},
                "analyst_signals": {
                    "technicals_agent": {
                        "SH600000": {"signal": "bullish", "confidence": 0.8, "reasoning": "RSI oversold"}
                    }
                },
            },
        }
        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals("SH600000", "2024-01-01", "2024-12-31", return_details=False)
        # Should be a plain DataFrame (existing behavior)
        assert isinstance(result, pd.DataFrame)
        assert result.index.names == ["datetime", "instrument"]

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_return_details_true_returns_dict(self, mock_run, mock_create) -> None:
        """return_details=True returns dict with signals and details."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {
            "data": {
                "portfolio_decision": {"score": 0.5, "action": "buy", "reasoning": "test"},
                "analyst_signals": {
                    "technicals_agent": {
                        "SH600000": {"signal": "bullish", "confidence": 0.8, "reasoning": "RSI oversold, MACD golden cross"}
                    },
                    "valuation_agent": {
                        "SH600000": {"signal": "neutral", "confidence": 0.5, "reasoning": "Fair valuation"}
                    },
                },
            },
        }
        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals("SH600000", "2024-01-01", "2024-12-31", return_details=True)
        # Should be a dict
        assert isinstance(result, dict)
        assert "signals" in result
        assert "details" in result
        # signals is still a DataFrame
        assert isinstance(result["signals"], pd.DataFrame)
        # details has per-instrument agent data
        assert "SH600000" in result["details"]
        details = result["details"]["SH600000"]
        assert "technicals_agent" in details
        assert details["technicals_agent"]["signal"] == "bullish"
        assert details["technicals_agent"]["confidence"] == 0.8
        assert "RSI" in details["technicals_agent"]["reasoning"]

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_details_multiple_instruments(self, mock_run, mock_create) -> None:
        """Details are preserved for multiple instruments."""
        mock_create.return_value = MagicMock()

        def side_effect(ticker, **kwargs):
            return {
                "data": {
                    "portfolio_decision": {"score": 0.6, "action": "buy", "reasoning": "test"},
                    "analyst_signals": {
                        "warren_buffett_agent": {
                            ticker: {"signal": "bullish", "confidence": 0.9, "reasoning": "Strong moat"}
                        }
                    },
                },
            }
        mock_run.side_effect = side_effect

        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals(["SH600000", "SZ000001"], "2024-01-01", "2024-12-31", return_details=True)
        assert "SH600000" in result["details"]
        assert "SZ000001" in result["details"]

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_details_empty_analyst_signals(self, mock_run, mock_create) -> None:
        """Empty analyst_signals still works gracefully."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {
            "data": {
                "portfolio_decision": {"score": 0.0, "action": "hold", "reasoning": "test"},
                "analyst_signals": {},
            },
        }
        gen = HedgeFundSignalGenerator()
        result = gen.generate_signals("SH600000", "2024-01-01", "2024-12-31", return_details=True)
        assert isinstance(result, dict)
        assert "SH600000" in result["details"]
        assert result["details"]["SH600000"] == {}

    @patch("big_a.models.hedge_fund.signal_generator.create_workflow")
    @patch("big_a.models.hedge_fund.signal_generator.run_workflow")
    def test_default_return_details_is_false(self, mock_run, mock_create) -> None:
        """Default behavior is return_details=False (backward compat)."""
        mock_create.return_value = MagicMock()
        mock_run.return_value = {
            "data": {
                "portfolio_decision": {"score": 0.3, "action": "buy", "reasoning": "test"},
                "analyst_signals": {"agent": {"SH600000": {"signal": "bullish", "confidence": 0.7, "reasoning": "test"}}},
            },
        }
        gen = HedgeFundSignalGenerator()
        # Call WITHOUT return_details — should return DataFrame
        result = gen.generate_signals("SH600000", "2024-01-01", "2024-12-31")
        assert isinstance(result, pd.DataFrame)
