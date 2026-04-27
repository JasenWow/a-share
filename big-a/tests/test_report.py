"""Tests for big_a.report module (scorer + formatter)."""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from rich.console import Console


@pytest.fixture
def sample_watchlist():
    """Sample watchlist data."""
    return {"SZ001309": "德明利", "SZ300476": "胜宏科技", "SH688256": "寒武纪"}


@pytest.fixture
def sample_kronos_scores():
    """Sample Kronos scores."""
    return pd.DataFrame({
        "instrument": ["SH688256", "SZ300476", "SZ001309"],
        "score": [2.35, 0.82, -0.45],
        "score_pct": [3.8, 1.2, -0.7],
        "signal_date": pd.to_datetime(["2024-12-31"] * 3),
    })


@pytest.fixture
def sample_lightgbm_scores():
    """Sample LightGBM scores."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-12-31"] * 3),
        "instrument": ["SH688256", "SZ300476", "SZ001309"],
        "score": [0.82, 0.45, -0.12],
    })


@pytest.fixture
def sample_portfolio():
    """Sample portfolio data."""
    return pd.DataFrame({
        "instrument": ["SH688256", "SZ300476", "CASH"],
        "name": ["寒武纪", "胜宏科技", "现金"],
        "weight": [0.25, 0.25, 0.50],
        "allocation": [250000, 250000, 500000],
        "signal": ["Strong Buy", "Buy", "Hold"],
        "entry_price": [45.20, 32.50, 1.0],
    })


@pytest.fixture
def sample_market_data():
    """Sample market data."""
    dates = pd.date_range("2024-12-18", periods=10, freq="B")
    instruments = ["SZ001309", "SZ300476", "SH688256"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["date", "instrument"])
    n = len(idx)
    return pd.DataFrame({
        "open": np.random.uniform(30, 50, n),
        "high": np.random.uniform(30, 50, n),
        "low": np.random.uniform(30, 50, n),
        "close": np.random.uniform(30, 50, n),
        "volume": np.random.uniform(1e6, 1e7, n),
        "amount": np.random.uniform(1e8, 1e9, n),
        "change_pct": np.random.uniform(-5, 5, n),
        "factor": np.ones(n),
    }, index=idx)


class TestWatchlistScorerLoadWatchlist:
    """Tests for WatchlistScorer.load_watchlist method."""

    @patch("big_a.report.scorer.load_config")
    def test_load_watchlist_valid(self, mock_config):
        """Test loading valid watchlist returns correct dict."""
        from big_a.report.scorer import WatchlistScorer

        mock_config.return_value = {
            "watchlist": {
                "SH600000": "浦发银行",
                "SH600519": "贵州茅台",
            }
        }

        scorer = WatchlistScorer()
        result = scorer.load_watchlist()

        assert result == {"SH600000": "浦发银行", "SH600519": "贵州茅台"}
        assert len(result) == 2
        mock_config.assert_called_once()

    @patch("big_a.report.scorer.load_config")
    def test_load_watchlist_empty(self, mock_config):
        """Test loading empty watchlist returns empty dict."""
        from big_a.report.scorer import WatchlistScorer

        mock_config.return_value = {"watchlist": {}}

        scorer = WatchlistScorer()
        result = scorer.load_watchlist()

        assert result == {}
        assert len(result) == 0

    @patch("big_a.report.scorer.load_config")
    def test_load_watchlist_invalid_type(self, mock_config):
        """Test loading watchlist with invalid type returns empty dict."""
        from big_a.report.scorer import WatchlistScorer

        mock_config.return_value = {"watchlist": ["SH600000", "SH600519"]}

        scorer = WatchlistScorer()
        result = scorer.load_watchlist()

        assert result == {}
        assert len(result) == 0


class TestWatchlistScorerScoreKronos:
    """Tests for WatchlistScorer.score_kronos method."""

    @patch("big_a.report.scorer.load_config")
    @patch("big_a.report.scorer.KronosSignalGenerator")
    def test_score_kronos_valid(self, mock_kronos, mock_config):
        """Test scoring with Kronos returns correct DataFrame."""
        from big_a.report.scorer import WatchlistScorer

        mock_config.return_value = {
            "kronos": {
                "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
                "model_id": "NeoQuasar/Kronos-base",
                "device": "cpu",
                "lookback": 90,
                "pred_len": 10,
                "max_context": 512,
                "signal_mode": "mean",
            }
        }

        mock_gen = MagicMock()
        mock_kronos.return_value = mock_gen

        signals_df = pd.DataFrame({
            "score": [2.35, 0.82, -0.45],
        }, index=pd.MultiIndex.from_tuples([
            (pd.Timestamp("2024-12-31"), "SH688256"),
            (pd.Timestamp("2024-12-31"), "SZ300476"),
            (pd.Timestamp("2024-12-31"), "SZ001309"),
        ], names=["signal_date", "instrument"]))

        mock_gen.generate_signals.return_value = signals_df

        with patch("qlib.data.D") as mock_D, \
             patch.object(WatchlistScorer, "fetch_market_data") as mock_fetch:

            mock_D.calendar.return_value = pd.date_range("2024-01-01", periods=100, freq="B")

            market_df = pd.DataFrame({
                "close": [45.20, 32.50, 28.30],
            }, index=pd.MultiIndex.from_product([
                [pd.Timestamp("2024-12-31")],
                ["SH688256", "SZ300476", "SZ001309"],
            ], names=["date", "instrument"]))
            mock_fetch.return_value = market_df

            scorer = WatchlistScorer()
            result = scorer.score_kronos(["SH688256", "SZ300476", "SZ001309"])

            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["instrument", "score", "score_pct", "signal_date"]
            assert len(result) == 3
            assert "SH688256" in result["instrument"].values
            mock_gen.generate_signals.assert_called_once()

    @patch("big_a.report.scorer.load_config")
    @patch("big_a.report.scorer.KronosSignalGenerator")
    def test_score_kronos_empty_signals(self, mock_kronos, mock_config):
        """Test scoring with empty signals returns empty DataFrame."""
        from big_a.report.scorer import WatchlistScorer

        mock_config.return_value = {"kronos": {"lookback": 90}}

        mock_gen = MagicMock()
        mock_kronos.return_value = mock_gen
        mock_gen.generate_signals.return_value = pd.DataFrame()

        with patch("qlib.data.D") as mock_D:
            mock_D.calendar.return_value = pd.date_range("2024-01-01", periods=100, freq="B")

            scorer = WatchlistScorer()
            result = scorer.score_kronos(["SH688256", "SZ300476"])

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            assert list(result.columns) == ["instrument", "score", "score_pct", "signal_date"]


class TestWatchlistScorerScoreKronosRolling:
    """Tests for WatchlistScorer.score_kronos_rolling method."""

    @patch("big_a.report.scorer.load_config")
    @patch("big_a.report.scorer.KronosSignalGenerator")
    def test_score_kronos_rolling_valid(self, mock_kronos, mock_config):
        """Test rolling scoring with Kronos generates correct records."""
        from big_a.report.scorer import WatchlistScorer

        mock_config.return_value = {"kronos": {"lookback": 90}}

        mock_gen = MagicMock()
        mock_kronos.return_value = mock_gen

        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        instruments = ["SH688256"]
        idx = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "date"])
        data_df = pd.DataFrame({
            "open": np.random.uniform(30, 50, 200),
            "close": np.random.uniform(30, 50, 200),
        }, index=idx)
        mock_gen.load_data.return_value = data_df

        pred_df = pd.DataFrame({
            "close": [46.0, 46.5, 47.0],
        })
        mock_gen.predict.return_value = pred_df
        mock_gen.signal_mode = "mean"

        with patch("qlib.data.D") as mock_D:
            mock_D.calendar.return_value = pd.date_range("2024-01-01", periods=100, freq="B")

            scorer = WatchlistScorer()
            result = scorer.score_kronos_rolling(["SH688256"], n_days=3)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["date", "instrument", "score", "score_pct"]
        mock_gen.load_data.assert_called_once()


class TestWatchlistScorerScoreLightgbm:
    """Tests for WatchlistScorer.score_lightgbm method."""

    @patch("big_a.report.scorer.PROJECT_ROOT")
    @patch("big_a.report.scorer.load_model")
    @patch("big_a.report.scorer.load_config")
    @patch("big_a.report.scorer.create_dataset")
    @patch("big_a.report.scorer.predict_to_dataframe")
    def test_score_lightgbm_valid(self, mock_predict, mock_create_ds,
                                   mock_config, mock_load, mock_root):
        """Test scoring with LightGBM returns correct DataFrame."""
        from big_a.report.scorer import WatchlistScorer

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        mock_config.return_value = {
            "dataset": {
                "kwargs": {
                    "handler": {
                        "kwargs": {
                            "instruments": "csi300"
                        }
                    }
                }
            }
        }

        mock_dataset = MagicMock()
        mock_create_ds.return_value = mock_dataset

        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_root.__truediv__.return_value = mock_path

        preds_df = pd.DataFrame({
            "score": [0.82, 0.45, -0.12],
        }, index=pd.MultiIndex.from_tuples([
            (pd.Timestamp("2024-12-31"), "SH688256"),
            (pd.Timestamp("2024-12-31"), "SZ300476"),
            (pd.Timestamp("2024-12-31"), "SZ001309"),
        ], names=["date", "instrument"]))
        mock_predict.return_value = preds_df

        with patch("qlib.data.D") as mock_D:
            mock_D.calendar.return_value = pd.date_range("2024-01-01", periods=100, freq="B")

            scorer = WatchlistScorer(lookback_days=10)
            instruments = ["SH688256", "SZ300476", "SZ001309"]
            result = scorer.score_lightgbm(instruments)

            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["date", "instrument", "score"]
            assert len(result) == 3

            call_kwargs = mock_create_ds.call_args[0][0]
            assert call_kwargs["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] == instruments

    @patch("big_a.report.scorer.PROJECT_ROOT")
    def test_score_lightgbm_model_not_found(self, mock_root):
        """Test scoring when model file not found returns empty DataFrame."""
        from big_a.report.scorer import WatchlistScorer

        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_root.__truediv__.return_value = mock_path

        scorer = WatchlistScorer()
        result = scorer.score_lightgbm(["SH688256"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["date", "instrument", "score"]


class TestWatchlistScorerFetchMarketData:
    """Tests for WatchlistScorer.fetch_market_data method."""

    def test_fetch_market_data_valid(self):
        """Test fetching market data returns correct DataFrame."""
        from big_a.report.scorer import WatchlistScorer

        dates = pd.date_range("2024-12-18", periods=10, freq="B")
        instruments = ["SZ001309", "SZ300476", "SH688256"]
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

        raw_df = pd.DataFrame({
            "$open": np.random.uniform(30, 50, 30),
            "$high": np.random.uniform(30, 50, 30),
            "$low": np.random.uniform(30, 50, 30),
            "$close": np.random.uniform(30, 50, 30),
            "$volume": np.random.uniform(1e6, 1e7, 30),
            "$amount": np.random.uniform(1e8, 1e9, 30),
        }, index=idx)

        with patch("qlib.data.D") as mock_D:
            mock_D.calendar.return_value = dates
            mock_D.features.return_value = raw_df

            scorer = WatchlistScorer()
            result = scorer.fetch_market_data(instruments, n_days=10)

            assert isinstance(result, pd.DataFrame)
            assert isinstance(result.index, pd.MultiIndex)
            assert "change_pct" in result.columns
            assert "open" in result.columns
            assert "$open" not in result.columns
            assert len(result) == 30

    def test_fetch_market_data_empty(self):
        """Test fetching market data when no data available returns empty DataFrame."""
        from big_a.report.scorer import WatchlistScorer

        with patch("qlib.data.D") as mock_D:
            mock_D.calendar.return_value = pd.date_range("2024-01-01", periods=10, freq="B")
            mock_D.features.return_value = pd.DataFrame()

            scorer = WatchlistScorer()
            result = scorer.fetch_market_data(["SH688256"])

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


class TestWatchlistScorerComputePortfolio:
    """Tests for WatchlistScorer.compute_portfolio method."""

    def test_compute_portfolio_valid(self, sample_watchlist):
        """Test portfolio computation with valid scores."""
        from big_a.report.scorer import WatchlistScorer

        scores = pd.DataFrame({
            "instrument": ["SH688256", "SZ300476", "SZ001309"],
            "combined_score": [0.8, 0.5, 0.2],
        })

        with patch.object(WatchlistScorer, "fetch_market_data") as mock_fetch:
            market_df = pd.DataFrame({
                "close": [45.20, 32.50, 28.30],
                "factor": [1.0, 1.0, 1.0],
            }, index=pd.MultiIndex.from_product([
                [pd.Timestamp("2024-12-31")],
                ["SH688256", "SZ300476", "SZ001309"],
            ], names=["date", "instrument"]))
            mock_fetch.return_value = market_df

            scorer = WatchlistScorer(account=1_000_000)
            result = scorer.compute_portfolio(scores, sample_watchlist, topk=5, max_weight=0.25)

            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["instrument", "name", "weight", "allocation", "signal", "entry_price"]
            assert "CASH" in result["instrument"].values
            assert len(result) == 4

    def test_compute_portfolio_weight_capping(self, sample_watchlist):
        """Test portfolio computation with weight capping."""
        from big_a.report.scorer import WatchlistScorer

        scores = pd.DataFrame({
            "instrument": ["SH688256", "SZ300476", "SZ001309"],
            "combined_score": [0.9, 0.8, 0.7],
        })

        with patch.object(WatchlistScorer, "fetch_market_data") as mock_fetch:
            market_df = pd.DataFrame({
                "close": [45.20, 32.50, 28.30],
                "factor": [1.0, 1.0, 1.0],
            }, index=pd.MultiIndex.from_product([
                [pd.Timestamp("2024-12-31")],
                ["SH688256", "SZ300476", "SZ001309"],
            ], names=["date", "instrument"]))
            mock_fetch.return_value = market_df

            scorer = WatchlistScorer(account=1_000_000)
            result = scorer.compute_portfolio(scores, sample_watchlist, topk=5, max_weight=0.20)

            stock_weights = result[result["instrument"] != "CASH"]["weight"]
            assert all(w <= 0.20 for w in stock_weights)

    def test_compute_portfolio_all_positive_scores(self, sample_watchlist):
        """Test portfolio computation with all positive scores."""
        from big_a.report.scorer import WatchlistScorer

        scores = pd.DataFrame({
            "instrument": ["SH688256", "SZ300476", "SZ001309"],
            "combined_score": [0.8, 0.6, 0.4],
        })

        with patch.object(WatchlistScorer, "fetch_market_data") as mock_fetch:
            market_df = pd.DataFrame({
                "close": [45.20, 32.50, 28.30],
                "factor": [1.0, 1.0, 1.0],
            }, index=pd.MultiIndex.from_product([
                [pd.Timestamp("2024-12-31")],
                ["SH688256", "SZ300476", "SZ001309"],
            ], names=["date", "instrument"]))
            mock_fetch.return_value = market_df

            scorer = WatchlistScorer(account=1_000_000)
            result = scorer.compute_portfolio(scores, sample_watchlist, topk=5, max_weight=0.25)

            stocks = result[result["instrument"] != "CASH"]
            assert len(stocks) == 3
            assert all(signal in ["Buy", "Strong Buy"] for signal in stocks["signal"])

    def test_compute_portfolio_fewer_than_topk(self, sample_watchlist):
        """Test portfolio computation with fewer stocks than topk."""
        from big_a.report.scorer import WatchlistScorer

        scores = pd.DataFrame({
            "instrument": ["SH688256", "SZ300476"],
            "combined_score": [0.8, 0.5],
        })

        with patch.object(WatchlistScorer, "fetch_market_data") as mock_fetch:
            market_df = pd.DataFrame({
                "close": [45.20, 32.50],
                "factor": [1.0, 1.0],
            }, index=pd.MultiIndex.from_product([
                [pd.Timestamp("2024-12-31")],
                ["SH688256", "SZ300476"],
            ], names=["date", "instrument"]))
            mock_fetch.return_value = market_df

            scorer = WatchlistScorer(account=1_000_000)
            result = scorer.compute_portfolio(scores, sample_watchlist, topk=5, max_weight=0.25)

            stocks = result[result["instrument"] != "CASH"]
            assert len(stocks) == 2

    def test_compute_portfolio_empty_scores(self, sample_watchlist):
        """Test portfolio computation with empty scores returns empty DataFrame."""
        from big_a.report.scorer import WatchlistScorer

        scores = pd.DataFrame(columns=["instrument", "combined_score"])

        scorer = WatchlistScorer()
        result = scorer.compute_portfolio(scores, sample_watchlist)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestWatchlistScorerRun:
    """Tests for WatchlistScorer.run method."""

    @patch("big_a.report.scorer.init_qlib")
    @patch("big_a.report.scorer.load_config")
    def test_run_valid(self, mock_config, mock_init):
        """Test full run pipeline returns expected structure."""
        from big_a.report.scorer import WatchlistScorer

        mock_config.return_value = {"watchlist": {"SH600000": "浦发银行"}}

        with patch("qlib.data.D") as mock_D, \
             patch.object(WatchlistScorer, "load_watchlist") as mock_load_watchlist, \
             patch.object(WatchlistScorer, "score_kronos") as mock_kronos, \
             patch.object(WatchlistScorer, "score_kronos_rolling") as mock_kronos_rolling, \
             patch.object(WatchlistScorer, "score_lightgbm") as mock_lgb, \
             patch.object(WatchlistScorer, "fetch_market_data") as mock_fetch, \
             patch.object(WatchlistScorer, "compute_portfolio") as mock_portfolio:

            mock_D.calendar.return_value = pd.date_range("2024-01-01", periods=100, freq="B")
            mock_load_watchlist.return_value = {"SH600000": "浦发银行"}

            kronos_df = pd.DataFrame({
                "instrument": ["SH600000"],
                "score_pct": [2.0],
            })
            mock_kronos.return_value = kronos_df
            mock_kronos_rolling.return_value = pd.DataFrame()

            mock_lgb.return_value = pd.DataFrame(columns=["date", "instrument", "score"])

            mock_fetch.return_value = pd.DataFrame()

            portfolio_df = pd.DataFrame({
                "instrument": ["SH600000"],
                "name": ["浦发银行"],
                "weight": [0.5],
                "allocation": [500000],
                "signal": ["Buy"],
                "entry_price": [10.0],
            })
            mock_portfolio.return_value = portfolio_df

            scorer = WatchlistScorer()
            result = scorer.run()

            assert isinstance(result, dict)
            assert set(result.keys()) == {
                "watchlist", "kronos_scores", "kronos_trend",
                "lightgbm_scores", "lightgbm_trend",
                "market_data", "portfolio", "summary"
            }
            assert result["watchlist"] == {"SH600000": "浦发银行"}
            assert "summary" in result
            assert "total_stocks" in result["summary"]

    @patch("big_a.report.scorer.init_qlib")
    @patch("big_a.report.scorer.load_config")
    def test_run_empty_watchlist(self, mock_config, mock_init):
        """Test run with empty watchlist returns empty results."""
        from big_a.report.scorer import WatchlistScorer

        mock_config.return_value = {"watchlist": {}}

        scorer = WatchlistScorer()
        result = scorer.run()

        assert isinstance(result, dict)
        assert result["watchlist"] == {}
        assert result["kronos_scores"].empty
        assert result["portfolio"].empty
        assert result["summary"] == {}


class TestQualitativeScoring:
    """Tests for qualitative (hedge fund) analysis integration in WatchlistScorer."""

    def test_qualitative_true_adds_hedge_fund_analysis(self) -> None:
        """When qualitative=True, run() returns hedge_fund_analysis key."""
        from big_a.report.scorer import WatchlistScorer

        with patch("big_a.report.scorer.init_qlib") as mock_init, \
             patch.object(WatchlistScorer, "load_watchlist") as mock_load, \
             patch.object(WatchlistScorer, "fetch_market_data") as mock_fetch, \
             patch("big_a.report.scorer.HedgeFundSignalGenerator") as mock_hf_gen_cls, \
             patch("qlib.data.D") as mock_d:

            mock_load.return_value = {"SH600000": "浦发银行"}
            mock_fetch.return_value = pd.DataFrame()
            mock_init.return_value = None

            # Mock qlib calendar
            mock_calendar = [pd.Timestamp(f"2024-12-{i:02d}") for i in range(1, 32)]
            mock_d.calendar.return_value = mock_calendar

            # Mock HedgeFundSignalGenerator
            mock_gen = MagicMock()
            mock_hf_gen_cls.return_value = mock_gen
            mock_gen.generate_signals.return_value = {
                "signals": pd.DataFrame(
                    {"score": [0.5]},
                    index=pd.MultiIndex.from_tuples(
                        [(pd.Timestamp("2024-12-31"), "SH600000")],
                        names=["datetime", "instrument"],
                    ),
                ),
                "details": {
                    "SH600000": {
                        "technicals_agent": {"signal": "bullish", "confidence": 0.8, "reasoning": "RSI oversold"},
                        "valuation_agent": {"signal": "neutral", "confidence": 0.5, "reasoning": "Fair value"},
                        "warren_buffett_agent": {"signal": "bullish", "confidence": 0.9, "reasoning": "Strong moat"},
                    }
                },
            }

            scorer = WatchlistScorer(qualitative=True, skip_trend=True, skip_lightgbm=True)
            scorer._kronos_config = {
                "lookback": 90, "pred_len": 10, "device": "cpu",
                "tokenizer_id": "t", "model_id": "m", "max_context": 512, "signal_mode": "mean",
            }
            results = scorer.run()

            assert "hedge_fund_analysis" in results
            assert "details" in results["hedge_fund_analysis"]
            assert "SH600000" in results["hedge_fund_analysis"]["details"]
            mock_gen.generate_signals.assert_called_once()

    def test_qualitative_false_no_hedge_fund(self) -> None:
        """When qualitative=False (default), hedge_fund_analysis is empty dict."""
        from big_a.report.scorer import WatchlistScorer

        with patch("big_a.report.scorer.init_qlib") as mock_init, \
             patch.object(WatchlistScorer, "load_watchlist") as mock_load, \
             patch.object(WatchlistScorer, "fetch_market_data") as mock_fetch:

            mock_load.return_value = {"SH600000": "浦发银行"}
            mock_fetch.return_value = pd.DataFrame()
            mock_init.return_value = None

            scorer = WatchlistScorer(skip_trend=True, skip_lightgbm=True)
            scorer._kronos_config = {
                "lookback": 90, "pred_len": 10, "device": "cpu",
                "tokenizer_id": "t", "model_id": "m", "max_context": 512, "signal_mode": "mean",
            }
            results = scorer.run()

            # qualitative=False returns empty dict, not missing key
            assert results.get("hedge_fund_analysis") == {}


class TestFormatReport:
    """Tests for formatter functions."""

    def test_format_report_valid(self, sample_kronos_scores, sample_lightgbm_scores, sample_portfolio):
        """Test format_report with valid data."""
        from big_a.report.formatter import format_report

        results = {
            "watchlist": {"SH688256": "寒武纪", "SZ300476": "胜宏科技"},
            "kronos_scores": sample_kronos_scores,
            "lightgbm_scores": sample_lightgbm_scores,
            "kronos_trend": pd.DataFrame(),
            "lightgbm_trend": pd.DataFrame(),
            "market_data": pd.DataFrame(),
            "portfolio": sample_portfolio,
            "summary": {
                "total_stocks": 2,
                "bullish_count": 2,
                "bearish_count": 0,
                "avg_score": 0.5,
                "best_stock": "SH688256",
                "worst_stock": "SZ300476",
            },
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_report(results, console)
        output = buf.getvalue()

        assert len(output) > 0

    def test_format_scores_table(self, sample_kronos_scores, sample_lightgbm_scores):
        """Test format_scores_table renders correctly."""
        from big_a.report.formatter import format_scores_table

        results = {
            "watchlist": {"SH688256": "寒武纪", "SZ300476": "胜宏科技"},
            "kronos_scores": sample_kronos_scores,
            "lightgbm_scores": sample_lightgbm_scores,
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_scores_table(results, console)
        output = buf.getvalue()

        assert len(output) > 0
        assert "模型打分" in output

    def test_format_scores_table_empty(self):
        """Test format_scores_table with empty data shows message."""
        from big_a.report.formatter import format_scores_table

        results = {
            "watchlist": {"SH688256": "寒武纪"},
            "kronos_scores": pd.DataFrame(),
            "lightgbm_scores": pd.DataFrame(),
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_scores_table(results, console)
        output = buf.getvalue()

        assert "暂无模型打分数据" in output

    def test_format_portfolio(self, sample_portfolio):
        """Test format_portfolio renders correctly."""
        from big_a.report.formatter import format_portfolio

        results = {
            "portfolio": sample_portfolio,
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_portfolio(results, console)
        output = buf.getvalue()

        assert len(output) > 0
        assert "模拟持仓" in output

    def test_format_portfolio_empty(self):
        """Test format_portfolio with empty data shows message."""
        from big_a.report.formatter import format_portfolio

        results = {
            "portfolio": pd.DataFrame(),
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_portfolio(results, console)
        output = buf.getvalue()

        assert "暂无持仓数据" in output

    def test_format_summary(self):
        """Test format_summary renders correctly."""
        from big_a.report.formatter import format_summary

        results = {
            "summary": {
                "total_stocks": 3,
                "bullish_count": 2,
                "bearish_count": 1,
                "avg_score": 0.35,
                "best_stock": "SH688256",
                "worst_stock": "SZ001309",
            },
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_summary(results, console)
        output = buf.getvalue()

        assert len(output) > 0
        assert "看涨" in output
        assert "SH688256" in output

    def test_format_summary_empty(self):
        """Test format_summary with empty data shows message."""
        from big_a.report.formatter import format_summary

        results = {
            "summary": {},
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_summary(results, console)
        output = buf.getvalue()

        assert "暂无总结数据" in output

    def test_format_trend_tables(self, sample_kronos_scores):
        """Test format_trend_tables renders correctly."""
        from big_a.report.formatter import format_trend_tables

        trend_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-12-20", "2024-12-20", "2024-12-21", "2024-12-21"]),
            "instrument": ["SH688256", "SZ300476", "SH688256", "SZ300476"],
            "score": [0.5, 0.3, 0.6, 0.4],
            "score_pct": [2.0, 1.2, 2.5, 1.5],
        })

        results = {
            "watchlist": {"SH688256": "寒武纪", "SZ300476": "胜宏科技"},
            "kronos_trend": trend_df,
            "lightgbm_trend": pd.DataFrame(),
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_trend_tables(results, console)
        output = buf.getvalue()

        assert len(output) > 0

    def test_format_market_data(self, sample_market_data):
        """Test format_market_data renders correctly."""
        from big_a.report.formatter import format_market_data

        results = {
            "watchlist": {"SH688256": "寒武纪", "SZ300476": "胜宏科技"},
            "market_data": sample_market_data,
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_market_data(results, console)
        output = buf.getvalue()

        assert len(output) > 0
        assert "近10日行情" in output

    def test_format_market_data_empty(self):
        """Test format_market_data with empty data shows message."""
        from big_a.report.formatter import format_market_data

        results = {
            "watchlist": {"SH688256": "寒武纪"},
            "market_data": pd.DataFrame(),
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)

        format_market_data(results, console)
        output = buf.getvalue()

        assert "暂无行情数据" in output

    def test_format_qualitative_analysis_with_data(self):
        """Test format_qualitative_analysis renders agent signals."""
        from big_a.report.formatter import format_qualitative_analysis

        results = {
            "watchlist": {"SH600000": "浦发银行"},
            "hedge_fund_analysis": {
                "details": {
                    "SH600000": {
                        "technicals_agent": {"signal": "bullish", "confidence": 0.8, "reasoning": "RSI oversold, MACD golden cross forming"},
                        "valuation_agent": {"signal": "neutral", "confidence": 0.5, "reasoning": "Fair valuation based on historical percentiles"},
                        "warren_buffett_agent": {"signal": "bullish", "confidence": 0.9, "reasoning": "Strong moat and competitive advantages"},
                    }
                },
                "signals": pd.DataFrame(),
            },
        }

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        format_qualitative_analysis(results, console)
        output = buf.getvalue()

        assert "AI 定性分析" in output
        assert "技术分析" in output
        assert "看涨" in output
        assert "RSI" in output

    def test_format_qualitative_analysis_empty(self):
        """Test format_qualitative_analysis skips when no data."""
        from big_a.report.formatter import format_qualitative_analysis

        results = {"hedge_fund_analysis": {}}

        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        format_qualitative_analysis(results, console)
        output = buf.getvalue()

        assert "AI 定性分析" not in output


class TestFormatterHelpers:
    """Tests for formatter helper functions."""

    def test_format_score_positive(self):
        """Test _format_score with positive value."""
        from big_a.report.formatter import _format_score

        result = _format_score(0.85)
        assert result.plain == "+0.85"
        assert "green" in str(result.style)

    def test_format_score_negative(self):
        """Test _format_score with negative value."""
        from big_a.report.formatter import _format_score

        result = _format_score(-0.75)
        assert result.plain == "-0.75"
        assert "red" in str(result.style)

    def test_format_score_zero(self):
        """Test _format_score with zero value."""
        from big_a.report.formatter import _format_score

        result = _format_score(0.0)
        assert result.plain == "+0.00"

    def test_format_pct_positive(self):
        """Test _format_pct with positive value."""
        from big_a.report.formatter import _format_pct

        result = _format_pct(3.5)
        assert result.plain == "+3.5%"
        assert "green" in str(result.style)

    def test_format_pct_negative(self):
        """Test _format_pct with negative value."""
        from big_a.report.formatter import _format_pct

        result = _format_pct(-2.3)
        assert result.plain == "-2.3%"
        assert "red" in str(result.style)

    def test_format_amount_yi(self):
        """Test _format_amount with amount in billions."""
        from big_a.report.formatter import _format_amount

        result = _format_amount(5.5e8)
        assert "亿" in result
        assert "5.5" in result

    def test_format_amount_wan(self):
        """Test _format_amount with amount in ten-thousands."""
        from big_a.report.formatter import _format_amount

        result = _format_amount(5.5e4)
        assert "万" in result
        assert "5.5" in result

    def test_get_signal_label_strong_buy(self):
        """Test _get_signal_label for strong buy signal."""
        from big_a.report.formatter import _get_signal_label

        result = _get_signal_label(0.8)
        assert "强烈看涨" in result.plain
        assert "green" in str(result.style)

    def test_get_signal_label_buy(self):
        """Test _get_signal_label for buy signal."""
        from big_a.report.formatter import _get_signal_label

        result = _get_signal_label(0.3)
        assert "看涨" in result.plain
        assert "green" in str(result.style)

    def test_get_signal_label_sell(self):
        """Test _get_signal_label for sell signal."""
        from big_a.report.formatter import _get_signal_label

        result = _get_signal_label(-0.3)
        assert "看跌" in result.plain
        assert "red" in str(result.style)

    def test_get_signal_label_strong_sell(self):
        """Test _get_signal_label for strong sell signal."""
        from big_a.report.formatter import _get_signal_label

        result = _get_signal_label(-0.8)
        assert "强烈看跌" in result.plain
        assert "red" in str(result.style)


class TestConvertMarketUnits:
    """Tests for _convert_market_units helper function."""

    def test_convert_market_units(self):
        """Test conversion from Qlib units to market units."""
        from big_a.report.scorer import _convert_market_units

        df = pd.DataFrame({
            "close": [50.0],
            "open": [49.0],
            "high": [51.0],
            "low": [48.0],
            "factor": [0.5],
            "volume": [1000.0],
            "amount": [5000.0],
            "change_pct": [1.5],
        })

        result = _convert_market_units(df)

        assert result.loc[0, "close_raw"] == 100.0
        assert result.loc[0, "open_raw"] == 98.0
        assert result.loc[0, "high_raw"] == 102.0
        assert result.loc[0, "low_raw"] == 96.0
        assert result.loc[0, "amount_yuan"] == 5_000_000.0
        assert result.loc[0, "volume_shares"] == 50_000.0

        assert result.loc[0, "close"] == 50.0
        assert result.loc[0, "factor"] == 0.5
        assert result.loc[0, "volume"] == 1000.0


class TestFormatVolume:
    """Tests for _format_volume helper function."""

    @pytest.mark.parametrize("value,expected", [
        (1e8, "1.0亿股"),
        (5.5e8, "5.5亿股"),
        (1e4, "1.0万股"),
        (5.5e4, "5.5万股"),
        (999, "999股"),
    ])
    def test_format_volume(self, value, expected):
        """Test volume formatting for 亿股, 万股, 股 ranges."""
        from big_a.report.formatter import _format_volume

        result = _format_volume(value)
        assert result == expected
