"""Tests for rolling backtest module."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from big_a.backtest.rolling import (
    RollingBacktester,
    WindowResult,
    aggregate_results,
    generate_windows,
    run_rolling,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signal():
    dates = pd.date_range("2022-01-04", periods=20, freq="B")
    instruments = [f"SH60000{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    return pd.DataFrame({"score": np.random.randn(len(idx))}, index=idx)


@pytest.fixture
def sample_report():
    dates = pd.date_range("2022-01-04", periods=20, freq="B")
    return pd.DataFrame(
        {
            "return": np.random.randn(20) * 0.01,
            "bench": np.random.randn(20) * 0.005,
            "cost": np.random.randn(20) * 0.001,
            "turnover": np.random.rand(20) * 0.1,
        },
        index=dates,
    )


@pytest.fixture
def mock_window():
    return {
        "window_idx": 0,
        "train_start": "2010-01-01",
        "train_end": "2014-12-31",
        "valid_start": "2015-01-01",
        "valid_end": "2015-12-31",
        "test_start": "2016-01-01",
        "test_end": "2016-12-31",
    }


@pytest.fixture
def sample_window_results(sample_report):
    return [
        WindowResult(
            window_idx=0,
            train_start="2010-01-01",
            train_end="2014-12-31",
            valid_start="2015-01-01",
            valid_end="2015-12-31",
            test_start="2016-01-01",
            test_end="2016-12-31",
            ic=0.05,
            rank_ic=0.06,
            icir=1.2,
            sharpe=1.5,
            max_drawdown=0.08,
            report=sample_report,
        ),
        WindowResult(
            window_idx=1,
            train_start="2011-01-01",
            train_end="2015-12-31",
            valid_start="2016-01-01",
            valid_end="2016-12-31",
            test_start="2017-01-01",
            test_end="2017-12-31",
            ic=0.04,
            rank_ic=0.05,
            icir=0.9,
            sharpe=1.1,
            max_drawdown=0.12,
            report=sample_report,
        ),
    ]


# ---------------------------------------------------------------------------
# Test generate_windows
# ---------------------------------------------------------------------------

class TestGenerateWindows:
    def test_generates_correct_count(self):
        windows = generate_windows(
            start_year=2010, end_year=2024,
            train_years=5, valid_years=1, test_years=1, step_years=1,
        )
        assert len(windows) == 9

    def test_first_window_boundaries(self):
        windows = generate_windows(start_year=2010, end_year=2024)
        w = windows[0]
        assert w["train_start"] == "2010-01-01"
        assert w["train_end"] == "2014-12-31"
        assert w["valid_start"] == "2015-01-01"
        assert w["valid_end"] == "2015-12-31"
        assert w["test_start"] == "2016-01-01"
        assert w["test_end"] == "2016-12-31"

    def test_last_window_does_not_exceed_end(self):
        windows = generate_windows(start_year=2010, end_year=2024)
        last = windows[-1]
        assert last["test_end"] <= "2024-12-31"

    def test_empty_when_data_too_short(self):
        windows = generate_windows(
            start_year=2020, end_year=2022,
            train_years=5, valid_years=1, test_years=1,
        )
        assert len(windows) == 0

    def test_step_years_affects_count(self):
        w1 = generate_windows(start_year=2010, end_year=2024, step_years=1)
        w2 = generate_windows(start_year=2010, end_year=2024, step_years=2)
        assert len(w2) < len(w1)

    def test_window_idx_sequential(self):
        windows = generate_windows(start_year=2010, end_year=2024)
        indices = [w["window_idx"] for w in windows]
        assert indices == list(range(len(windows)))

    def test_custom_window_sizes(self):
        windows = generate_windows(
            start_year=2010, end_year=2024,
            train_years=3, valid_years=2, test_years=1, step_years=1,
        )
        w = windows[0]
        assert w["train_end"] == "2012-12-31"
        assert w["valid_start"] == "2013-01-01"
        assert w["valid_end"] == "2014-12-31"
        assert w["test_start"] == "2015-01-01"


# ---------------------------------------------------------------------------
# Test RollingBacktester
# ---------------------------------------------------------------------------

class TestRollingBacktester:
    def test_init_defaults(self):
        rb = RollingBacktester()
        assert rb.model_type == "lightgbm"
        assert rb.train_years == 5
        assert rb.valid_years == 1
        assert rb.test_years == 1
        assert rb.step_years == 1

    def test_init_custom(self):
        rb = RollingBacktester(
            model_type="kronos",
            train_years=3,
            start_year=2015,
            end_year=2023,
        )
        assert rb.model_type == "kronos"
        assert rb.train_years == 3
        assert rb.start_year == 2015

    @patch("big_a.backtest.rolling.RollingBacktester._run_window")
    def test_run_rolling_returns_window_results(self, mock_run, sample_window_results):
        mock_run.side_effect = sample_window_results

        rb = RollingBacktester(start_year=2010, end_year=2017)
        results = rb.run_rolling()

        assert len(results) == len(sample_window_results)
        assert all(isinstance(r, WindowResult) for r in results)

    @patch("big_a.backtest.rolling.RollingBacktester._run_window")
    def test_run_rolling_empty_windows(self, mock_run):
        rb = RollingBacktester(start_year=2020, end_year=2022, train_years=5)
        results = rb.run_rolling()
        assert results == []
        mock_run.assert_not_called()

    @patch("big_a.backtest.engine.run_backtest")
    def test_run_window_lightgbm(self, mock_bt, mock_window, sample_signal, sample_report):
        mock_bt.return_value = (sample_report, {})

        with patch("big_a.backtest.rolling.RollingBacktester._run_lightgbm_window") as mock_lgb:
            mock_lgb.return_value = sample_signal
            rb = RollingBacktester(model_type="lightgbm")
            result = rb._run_window(mock_window, {})

        assert isinstance(result, WindowResult)
        assert result.window_idx == 0
        mock_lgb.assert_called_once_with(mock_window)

    @patch("big_a.backtest.engine.run_backtest")
    def test_run_window_kronos(self, mock_bt, mock_window, sample_signal, sample_report):
        mock_bt.return_value = (sample_report, {})

        with patch("big_a.backtest.rolling.RollingBacktester._run_kronos_window") as mock_kron:
            mock_kron.return_value = sample_signal
            rb = RollingBacktester(model_type="kronos")
            result = rb._run_window(mock_window, {})

        assert isinstance(result, WindowResult)
        assert result.window_idx == 0
        mock_kron.assert_called_once_with(mock_window)

    def test_run_window_empty_signal(self, mock_window):
        rb = RollingBacktester()
        with patch.object(rb, "_run_lightgbm_window", return_value=pd.DataFrame()):
            result = rb._run_window(mock_window, {})
        assert result.ic != result.ic  # NaN check

    def test_patch_dataset_segments(self, mock_window):
        rb = RollingBacktester()
        config = {
            "dataset": {
                "kwargs": {
                    "handler": {
                        "kwargs": {
                            "start_time": "old",
                            "end_time": "old",
                            "fit_start_time": "old",
                            "fit_end_time": "old",
                        }
                    },
                    "segments": {
                        "train": ["old", "old"],
                        "valid": ["old", "old"],
                        "test": ["old", "old"],
                    }
                }
            }
        }
        rb._patch_dataset_segments(config, mock_window)
        h = config["dataset"]["kwargs"]["handler"]["kwargs"]
        assert h["start_time"] == "2010-01-01"
        assert h["end_time"] == "2016-12-31"
        s = config["dataset"]["kwargs"]["segments"]
        assert s["test"] == ["2016-01-01", "2016-12-31"]

    @patch("big_a.models.lightgbm_model.create_dataset")
    @patch("big_a.models.lightgbm_model.create_model")
    @patch("big_a.models.lightgbm_model.predict_to_dataframe")
    @patch("big_a.qlib_config.init_qlib")
    def test_lightgbm_window_pipeline(self, mock_init, mock_pred, mock_create_model, mock_create_dataset, mock_window, sample_signal):
        mock_pred.return_value = sample_signal

        rb = RollingBacktester()
        result = rb._run_lightgbm_window(mock_window)

        mock_create_model.assert_called_once()
        mock_create_model.return_value.fit.assert_called_once()
        mock_pred.assert_called_once()

    @patch("big_a.models.kronos.KronosSignalGenerator")
    @patch("big_a.qlib_config.init_qlib")
    def test_kronos_window_pipeline(self, mock_init, mock_gen_cls, mock_window, sample_signal):
        mock_gen = MagicMock()
        mock_gen.generate_signals.return_value = sample_signal
        mock_gen_cls.return_value = mock_gen

        rb = RollingBacktester(model_type="kronos")
        with patch.object(rb, "_get_instruments", return_value=["SH600000"]):
            result = rb._run_kronos_window(mock_window)

        mock_gen.load_model.assert_called_once()
        mock_gen.generate_signals.assert_called_once()


# ---------------------------------------------------------------------------
# Test aggregate_results
# ---------------------------------------------------------------------------

class TestAggregateResults:
    def test_empty_results(self):
        agg = aggregate_results([])
        assert agg["mean_ic"] != agg["mean_ic"]  # NaN
        assert len(agg["summary_df"]) == 0
        assert agg["combined_report"] is None

    def test_aggregates_metrics(self, sample_window_results):
        agg = aggregate_results(sample_window_results)
        assert len(agg["summary_df"]) == 2
        assert abs(agg["mean_ic"] - 0.045) < 1e-10
        assert abs(agg["mean_sharpe"] - 1.3) < 1e-10
        assert abs(agg["mean_max_drawdown"] - 0.10) < 1e-10

    def test_combined_report(self, sample_window_results):
        agg = aggregate_results(sample_window_results)
        assert agg["combined_report"] is not None
        assert len(agg["combined_report"]) == 40  # 2 windows × 20 rows

    def test_handles_none_reports(self):
        results = [
            WindowResult(
                window_idx=0,
                train_start="2010-01-01",
                train_end="2014-12-31",
                valid_start="2015-01-01",
                valid_end="2015-12-31",
                test_start="2016-01-01",
                test_end="2016-12-31",
                ic=0.05,
                sharpe=1.5,
            ),
        ]
        agg = aggregate_results(results)
        assert agg["combined_report"] is None
        assert agg["mean_ic"] == 0.05

    def test_summary_df_columns(self, sample_window_results):
        agg = aggregate_results(sample_window_results)
        df = agg["summary_df"]
        expected_cols = {"window", "test_start", "test_end", "ic", "rank_ic", "icir", "sharpe", "max_drawdown"}
        assert expected_cols == set(df.columns)


# ---------------------------------------------------------------------------
# Test run_rolling convenience function
# ---------------------------------------------------------------------------

class TestRunRolling:
    @patch("big_a.backtest.rolling.RollingBacktester.run_rolling")
    def test_delegates_to_rolling_backtester(self, mock_run, sample_window_results):
        mock_run.return_value = sample_window_results
        result = run_rolling(model_type="lightgbm")

        assert "summary_df" in result
        assert "mean_ic" in result
        mock_run.assert_called_once()

    @patch("big_a.backtest.rolling.RollingBacktester.run_rolling")
    def test_uses_config_file(self, mock_run, sample_window_results):
        mock_run.return_value = sample_window_results
        run_rolling(model_type="kronos", config_path="configs/backtest/rolling_csi300.yaml")
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Test WindowResult dataclass
# ---------------------------------------------------------------------------

class TestWindowResult:
    def test_default_values(self):
        r = WindowResult(
            window_idx=0,
            train_start="2010-01-01",
            train_end="2014-12-31",
            valid_start="2015-01-01",
            valid_end="2015-12-31",
            test_start="2016-01-01",
            test_end="2016-12-31",
        )
        assert r.ic != r.ic  # NaN
        assert r.report is None
        assert r.signal is None

    def test_stores_all_fields(self, sample_report, sample_signal):
        r = WindowResult(
            window_idx=0,
            train_start="2010-01-01",
            train_end="2014-12-31",
            valid_start="2015-01-01",
            valid_end="2015-12-31",
            test_start="2016-01-01",
            test_end="2016-12-31",
            ic=0.05,
            rank_ic=0.06,
            icir=1.2,
            sharpe=1.5,
            max_drawdown=0.08,
            report=sample_report,
            signal=sample_signal,
        )
        assert r.ic == 0.05
        assert r.sharpe == 1.5
        assert len(r.report) == 20
        assert "score" in r.signal.columns
