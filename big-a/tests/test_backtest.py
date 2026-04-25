"""Tests for backtest engine."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from big_a.backtest.engine import (
    DEFAULT_BACKTEST_KWARGS,
    DEFAULT_EXCHANGE_KWARGS,
    compute_analysis,
    load_backtest_config,
    run_backtest,
)


@pytest.fixture
def sample_signal_df():
    dates = pd.date_range("2022-01-04", periods=20, freq="B")
    instruments = [f"SH60000{i}" for i in range(10)]
    multi_index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    return pd.DataFrame(
        {"score": np.random.randn(len(multi_index))},
        index=multi_index,
    )


@pytest.fixture
def sample_signal_series(sample_signal_df):
    return sample_signal_df["score"]


@pytest.fixture
def mock_report():
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
def sample_config():
    return {
        "strategy": {
            "kwargs": {"topk": 30, "n_drop": 3},
        },
        "backtest": {
            "start_time": "2022-01-04",
            "end_time": "2022-01-31",
            "account": 50000000,
            "benchmark": "SH000300",
            "exchange_kwargs": {"limit_threshold": 0.095},
        },
    }


def _mock_qlib_contrib(mock_bt_return):
    mock_bt = MagicMock(return_value=mock_bt_return)
    mock_strat_cls = MagicMock()
    mock_risk = MagicMock(return_value=pd.DataFrame(
        {"mean": [0.01], "std": [0.02]},
        index=["excess_return"],
    ))
    return mock_bt, mock_strat_cls, mock_risk


class TestRunBacktest:
    @patch("qlib.contrib.strategy.TopkDropoutStrategy")
    @patch("qlib.contrib.evaluate.backtest_daily")
    def test_runs_with_dataframe_signal(self, mock_bt, mock_strat_cls, sample_signal_df, mock_report):
        mock_bt.return_value = (mock_report, {"pos": MagicMock()})
        report, positions = run_backtest(sample_signal_df)

        mock_strat_cls.assert_called_once()
        call_kwargs = mock_strat_cls.call_args[1]
        assert call_kwargs["topk"] == 50
        assert call_kwargs["n_drop"] == 5
        pd.testing.assert_frame_equal(call_kwargs["signal"], sample_signal_df)

        mock_bt.assert_called_once()
        bt_kwargs = mock_bt.call_args[1]
        assert bt_kwargs["account"] == DEFAULT_BACKTEST_KWARGS["account"]
        assert bt_kwargs["benchmark"] == DEFAULT_BACKTEST_KWARGS["benchmark"]
        assert bt_kwargs["exchange_kwargs"]["limit_threshold"] == 0.095
        assert bt_kwargs["exchange_kwargs"]["deal_price"] == "close"

    @patch("qlib.contrib.strategy.TopkDropoutStrategy")
    @patch("qlib.contrib.evaluate.backtest_daily")
    def test_converts_series_to_dataframe(self, mock_bt, mock_strat_cls, sample_signal_series, mock_report):
        mock_bt.return_value = (mock_report, {})
        run_backtest(sample_signal_series)

        call_kwargs = mock_strat_cls.call_args[1]
        assert isinstance(call_kwargs["signal"], pd.DataFrame)
        assert "score" in call_kwargs["signal"].columns

    @patch("qlib.contrib.strategy.TopkDropoutStrategy")
    @patch("qlib.contrib.evaluate.backtest_daily")
    def test_uses_config_params(self, mock_bt, mock_strat_cls, sample_signal_df, sample_config, mock_report):
        mock_bt.return_value = (mock_report, {})
        run_backtest(sample_signal_df, config=sample_config)

        call_kwargs = mock_strat_cls.call_args[1]
        assert call_kwargs["topk"] == 30
        assert call_kwargs["n_drop"] == 3

        bt_kwargs = mock_bt.call_args[1]
        assert bt_kwargs["account"] == 50000000
        assert bt_kwargs["start_time"] == "2022-01-04"
        assert bt_kwargs["end_time"] == "2022-01-31"

    @patch("qlib.contrib.strategy.TopkDropoutStrategy")
    @patch("qlib.contrib.evaluate.backtest_daily")
    def test_derives_time_from_signal_if_missing(self, mock_bt, mock_strat_cls, sample_signal_df, mock_report):
        mock_bt.return_value = (mock_report, {})
        run_backtest(sample_signal_df, config={"strategy": {"kwargs": {}}})

        bt_kwargs = mock_bt.call_args[1]
        assert bt_kwargs["start_time"] == sample_signal_df.index.get_level_values("datetime").min()
        assert bt_kwargs["end_time"] == sample_signal_df.index.get_level_values("datetime").max()

    @patch("qlib.contrib.strategy.TopkDropoutStrategy")
    @patch("qlib.contrib.evaluate.backtest_daily")
    def test_merges_exchange_kwargs(self, mock_bt, mock_strat_cls, sample_signal_df, mock_report):
        mock_bt.return_value = (mock_report, {})
        config = {"backtest": {"exchange_kwargs": {"limit_threshold": 0.05}}}
        run_backtest(sample_signal_df, config=config)

        bt_kwargs = mock_bt.call_args[1]
        assert bt_kwargs["exchange_kwargs"]["limit_threshold"] == 0.05
        assert bt_kwargs["exchange_kwargs"]["deal_price"] == "close"

    @patch("qlib.contrib.strategy.TopkDropoutStrategy")
    @patch("qlib.contrib.evaluate.backtest_daily")
    def test_renames_first_column_to_score(self, mock_bt, mock_strat_cls, mock_report):
        dates = pd.date_range("2022-01-04", periods=5, freq="B")
        instruments = ["SH600000", "SH600001"]
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        signal = pd.DataFrame({"prediction": np.random.randn(len(idx))}, index=idx)

        mock_bt.return_value = (mock_report, {})
        run_backtest(signal)

        call_kwargs = mock_strat_cls.call_args[1]
        assert "score" in call_kwargs["signal"].columns


class TestComputeAnalysis:
    def test_returns_dataframe(self, mock_report):
        with patch("qlib.contrib.evaluate.risk_analysis") as mock_risk:
            mock_risk.return_value = pd.DataFrame(
                {"mean": [0.01], "std": [0.02], "sharpe": [0.5]},
                index=["excess_return"],
            )
            result = compute_analysis(mock_report)
            assert isinstance(result, pd.DataFrame)
            assert mock_risk.call_count == 2


class TestDefaultParams:
    def test_exchange_defaults(self):
        assert DEFAULT_EXCHANGE_KWARGS["limit_threshold"] == 0.095
        assert DEFAULT_EXCHANGE_KWARGS["deal_price"] == "close"
        assert DEFAULT_EXCHANGE_KWARGS["open_cost"] == 0.0005
        assert DEFAULT_EXCHANGE_KWARGS["close_cost"] == 0.0015
        assert DEFAULT_EXCHANGE_KWARGS["min_cost"] == 5

    def test_backtest_defaults(self):
        assert DEFAULT_BACKTEST_KWARGS["account"] == 100000000
        assert DEFAULT_BACKTEST_KWARGS["benchmark"] == "SH000300"


class TestLoadBacktestConfig:
    def test_loads_topk_csi300(self):
        config = load_backtest_config("configs/backtest/topk_csi300.yaml")
        assert "strategy" in config
        assert "backtest" in config
        assert config["strategy"]["kwargs"]["topk"] == 50
        assert config["backtest"]["account"] == 100000000

    def test_loads_default_path(self):
        config = load_backtest_config()
        assert "strategy" in config
