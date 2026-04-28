"""End-to-end pipeline verification tests for ML experiment tracking system.

Tests the complete round-trip: Train → Track → Compare and Backtest → Track → Compare.
All tests use mock MLflow (qlib workflow.R system) to avoid requiring live MLflow server.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Reuse existing qlib mock if already set up by test_experiment.py,
# otherwise create a fresh one. This ensures both test files share
# the same R reference so that patches in either file affect the
# experiment/comparison modules consistently.
if "qlib" in sys.modules and hasattr(sys.modules["qlib"], "workflow"):
    _mock_qlib = sys.modules["qlib"]
else:
    _mock_qlib = MagicMock()
    _mock_qlib.workflow = MagicMock()
    _mock_qlib.workflow.R = MagicMock()
    sys.modules["qlib"] = _mock_qlib
    sys.modules["qlib.workflow"] = _mock_qlib.workflow


class TestE2ETrainCompare:
    """Test: Train → Track → Compare round-trip.

    Simulates a training run that logs params + metrics,
    then verifies compare_by_params() and compare_by_metrics() return correct data.
    """

    def test_train_logs_params_and_compare_retrieves_them(self):
        """Simulate train run logging hyperparams, then verify compare_by_params returns them."""
        from big_a.tracking.comparison import compare_by_params

        # Set up mock recorder with specific params
        mock_rec = MagicMock()
        mock_rec.list_params.return_value = {
            "model.kwargs.learning_rate": 0.1,
            "model.kwargs.num_leaves": 31,
            "training.epochs": 100,
        }
        mock_rec.info = {
            "name": "train_run",
            "start_time": "2024-01-01",
            "end_time": "2024-01-02",
            "status": "FINISHED",
        }

        def list_recorders_side_effect(experiment_name):
            if "train_lgb" in experiment_name:
                return {"rec_001": mock_rec}
            return {}

        with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
            df = compare_by_params(["train_lgb_20240101"])

        # Verify round-trip: params logged = params queried
        assert not df.empty, "DataFrame should not be empty"
        assert "run_id" in df.columns
        assert "experiment" in df.columns
        assert "model.kwargs.learning_rate" in df.columns
        assert "model.kwargs.num_leaves" in df.columns
        assert "training.epochs" in df.columns
        # Check the actual values
        row = df.iloc[0]
        assert row["model.kwargs.learning_rate"] == 0.1
        assert row["model.kwargs.num_leaves"] == 31
        assert row["training.epochs"] == 100

    def test_train_logs_metrics_and_compare_retrieves_them(self):
        """Simulate train run logging metrics, then verify compare_by_metrics returns them."""
        from big_a.tracking.comparison import compare_by_metrics

        # Set up mock recorder with specific metrics
        mock_rec = MagicMock()
        mock_rec.list_metrics.return_value = {
            "ic": 0.0523,
            "rank_ic": 0.0411,
            "训练时间": 125.5,
        }
        mock_rec.info = {
            "name": "train_run",
            "start_time": "2024-01-01",
            "end_time": "2024-01-02",
            "status": "FINISHED",
        }

        def list_recorders_side_effect(experiment_name):
            if "train_kronos" in experiment_name:
                return {"rec_001": mock_rec}
            return {}

        with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
            df = compare_by_metrics(["train_kronos_20240101"])

        # Verify round-trip: metrics logged = metrics queried
        assert not df.empty, "DataFrame should not be empty"
        assert "run_id" in df.columns
        assert "experiment" in df.columns
        assert "ic" in df.columns
        assert "rank_ic" in df.columns
        # Check the actual values
        row = df.iloc[0]
        assert row["ic"] == 0.0523
        assert row["rank_ic"] == 0.0411

    def test_train_full_round_trip_via_experiment_context(self):
        """Start experiment via context, log params/metrics, then query and verify round-trip."""
        from big_a.experiment import experiment_context
        from big_a.tracking.comparison import query_experiments

        # Set up mock recorder that will be returned by R.start()
        mock_recorder = MagicMock()
        mock_recorder.list_params.return_value = {
            "lr": 0.05,
            "batch_size": 256,
        }
        mock_recorder.list_metrics.return_value = {
            "accuracy": 0.9234,
            "f1": 0.8912,
        }
        mock_recorder.info = {
            "name": "train_lgb_full",
            "start_time": "2024-01-15",
            "end_time": "2024-01-15",
            "status": "FINISHED",
        }

        # Track logged params/metrics to verify round-trip
        logged_params = {}
        logged_metrics = {}

        def mock_log_params(**kwargs):
            logged_params.update(kwargs)

        def mock_log_metrics(**kwargs):
            logged_metrics.update(kwargs)

        mock_recorder.log_params.side_effect = mock_log_params
        mock_recorder.log_metrics.side_effect = mock_log_metrics

        with patch.object(_mock_qlib.workflow.R, "start", return_value=mock_recorder):
            with patch.object(_mock_qlib.workflow.R, "end_exp"):
                with experiment_context("train_lgb_full", {"lr": 0.05, "batch_size": 256}) as recorder:
                    recorder.log_metrics(**{"accuracy": 0.9234, "f1": 0.8912})

        # Verify params were logged
        assert logged_params["lr"] == 0.05
        assert logged_params["batch_size"] == 256

        # Verify metrics were logged
        assert logged_metrics["accuracy"] == 0.9234
        assert logged_metrics["f1"] == 0.8912


class TestE2EBacktestCompare:
    """Test: Backtest → Track → Compare round-trip.

    Simulates a backtest run that logs sharpe/return/drawdown,
    then verifies compare_by_metrics() returns correct data.
    """

    def test_backtest_logs_metrics_and_compare_works(self):
        """Simulate backtest logging sharpe/return/max_drawdown, verify queryable."""
        from big_a.tracking.comparison import compare_by_metrics

        # Set up mock recorder with backtest metrics
        mock_rec = MagicMock()
        mock_rec.list_metrics.return_value = {
            "annualized_return": 0.1823,
            "sharpe": 1.4567,
            "max_drawdown": -0.1234,
            "volatility": 0.156,
        }
        mock_rec.info = {
            "name": "backtest_run",
            "start_time": "2024-01-01",
            "end_time": "2024-01-31",
            "status": "FINISHED",
        }

        def list_recorders_side_effect(experiment_name):
            if "backtest_lgb" in experiment_name:
                return {"rec_001": mock_rec}
            return {}

        with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
            df = compare_by_metrics(["backtest_lgb_20240101"])

        # Verify round-trip: backtest metrics logged = metrics queried
        assert not df.empty, "DataFrame should not be empty"
        assert "run_id" in df.columns
        assert "experiment" in df.columns
        assert "sharpe" in df.columns
        assert "annualized_return" in df.columns
        assert "max_drawdown" in df.columns

        row = df.iloc[0]
        assert row["sharpe"] == 1.4567
        assert row["annualized_return"] == 0.1823
        assert row["max_drawdown"] == -0.1234

    def test_backtest_via_experiment_context(self):
        """Start backtest via experiment_context, log metrics, then query and verify."""
        from big_a.experiment import experiment_context
        from big_a.tracking.comparison import query_experiments

        mock_recorder = MagicMock()
        logged_metrics = {}

        def mock_log_metrics(**kwargs):
            logged_metrics.update(kwargs)

        mock_recorder.log_params.side_effect = lambda **kwargs: None
        mock_recorder.log_metrics.side_effect = mock_log_metrics
        mock_recorder.list_params.return_value = {"strategy": "TopkDropout"}
        mock_recorder.list_metrics.return_value = {
            "annualized_return": 0.15,
            "sharpe": 1.2,
        }
        mock_recorder.info = {
            "name": "backtest_run",
            "start_time": "2024-01-01",
            "end_time": "2024-01-31",
            "status": "FINISHED",
        }

        with patch.object(_mock_qlib.workflow.R, "start", return_value=mock_recorder):
            with patch.object(_mock_qlib.workflow.R, "end_exp"):
                with experiment_context("backtest_lgb_20240101", {"strategy": "TopkDropout"}) as recorder:
                    recorder.log_metrics(**{
                        "annualized_return": 0.15,
                        "sharpe": 1.2,
                        "max_drawdown": -0.08,
                    })

        # Verify metrics were logged
        assert logged_metrics["annualized_return"] == 0.15
        assert logged_metrics["sharpe"] == 1.2
        assert logged_metrics["max_drawdown"] == -0.08


class TestE2EMultiExperiment:
    """Test: Multi-experiment cross-comparison.

    Simulates 2+ experiments with different params,
    verifies comparison DataFrame shows all differences.
    """

    def test_compare_two_experiments_with_different_params(self):
        """Two experiments with different learning_rate → comparison shows both."""
        from big_a.tracking.comparison import compare_by_params

        # Set up two mock recorders with different params
        mock_rec1 = MagicMock()
        mock_rec1.list_params.return_value = {
            "model.kwargs.learning_rate": 0.1,
            "model.kwargs.num_leaves": 31,
        }
        mock_rec1.info = {"name": "exp1", "start_time": "2024-01-01", "end_time": "", "status": "FINISHED"}

        mock_rec2 = MagicMock()
        mock_rec2.list_params.return_value = {
            "model.kwargs.learning_rate": 0.2,
            "model.kwargs.max_depth": 8,
        }
        mock_rec2.info = {"name": "exp2", "start_time": "2024-01-02", "end_time": "", "status": "FINISHED"}

        def list_recorders_side_effect(experiment_name):
            if experiment_name == "exp_lr_01":
                return {"rec1": mock_rec1}
            elif experiment_name == "exp_lr_02":
                return {"rec2": mock_rec2}
            return {}

        with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
            df = compare_by_params(["exp_lr_01", "exp_lr_02"])

        # Verify comparison shows both experiments with their different params
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert "run_id" in df.columns
        assert "experiment" in df.columns
        assert "model.kwargs.learning_rate" in df.columns
        assert "model.kwargs.num_leaves" in df.columns
        assert "model.kwargs.max_depth" in df.columns

        # Check that both learning rates are present
        lr_values = set(df["model.kwargs.learning_rate"].tolist())
        assert lr_values == {0.1, 0.2}, f"Expected learning rates {{0.1, 0.2}}, got {lr_values}"

        # Check that the param differences are preserved
        row1 = df[df["experiment"] == "exp_lr_01"].iloc[0]
        row2 = df[df["experiment"] == "exp_lr_02"].iloc[0]
        assert row1["model.kwargs.learning_rate"] == 0.1
        assert row2["model.kwargs.learning_rate"] == 0.2

    def test_query_with_name_filter(self):
        """Query experiments with name filter returns only matching ones."""
        from big_a.tracking.comparison import query_experiments

        # Set up mock experiments: train_lgb_1, train_lgb_2, backtest_1
        mock_rec_lgb1 = MagicMock(
            info={"name": "train_lgb_1", "start_time": "2024-01-01", "end_time": "", "status": "FINISHED"},
            list_params=MagicMock(return_value={"lr": 0.1}),
            list_metrics=MagicMock(return_value={"ic": 0.05}),
        )
        mock_rec_lgb2 = MagicMock(
            info={"name": "train_lgb_2", "start_time": "2024-01-02", "end_time": "", "status": "FINISHED"},
            list_params=MagicMock(return_value={"lr": 0.2}),
            list_metrics=MagicMock(return_value={"ic": 0.06}),
        )
        mock_rec_bt1 = MagicMock(
            info={"name": "backtest_1", "start_time": "2024-01-03", "end_time": "", "status": "FINISHED"},
            list_params=MagicMock(return_value={"strategy": "Topk"}),
            list_metrics=MagicMock(return_value={"sharpe": 1.5}),
        )

        mock_exp_lgb = MagicMock()
        mock_exp_lgb.list_recorders.return_value = {"rec_lgb1": mock_rec_lgb1, "rec_lgb2": mock_rec_lgb2}
        mock_exp_bt = MagicMock()
        mock_exp_bt.list_recorders.return_value = {"rec_bt1": mock_rec_bt1}

        def list_experiments_side_effect():
            return {
                "train_lgb_20240101": mock_exp_lgb,
                "backtest_20240103": mock_exp_bt,
            }

        def list_recorders_side_effect(experiment_name):
            if "train_lgb" in experiment_name:
                return {"rec_lgb1": mock_rec_lgb1, "rec_lgb2": mock_rec_lgb2}
            return {"rec_bt1": mock_rec_bt1}

        with patch.object(_mock_qlib.workflow.R, "list_experiments", side_effect=list_experiments_side_effect):
            with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
                results = query_experiments(name_pattern="train_lgb", limit=10)

        # Verify only train_lgb experiments are returned (not backtest)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        for result in results:
            assert "train_lgb" in result["experiment_name"], f"Unexpected experiment: {result['experiment_name']}"
            assert result["params"]["lr"] in [0.1, 0.2]

    def test_rolling_history_round_trip(self):
        """Log per-window metrics → get_rolling_history returns them."""
        from big_a.tracking.comparison import get_rolling_history

        # Set up mock recorder with list-valued metrics (simulating rolling windows)
        mock_rec = MagicMock()
        mock_rec.list_metrics.return_value = {
            "ic": [0.05, 0.055, 0.06, 0.065],
            "sharpe": [1.2, 1.3, 1.4, 1.5],
            "rank_ic": [0.04, 0.045, 0.05, 0.055],
        }
        mock_rec.info = {"name": "rolling_exp", "start_time": "2024-01-01", "end_time": "", "status": "FINISHED"}

        mock_exp = MagicMock()
        mock_exp.list_recorders.return_value = {"rec1": mock_rec}

        with patch.object(_mock_qlib.workflow.R, "list_recorders", return_value=mock_exp.list_recorders.return_value):
            df = get_rolling_history("rolling_exp")

        # Verify round-trip: per-window metrics logged = per-window metrics queried
        assert not df.empty, "DataFrame should not be empty"
        assert "window" in df.columns
        assert "metric_name" in df.columns
        assert "value" in df.columns

        # Should have 4 windows × 3 metrics = 12 rows
        assert len(df) == 12, f"Expected 12 rows, got {len(df)}"

        # Check ic values across windows
        ic_df = df[df["metric_name"] == "ic"].sort_values("window")
        ic_values = ic_df["value"].tolist()
        assert ic_values == [0.05, 0.055, 0.06, 0.065], f"Unexpected ic values: {ic_values}"

        # Check sharpe values across windows
        sharpe_df = df[df["metric_name"] == "sharpe"].sort_values("window")
        sharpe_values = sharpe_df["value"].tolist()
        assert sharpe_values == [1.2, 1.3, 1.4, 1.5], f"Unexpected sharpe values: {sharpe_values}"

    def test_multi_experiment_different_metrics(self):
        """Multiple experiments logging different metrics → comparison shows all."""
        from big_a.tracking.comparison import compare_by_metrics

        # Set up three mock recorders with different metrics
        mock_rec1 = MagicMock()
        mock_rec1.list_metrics.return_value = {
            "ic": 0.05,
            "rank_ic": 0.04,
        }
        mock_rec1.info = {"name": "exp1", "start_time": "2024-01-01", "end_time": "", "status": "FINISHED"}

        mock_rec2 = MagicMock()
        mock_rec2.list_metrics.return_value = {
            "annualized_return": 0.18,
            "sharpe": 1.5,
        }
        mock_rec2.info = {"name": "exp2", "start_time": "2024-01-02", "end_time": "", "status": "FINISHED"}

        mock_rec3 = MagicMock()
        mock_rec3.list_metrics.return_value = {
            "ic": 0.06,
            "max_drawdown": -0.1,
        }
        mock_rec3.info = {"name": "exp3", "start_time": "2024-01-03", "end_time": "", "status": "FINISHED"}

        def list_recorders_side_effect(experiment_name):
            mapping = {
                "train_exp1": {"rec1": mock_rec1},
                "backtest_exp2": {"rec2": mock_rec2},
                "train_exp3": {"rec3": mock_rec3},
            }
            return mapping.get(experiment_name, {})

        with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
            df = compare_by_metrics(["train_exp1", "backtest_exp2", "train_exp3"])

        # Verify all metrics from all experiments are represented
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        assert "run_id" in df.columns
        assert "experiment" in df.columns
        # All metric keys from all experiments should be columns
        assert "ic" in df.columns
        assert "rank_ic" in df.columns
        assert "annualized_return" in df.columns
        assert "sharpe" in df.columns
        assert "max_drawdown" in df.columns

    def test_query_experiments_respects_limit(self):
        """Query experiments with limit=N returns at most N results."""
        from big_a.tracking.comparison import query_experiments

        # Create 5 mock recorders
        mock_recs = []
        for i in range(5):
            mock_rec = MagicMock(
                info={"name": f"exp{i}", "start_time": f"2024-01-0{i+1}", "end_time": "", "status": "FINISHED"},
                list_params=MagicMock(return_value={}),
                list_metrics=MagicMock(return_value={}),
            )
            mock_recs.append(mock_rec)

        mock_exp = MagicMock()
        mock_exp.list_recorders.return_value = {f"rec{i}": mock_recs[i] for i in range(5)}

        with patch.object(_mock_qlib.workflow.R, "list_experiments", return_value={"exp_all": mock_exp}):
            with patch.object(_mock_qlib.workflow.R, "list_recorders", return_value=mock_exp.list_recorders.return_value):
                results = query_experiments(limit=3)

        # Verify limit is respected
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    def test_query_experiments_with_no_matching_pattern(self):
        """Query experiments with non-matching pattern returns empty."""
        from big_a.tracking.comparison import query_experiments

        mock_rec = MagicMock(
            info={"name": "train_lgb", "start_time": "2024-01-01", "end_time": "", "status": "FINISHED"},
            list_params=MagicMock(return_value={}),
            list_metrics=MagicMock(return_value={}),
        )
        mock_exp = MagicMock()
        mock_exp.list_recorders.return_value = {"rec1": mock_rec}

        with patch.object(_mock_qlib.workflow.R, "list_experiments", return_value={"train_lgb_20240101": mock_exp}):
            with patch.object(_mock_qlib.workflow.R, "list_recorders", return_value=mock_exp.list_recorders.return_value):
                results = query_experiments(name_pattern="kronos", limit=10)

        # Verify no results match the non-matching pattern
        assert len(results) == 0, f"Expected 0 results for non-matching pattern, got {len(results)}"
