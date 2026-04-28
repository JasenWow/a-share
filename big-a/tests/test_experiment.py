"""Comprehensive tests for new experiment module functions."""
from __future__ import annotations

import sys
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Mock qlib before importing experiment module
_mock_qlib = MagicMock()
_mock_qlib.workflow = MagicMock()
_mock_qlib.workflow.R = MagicMock()
sys.modules["qlib"] = _mock_qlib
sys.modules["qlib.workflow"] = _mock_qlib.workflow


class TestLogHyperparamsFromConfig:
    """Tests for log_hyperparams_from_config function."""

    def test_log_hyperparams_flat_dict(self):
        """Flat dict logs all key-values."""
        from big_a.experiment import log_hyperparams_from_config

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=mock_recorder):
            log_hyperparams_from_config({"lr": 0.1, "epochs": 10})
            mock_recorder.log_params.assert_called_once_with(lr=0.1, epochs=10)

    def test_log_hyperparams_nested_dict(self):
        """Nested dict flattens with dots."""
        from big_a.experiment import log_hyperparams_from_config

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=mock_recorder):
            config = {"model": {"kwargs": {"learning_rate": 0.2, "num_leaves": 210}}}
            log_hyperparams_from_config(config)
            mock_recorder.log_params.assert_called_once_with(
                **{
                    "model.kwargs.learning_rate": 0.2,
                    "model.kwargs.num_leaves": 210,
                }
            )

    def test_log_hyperparams_with_prefix(self):
        """Prefix prepended to all keys."""
        from big_a.experiment import log_hyperparams_from_config

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=mock_recorder):
            config = {"learning_rate": 0.1}
            log_hyperparams_from_config(config, prefix="lgbm")
            mock_recorder.log_params.assert_called_once_with(**{"lgbm.learning_rate": 0.1})

    def test_log_hyperparams_empty_dict(self):
        """Empty dict, no crash, no calls."""
        from big_a.experiment import log_hyperparams_from_config

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=mock_recorder):
            log_hyperparams_from_config({})
            mock_recorder.log_params.assert_not_called()

    def test_log_hyperparams_no_recorder(self):
        """No active recorder - logs warning, no crash."""
        from big_a.experiment import log_hyperparams_from_config

        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=None):
            with patch("big_a.experiment.logger") as mock_logger:
                log_hyperparams_from_config({"lr": 0.1})
                mock_logger.warning.assert_called_once()


class TestLogDataVersion:
    """Tests for log_data_version function."""

    def test_log_data_version_success(self, tmp_path):
        """Mock file, verify date + checksum logged."""
        from big_a.experiment import log_data_version

        calendars_dir = tmp_path / "calendars"
        calendars_dir.mkdir(parents=True)
        day_file = calendars_dir / "day.txt"
        day_file.write_text("2024-01-15\n2024-01-16\n2024-01-17\n")

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=mock_recorder):
            with patch("big_a.data.updater.get_last_update_date") as mock_get_date:
                mock_get_date.return_value = "2024-01-17"
                log_data_version(str(tmp_path))
                mock_get_date.assert_called_once()
                call_kwargs = mock_recorder.log_params.call_args.kwargs
                assert call_kwargs["data_snapshot_date"] == "2024-01-17"
                assert "data_checksum" in call_kwargs
                assert len(call_kwargs["data_checksum"]) == 32  # MD5 hex length

    def test_log_data_version_missing_dir(self):
        """Nonexistent dir, warning logged, no crash."""
        from big_a.experiment import log_data_version

        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=MagicMock()):
            with patch("big_a.experiment.logger") as mock_logger:
                log_data_version("/nonexistent/path")
                mock_logger.warning.assert_called()


class TestLogModelArtifact:
    """Tests for log_model_artifact function."""

    def test_log_model_artifact_success(self, tmp_path):
        """Temp file, verify save_objects called."""
        from big_a.experiment import log_model_artifact

        model_file = tmp_path / "model.txt"
        model_file.write_bytes(b"model content here")

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=mock_recorder):
            log_model_artifact(model_file)
            mock_recorder.save_objects.assert_called_once()
            saved_data = mock_recorder.save_objects.call_args[0][0]
            assert "model.txt" in saved_data
            assert saved_data["model.txt"] == b"model content here"

    def test_log_model_artifact_custom_name(self, tmp_path):
        """Custom artifact name used."""
        from big_a.experiment import log_model_artifact

        model_file = tmp_path / "model.txt"
        model_file.write_bytes(b"model content")

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=mock_recorder):
            log_model_artifact(model_file, artifact_name="my_model.bin")
            mock_recorder.save_objects.assert_called_once()
            saved_data = mock_recorder.save_objects.call_args[0][0]
            assert "my_model.bin" in saved_data

    def test_log_model_artifact_missing_file(self):
        """Nonexistent file raises FileNotFoundError."""
        from big_a.experiment import log_model_artifact

        with pytest.raises(FileNotFoundError):
            log_model_artifact("/nonexistent/model.bin")

    def test_log_model_artifact_no_recorder(self, tmp_path):
        """No recorder - logs warning, doesn't crash."""
        from big_a.experiment import log_model_artifact

        model_file = tmp_path / "model.txt"
        model_file.write_bytes(b"content")

        with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=None):
            with patch("big_a.experiment.logger") as mock_logger:
                log_model_artifact(model_file)
                mock_logger.warning.assert_called()


class TestExperimentContext:
    """Tests for experiment_context function."""

    def test_experiment_context_normal_flow(self):
        """Start called, end called, params logged."""
        from big_a.experiment import experiment_context

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "start", return_value=mock_recorder):
            with patch.object(_mock_qlib.workflow.R, "end_exp"):
                with experiment_context("test_exp", {"lr": 0.1}) as recorder:
                    assert recorder is mock_recorder
                _mock_qlib.workflow.R.start.assert_called_once_with(experiment_name="test_exp")
                _mock_qlib.workflow.R.end_exp.assert_called_once()
                mock_recorder.log_params.assert_called_once_with(lr=0.1)

    def test_experiment_context_no_params(self):
        """Without params, just start and end."""
        from big_a.experiment import experiment_context

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "start", return_value=mock_recorder):
            with patch.object(_mock_qlib.workflow.R, "end_exp"):
                with experiment_context("test_exp") as recorder:
                    assert recorder is mock_recorder
                _mock_qlib.workflow.R.end_exp.assert_called_once()
                mock_recorder.log_params.assert_not_called()

    def test_experiment_context_exception_flow(self):
        """Raise inside with, end STILL called."""
        from big_a.experiment import experiment_context

        mock_recorder = MagicMock()
        with patch.object(_mock_qlib.workflow.R, "start", return_value=mock_recorder):
            with patch.object(_mock_qlib.workflow.R, "end_exp"):
                with pytest.raises(ValueError):
                    with experiment_context("test_exp"):
                        raise ValueError("test error")
                _mock_qlib.workflow.R.end_exp.assert_called_once()

    def test_experiment_context_start_failure(self):
        """Start fails - exception propagates, end still called per spec."""
        from big_a.experiment import experiment_context

        with patch.object(_mock_qlib.workflow.R, "start", side_effect=RuntimeError("start failed")):
            with patch.object(_mock_qlib.workflow.R, "end_exp") as mock_end:
                with pytest.raises(RuntimeError, match="start failed"):
                    with experiment_context("test_exp"):
                        pass
                mock_end.assert_called_once()


class TestMakeExperimentName:
    """Tests for make_experiment_name function."""

    def test_make_experiment_name_with_model(self):
        """Matches pattern script_model_YYYYMMDD_HHMMSS."""
        from big_a.experiment import make_experiment_name
        import re

        result = make_experiment_name("train", "lightgbm")
        pattern = r"^train_lightgbm_\d{8}_\d{6}$"
        assert re.match(pattern, result), f"Result {result} doesn't match pattern"

    def test_make_experiment_name_without_model(self):
        """Matches pattern script_YYYYMMDD_HHMMSS."""
        from big_a.experiment import make_experiment_name
        import re

        result = make_experiment_name("train")
        pattern = r"^train_\d{8}_\d{6}$"
        assert re.match(pattern, result), f"Result {result} doesn't match pattern"

    def test_make_experiment_name_empty_script(self):
        """Empty script raises ValueError."""
        from big_a.experiment import make_experiment_name

        with pytest.raises(ValueError, match="non-empty string"):
            make_experiment_name("")

    def test_make_experiment_name_whitespace_script(self):
        """Whitespace-only script raises ValueError."""
        from big_a.experiment import make_experiment_name

        with pytest.raises(ValueError, match="non-empty string"):
            make_experiment_name("   ")

    def test_make_experiment_name_non_string(self):
        """Non-string script raises ValueError."""
        from big_a.experiment import make_experiment_name

        with pytest.raises(ValueError, match="non-empty string"):
            make_experiment_name(123)  # type: ignore


class TestNoDirectMlflowImport:
    """Verify experiment.py source doesn't contain direct mlflow import."""

    def test_no_direct_mlflow_import(self):
        """Verify experiment.py source doesn't contain 'import mlflow'."""
        import big_a.experiment as exp_module
        source = inspect.getsource(exp_module)
        assert "import mlflow" not in source


class TestImportNewFunctions:
    """Verify all new functions can be imported."""

    def test_import_log_hyperparams_from_config(self):
        from big_a.experiment import log_hyperparams_from_config
        assert callable(log_hyperparams_from_config)

    def test_import_log_data_version(self):
        from big_a.experiment import log_data_version
        assert callable(log_data_version)

    def test_import_log_model_artifact(self):
        from big_a.experiment import log_model_artifact
        assert callable(log_model_artifact)

    def test_import_experiment_context(self):
        from big_a.experiment import experiment_context
        assert callable(experiment_context)

    def test_import_make_experiment_name(self):
        from big_a.experiment import make_experiment_name
        assert callable(make_experiment_name)


class TestComparisonFunctions:
    """Tests for comparison module functions."""

    def test_query_experiments_with_filter(self):
        """Verify name filtering works."""
        from big_a.tracking.comparison import query_experiments
        from unittest.mock import MagicMock, patch

        mock_rec1 = MagicMock(
            info={"name": "run1", "start_time": "2024-01-01", "end_time": "", "status": "FINISHED"},
            list_params=MagicMock(return_value={"lr": 0.1}),
            list_metrics=MagicMock(return_value={"accuracy": 0.9}),
        )
        mock_rec2 = MagicMock(
            info={"name": "run2", "start_time": "2024-01-02", "end_time": "", "status": "FINISHED"},
            list_params=MagicMock(return_value={"lr": 0.2}),
            list_metrics=MagicMock(return_value={"accuracy": 0.95}),
        )

        mock_exp1 = MagicMock()
        mock_exp1.list_recorders.return_value = {"rec1": mock_rec1}
        mock_exp2 = MagicMock()
        mock_exp2.list_recorders.return_value = {"rec2": mock_rec2}

        def list_recorders_side_effect(experiment_name):
            if "lightgbm" in experiment_name:
                return {"rec1": mock_rec1}
            return {"rec2": mock_rec2}

        with patch.object(_mock_qlib.workflow.R, "list_experiments", return_value={
            "train_lightgbm_20240101": mock_exp1,
            "train_kronos_20240102": mock_exp2,
        }):
            with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
                results = query_experiments(name_pattern="lightgbm", limit=10)

        assert len(results) == 1
        assert results[0]["experiment_name"] == "train_lightgbm_20240101"
        assert results[0]["params"]["lr"] == 0.1
        assert results[0]["metrics"]["accuracy"] == 0.9

    def test_query_experiments_limit(self):
        """Verify limit is applied to total results."""
        from big_a.tracking.comparison import query_experiments
        from unittest.mock import MagicMock, patch

        mock_recorder = MagicMock(
            info={"name": "run", "start_time": "2024-01-01", "end_time": "", "status": "FINISHED"},
            list_params=MagicMock(return_value={}),
            list_metrics=MagicMock(return_value={}),
        )
        mock_exp = MagicMock()
        all_recorders = {f"rec{i}": mock_recorder for i in range(5)}
        mock_exp.list_recorders.return_value = all_recorders

        with patch("big_a.tracking.comparison.R") as mock_R:
            mock_R.list_experiments.return_value = {"train_exp": mock_exp}
            mock_R.list_recorders.return_value = all_recorders
            results = query_experiments(limit=3)

        assert len(results) == 3

    def test_compare_by_params(self):
        """Mock 2 experiments with different params, verify DataFrame."""
        from big_a.tracking.comparison import compare_by_params
        from unittest.mock import MagicMock, patch

        mock_rec1 = MagicMock(
            info={},
            list_params=MagicMock(return_value={"lr": 0.1, "epochs": 10}),
        )
        mock_rec2 = MagicMock(
            info={},
            list_params=MagicMock(return_value={"lr": 0.2, "batch_size": 32}),
        )

        mock_exp1 = MagicMock()
        mock_exp1.list_recorders.return_value = {"rec1": mock_rec1}
        mock_exp2 = MagicMock()
        mock_exp2.list_recorders.return_value = {"rec2": mock_rec2}

        def list_recorders_side_effect(experiment_name):
            if experiment_name == "exp1":
                return {"rec1": mock_rec1}
            return {"rec2": mock_rec2}

        with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
            df = compare_by_params(["exp1", "exp2"])

        assert "run_id" in df.columns
        assert "experiment" in df.columns
        assert "lr" in df.columns
        assert "epochs" in df.columns
        assert "batch_size" in df.columns

    def test_compare_by_metrics(self):
        """Mock 2 experiments with different metrics, verify DataFrame."""
        from big_a.tracking.comparison import compare_by_metrics
        from unittest.mock import MagicMock, patch

        mock_rec1 = MagicMock(
            info={},
            list_metrics=MagicMock(return_value={"ic": 0.05, "sharpe": 1.5}),
        )
        mock_rec2 = MagicMock(
            info={},
            list_metrics=MagicMock(return_value={"ic": 0.08, "return": 0.12}),
        )

        def list_recorders_side_effect(experiment_name):
            if experiment_name == "exp1":
                return {"rec1": mock_rec1}
            return {"rec2": mock_rec2}

        with patch.object(_mock_qlib.workflow.R, "list_recorders", side_effect=list_recorders_side_effect):
            df = compare_by_metrics(["exp1", "exp2"])

        assert "run_id" in df.columns
        assert "experiment" in df.columns
        assert "ic" in df.columns
        assert "sharpe" in df.columns
        assert "return" in df.columns

    def test_get_rolling_history(self):
        """Verify per-window metrics extracted correctly."""
        from big_a.tracking.comparison import get_rolling_history
        from unittest.mock import MagicMock, patch

        mock_rec = MagicMock(
            list_metrics=MagicMock(return_value={
                "ic": [0.05, 0.06, 0.07],
                "sharpe": [1.5, 1.6, 1.7],
            }),
        )
        mock_exp = MagicMock()
        mock_exp.list_recorders.return_value = {"rec1": mock_rec}

        with patch.object(_mock_qlib.workflow.R, "list_recorders", return_value=mock_exp.list_recorders.return_value):
            df = get_rolling_history("rolling_exp")

        assert not df.empty
        assert "window" in df.columns
        assert "metric_name" in df.columns
        assert "value" in df.columns
        assert len(df) == 6


class TestRollingTracking:
    """Tests for rolling backtest tracking functionality."""

    def test_log_hyperparams_from_config_called_in_run_rolling(self):
        """Verify log_hyperparams_from_config is called with prefix='rolling'."""
        from big_a.backtest.rolling import RollingBacktester
        from big_a.experiment import log_hyperparams_from_config

        with patch.object(_mock_qlib.workflow.R, "start"):
            with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=MagicMock()):
                with patch.object(_mock_qlib.workflow.R, "end_exp"):
                    with patch("big_a.backtest.rolling.log_hyperparams_from_config") as mock_log_hp:
                        with patch("big_a.backtest.rolling.generate_windows") as mock_gen_windows:
                            with patch.object(_mock_qlib.workflow.R, "list_experiments", return_value={}):
                                with patch.object(_mock_qlib.workflow.R, "list_recorders", return_value={}):
                                    mock_gen_windows.return_value = [
                                        {
                                            "window_idx": 0,
                                            "train_start": "2010-01-01",
                                            "train_end": "2014-12-31",
                                            "valid_start": "2015-01-01",
                                            "valid_end": "2015-12-31",
                                            "test_start": "2016-01-01",
                                            "test_end": "2016-12-31",
                                        },
                                    ]

                                    tester = RollingBacktester(
                                        model_type="kronos",
                                        start_year=2010,
                                        end_year=2016,
                                    )
                                    with patch.object(tester, "_run_window") as mock_run:
                                        mock_run.return_value = MagicMock(
                                            window_idx=0,
                                            ic=0.05,
                                            rank_ic=0.04,
                                            sharpe=1.0,
                                            max_drawdown=0.1,
                                            report=None,
                                            signal=MagicMock(),
                                        )
                                        tester.run_rolling(config={"lr": 0.1})

                                    assert mock_log_hp.call_count >= 1
                                    calls = mock_log_hp.call_args_list
                                    prefixes = [c.kwargs.get("prefix") or (c.args[1] if len(c.args) > 1 else None) for c in calls]
                                    assert "rolling" in prefixes

    def test_per_window_metrics_logged_with_step(self):
        """Verify per-window metrics are logged with step parameter set to window_idx."""
        from big_a.backtest.rolling import RollingBacktester

        with patch.object(_mock_qlib.workflow.R, "start"):
            mock_recorder = MagicMock()
            with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=mock_recorder):
                with patch.object(_mock_qlib.workflow.R, "end_exp"):
                    with patch("big_a.backtest.rolling.generate_windows") as mock_gen_windows:
                        with patch.object(_mock_qlib.workflow.R, "list_experiments", return_value={}):
                            with patch.object(_mock_qlib.workflow.R, "list_recorders", return_value={}):
                                mock_gen_windows.return_value = [
                                    {
                                        "window_idx": 0,
                                        "train_start": "2010-01-01",
                                        "train_end": "2014-12-31",
                                        "valid_start": "2015-01-01",
                                        "valid_end": "2015-12-31",
                                        "test_start": "2016-01-01",
                                        "test_end": "2016-12-31",
                                    },
                                    {
                                        "window_idx": 1,
                                        "train_start": "2011-01-01",
                                        "train_end": "2015-12-31",
                                        "valid_start": "2016-01-01",
                                        "valid_end": "2016-12-31",
                                        "test_start": "2017-01-01",
                                        "test_end": "2017-12-31",
                                    },
                                ]

                                tester = RollingBacktester(
                                    model_type="kronos",
                                    start_year=2010,
                                    end_year=2017,
                                    step_years=1,
                                )
                                with patch.object(tester, "_run_window") as mock_run:
                                    mock_run.side_effect = [
                                        MagicMock(window_idx=0, ic=0.05, rank_ic=0.04, sharpe=1.0, max_drawdown=0.1, report=None, signal=MagicMock()),
                                        MagicMock(window_idx=1, ic=0.06, rank_ic=0.05, sharpe=1.1, max_drawdown=0.15, report=None, signal=MagicMock()),
                                    ]
                                    tester.run_rolling()

                                assert mock_recorder.log_metrics.called
                                calls = mock_recorder.log_metrics.call_args_list
                                steps = []
                                for call in calls:
                                    if call.kwargs.get("step") is not None:
                                        steps.append(call.kwargs["step"])
                                    elif len(call.args) > 0:
                                        steps.append(call.args[-1] if len(call.args) > 1 else None)
                                assert 0 in steps, f"Expected step 0 in {steps}"
                                assert 1 in steps, f"Expected step 1 in {steps}"

    def test_summary_artifact_saved(self):
        """Verify summary parquet is saved as artifact after aggregation."""
        from big_a.backtest.rolling import RollingBacktester

        artifact_calls = []

        def capture_log_artifact(path, name=None):
            artifact_calls.append({"path": path, "name": name})

        with patch.object(_mock_qlib.workflow.R, "start"):
            with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=MagicMock()):
                with patch.object(_mock_qlib.workflow.R, "end_exp"):
                    with patch("big_a.backtest.rolling.log_model_artifact", side_effect=capture_log_artifact):
                        with patch("big_a.backtest.rolling.generate_windows") as mock_gen_windows:
                            with patch.object(_mock_qlib.workflow.R, "list_experiments", return_value={}):
                                with patch.object(_mock_qlib.workflow.R, "list_recorders", return_value={}):
                                    mock_gen_windows.return_value = [
                                        {
                                            "window_idx": 0,
                                            "train_start": "2010-01-01",
                                            "train_end": "2014-12-31",
                                            "valid_start": "2015-01-01",
                                            "valid_end": "2015-12-31",
                                            "test_start": "2016-01-01",
                                            "test_end": "2016-12-31",
                                        },
                                    ]

                                    tester = RollingBacktester(
                                        model_type="kronos",
                                        start_year=2010,
                                        end_year=2016,
                                    )
                                    with patch.object(tester, "_run_window") as mock_run:
                                        mock_run.return_value = MagicMock(
                                            window_idx=0,
                                            train_start="2010-01-01",
                                            train_end="2014-12-31",
                                            valid_start="2015-01-01",
                                            valid_end="2015-12-31",
                                            test_start="2016-01-01",
                                            test_end="2016-12-31",
                                            ic=0.05,
                                            rank_ic=0.04,
                                            sharpe=1.0,
                                            max_drawdown=0.1,
                                            report=None,
                                            signal=MagicMock(),
                                        )
                                        with patch("pandas.DataFrame.to_parquet"):
                                            tester.run_rolling()

                                    artifact_names = [a["name"] for a in artifact_calls]
                                    assert "rolling_summary.parquet" in artifact_names

    def test_log_hyperparams_from_config_per_window(self):
        """Verify log_hyperparams_from_config is called with window-specific prefix during lightgbm window."""
        from big_a.backtest.rolling import RollingBacktester
        from big_a.experiment import log_hyperparams_from_config

        logged_hyperparams = []

        def capture_log_hyperparams(config, prefix=""):
            logged_hyperparams.append({"config": config, "prefix": prefix})

        with patch.object(_mock_qlib.workflow.R, "start"):
            with patch.object(_mock_qlib.workflow.R, "get_recorder", return_value=MagicMock()):
                with patch.object(_mock_qlib.workflow.R, "end_exp"):
                    with patch("big_a.backtest.rolling.log_hyperparams_from_config", side_effect=capture_log_hyperparams):
                        with patch("big_a.backtest.rolling.generate_windows") as mock_gen_windows:
                            with patch.object(_mock_qlib.workflow.R, "list_experiments", return_value={}):
                                with patch.object(_mock_qlib.workflow.R, "list_recorders", return_value={}):
                                    mock_gen_windows.return_value = [
                                        {
                                            "window_idx": 0,
                                            "train_start": "2010-01-01",
                                            "train_end": "2014-12-31",
                                            "valid_start": "2015-01-01",
                                            "valid_end": "2015-12-31",
                                            "test_start": "2016-01-01",
                                            "test_end": "2016-12-31",
                                        },
                                        {
                                            "window_idx": 1,
                                            "train_start": "2011-01-01",
                                            "train_end": "2015-12-31",
                                            "valid_start": "2016-01-01",
                                            "valid_end": "2016-12-31",
                                            "test_start": "2017-01-01",
                                            "test_end": "2017-12-31",
                                        },
                                    ]

                                    tester = RollingBacktester(
                                        model_type="kronos",
                                        start_year=2010,
                                        end_year=2017,
                                        step_years=1,
                                    )
                                    with patch.object(tester, "_run_window") as mock_run:
                                        mock_run.side_effect = [
                                            MagicMock(window_idx=0, ic=0.05, rank_ic=0.04, sharpe=1.0, max_drawdown=0.1, report=None, signal=MagicMock()),
                                            MagicMock(window_idx=1, ic=0.06, rank_ic=0.05, sharpe=1.1, max_drawdown=0.15, report=None, signal=MagicMock()),
                                        ]
                                        tester.run_rolling()

                                    prefixes = [h["prefix"] for h in logged_hyperparams]
                                    assert "rolling" in prefixes
