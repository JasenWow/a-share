from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from qlib.workflow import R


def start_experiment(name: str, params: dict | None = None) -> Any:
    """Start a new MLflow experiment using Qlib's R system.

    Args:
        name: Name of the experiment to start.
        params: Optional dictionary of parameters to log immediately.

    Returns:
        The recorder object for the started experiment.

    Raises:
        RuntimeError: If R.start() fails.
    """
    try:
        recorder = R.start(experiment_name=name)
        logger.info(f"Started experiment: {name}")

        if params:
            log_params(params)

        return recorder
    except Exception as e:
        logger.error(f"Failed to start experiment '{name}': {e}")
        raise RuntimeError(f"Failed to start experiment: {e}") from e


def log_params(params: dict) -> None:
    """Log parameters to the current MLflow run.

    Args:
        params: Dictionary of parameters to log.

    Raises:
        RuntimeError: If no active recorder exists or logging fails.
    """
    try:
        recorder = R.get_recorder()
        if recorder is None:
            logger.warning("No active recorder found. Cannot log params.")
            return

        recorder.log_params(**params)
        logger.debug(f"Logged {len(params)} parameters")
    except Exception as e:
        logger.error(f"Failed to log parameters: {e}")
        raise RuntimeError(f"Failed to log parameters: {e}") from e


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log metrics to the current MLflow run.

    Args:
        metrics: Dictionary of metrics to log.
        step: Optional step number for the metrics.

    Raises:
        RuntimeError: If no active recorder exists or logging fails.
    """
    try:
        recorder = R.get_recorder()
        if recorder is None:
            logger.warning("No active recorder found. Cannot log metrics.")
            return

        if step is not None:
            recorder.log_metrics(**metrics, step=step)
        else:
            recorder.log_metrics(**metrics)

        logger.debug(f"Logged {len(metrics)} metrics" + (f" at step {step}" if step is not None else ""))
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")
        raise RuntimeError(f"Failed to log metrics: {e}") from e


def log_artifact(local_path: str | Path, artifact_path: str | None = None) -> None:
    """Save a local file as an artifact in the current MLflow run.

    Args:
        local_path: Path to the local file to save.
        artifact_path: Optional destination path within the MLflow run.

    Raises:
        RuntimeError: If no active recorder exists or saving fails.
        FileNotFoundError: If the local file does not exist.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        logger.error(f"Artifact file not found: {local_path}")
        raise FileNotFoundError(f"Artifact file not found: {local_path}")

    try:
        recorder = R.get_recorder()
        if recorder is None:
            logger.warning("No active recorder found. Cannot log artifact.")
            return

        artifact_name = artifact_path or local_path.name
        with open(local_path, "rb") as f:
            recorder.save_objects({artifact_name: f.read()})

        logger.info(f"Logged artifact: {local_path} -> {artifact_name}")
    except Exception as e:
        logger.error(f"Failed to log artifact '{local_path}': {e}")
        raise RuntimeError(f"Failed to log artifact: {e}") from e


def log_model_config(config: dict) -> None:
    """Save model configuration as a JSON artifact.

    Args:
        config: Dictionary containing model configuration.

    Raises:
        RuntimeError: If saving the configuration fails.
    """
    import json

    try:
        config_json = json.dumps(config, indent=2, default=str)
        recorder = R.get_recorder()
        if recorder is None:
            logger.warning("No active recorder found. Cannot log model config.")
            return

        recorder.save_objects({"model_config.json": config_json})
        logger.info("Logged model configuration")
    except Exception as e:
        logger.error(f"Failed to log model config: {e}")
        raise RuntimeError(f"Failed to log model config: {e}") from e


def log_backtest_config(config: dict) -> None:
    """Save backtest configuration as a JSON artifact.

    Args:
        config: Dictionary containing backtest configuration.

    Raises:
        RuntimeError: If saving the configuration fails.
    """
    import json

    try:
        config_json = json.dumps(config, indent=2, default=str)
        recorder = R.get_recorder()
        if recorder is None:
            logger.warning("No active recorder found. Cannot log backtest config.")
            return

        recorder.save_objects({"backtest_config.json": config_json})
        logger.info("Logged backtest configuration")
    except Exception as e:
        logger.error(f"Failed to log backtest config: {e}")
        raise RuntimeError(f"Failed to log backtest config: {e}") from e


def end_experiment(status: str = "FINISHED") -> None:
    """End the current MLflow experiment run.

    Args:
        status: Status of the run (e.g., "FINISHED", "FAILED", "KILLED").

    Raises:
        RuntimeError: If ending the recorder fails.
    """
    try:
        R.end_recorder()
        logger.info(f"Ended experiment with status: {status}")
    except Exception as e:
        logger.error(f"Failed to end experiment: {e}")
        raise RuntimeError(f"Failed to end experiment: {e}") from e


def get_experiment_summary(experiment_name: str) -> pd.DataFrame:
    """Query all runs for an experiment and return a summary table.

    Args:
        experiment_name: Name of the experiment to query.

    Returns:
        DataFrame containing run information and metrics.

    Raises:
        RuntimeError: If querying experiment data fails.
    """
    try:
        recorders = R.list_recorders(experiment_name=experiment_name)

        if not recorders:
            logger.warning(f"No recorders found for experiment: {experiment_name}")
            return pd.DataFrame()

        summary_data = []
        for recorder_id, recorder in recorders.items():
            try:
                info = recorder.info
                run_data = {
                    "run_id": recorder_id,
                    "name": info.get("name", ""),
                    "start_time": info.get("start_time", ""),
                    "end_time": info.get("end_time", ""),
                    "status": info.get("status", ""),
                }

                params = recorder.list_params()
                for key, value in params.items():
                    run_data[f"param_{key}"] = value

                metrics = recorder.list_metrics()
                for key, value in metrics.items():
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        run_data[f"metric_{key}"] = value[-1]
                    else:
                        run_data[f"metric_{key}"] = value

                summary_data.append(run_data)
            except Exception as e:
                logger.warning(f"Failed to extract data from recorder {recorder_id}: {e}")
                continue

        df = pd.DataFrame(summary_data)
        logger.info(f"Retrieved summary for {len(df)} runs in experiment '{experiment_name}'")
        return df
    except Exception as e:
        logger.error(f"Failed to get experiment summary for '{experiment_name}': {e}")
        raise RuntimeError(f"Failed to get experiment summary: {e}") from e
