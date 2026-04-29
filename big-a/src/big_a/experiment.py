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
        logger.warning(f"Failed to start experiment '{name}' (non-fatal): {e}")
        return None


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
        logger.warning(f"Failed to log parameters (non-fatal): {e}")


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
        logger.warning(f"Failed to log metrics (non-fatal): {e}")


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
        R.end_exp()
        logger.info(f"Ended experiment with status: {status}")
    except Exception as e:
        logger.error(f"Failed to end experiment: {e}")
        raise RuntimeError(f"Failed to end experiment: {e}") from e


def log_hyperparams_from_config(config: dict, prefix: str = "") -> None:
    """Recursively flatten a nested dict and log as hyperparams.

    Converts nested dict to dot-notation key-value pairs.
    Example: {"model": {"kwargs": {"learning_rate": 0.2}}}
    becomes: {"model.kwargs.learning_rate": 0.2}

    Args:
        config: The configuration dict to flatten and log.
        prefix: Optional prefix to prepend to all keys.
    """

    def _flatten(d: dict, parent_key: str = "") -> list[tuple[str, Any]]:
        """Recursively flatten dict, yielding (key, value) pairs."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key))
            elif v is None or isinstance(v, (str, int, float, bool)):
                items.append((new_key, v))
            # Skip lists and other non-leaf types
        return items

    flattened = dict(_flatten(config, prefix))

    if not flattened:
        logger.debug("No leaf parameters to log from config")
        return

    try:
        recorder = R.get_recorder()
        if recorder is None:
            logger.warning("No active recorder found. Cannot log hyperparams from config.")
            return

        recorder.log_params(**flattened)
        logger.debug(f"Logged {len(flattened)} hyperparams from config")
    except Exception as e:
        logger.error(f"Failed to log hyperparams from config: {e}")
        # Don't crash - just log error


def log_data_version(data_dir: str | Path | None = None) -> None:
    """Log data version info: snapshot date and MD5 checksum of calendars/day.txt.

    Args:
        data_dir: Path to the Qlib data directory. Uses default if None.
    """
    import hashlib

    try:
        from big_a.data.updater import get_last_update_date

        # Resolve data directory - reuse pattern from updater
        if data_dir is None:
            from big_a.config import PROJECT_ROOT
            data_dir = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"
        else:
            data_dir = Path(data_dir)

        day_file = data_dir / "calendars" / "day.txt"
        if not day_file.exists():
            logger.warning(f"Data version: calendars/day.txt not found at {data_dir}")
            return

        # Get the last update date
        snapshot_date = get_last_update_date(str(data_dir))

        # Calculate MD5 checksum of the file
        md5_hash = hashlib.md5(day_file.read_bytes()).hexdigest()

        recorder = R.get_recorder()
        if recorder is None:
            logger.warning("No active recorder found. Cannot log data version.")
            return

        recorder.log_params(data_snapshot_date=snapshot_date, data_checksum=md5_hash)
        logger.debug(f"Logged data version: date={snapshot_date}, checksum={md5_hash}")
    except FileNotFoundError as e:
        logger.warning(f"Data version logging skipped: {e}")
    except Exception as e:
        logger.error(f"Failed to log data version: {e}")
        # Don't crash


def log_model_artifact(model_path: str | Path, artifact_name: str | None = None) -> None:
    """Read a model file and save it as an artifact.

    Args:
        model_path: Path to the model file to log.
        artifact_name: Optional name for the artifact. Defaults to filename.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    artifact_name = artifact_name or model_path.name

    try:
        recorder = R.get_recorder()
        if recorder is None:
            logger.warning("No active recorder found. Cannot log model artifact.")
            return

        file_bytes = model_path.read_bytes()
        recorder.save_objects({artifact_name: file_bytes})
        logger.info(f"Logged model artifact: {model_path} as {artifact_name}")
    except Exception as e:
        logger.error(f"Failed to log model artifact '{model_path}': {e}")
        raise RuntimeError(f"Failed to log model artifact: {e}") from e


def experiment_context(name: str, params: dict | None = None):
    """Context manager for running code within an experiment.

    Usage:
        with experiment_context("my_experiment", {"lr": 0.1}) as recorder:
            # your code here
            recorder.log_metrics({"accuracy": 0.95})

    Args:
        name: Name of the experiment.
        params: Optional parameters to log when entering.

    Yields:
        The recorder object for the current experiment.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        recorder = None
        try:
            recorder = R.start(experiment_name=name)
            logger.info(f"Started experiment context: {name}")
        except Exception as e:
            logger.warning(f"Failed to start experiment context '{name}' (non-fatal): {e}")

        if params and recorder is not None:
            try:
                recorder.log_params(**params)
            except Exception as e:
                logger.warning(f"Failed to log params in experiment context: {e}")

        try:
            yield recorder
        finally:
            try:
                R.end_exp()
                logger.debug(f"Ended experiment context: {name}")
            except Exception as e:
                logger.warning(f"Failed to end experiment context: {e}")

    return _ctx()


def make_experiment_name(script: str, model: str = "") -> str:
    """Create a timestamped experiment name.

    Format:
        - With model: {script}_{model}_{YYYYMMDD_HHMMSS}
        - Without model: {script}_{YYYYMMDD_HHMMSS}

    Args:
        script: Name of the script (must be non-empty string).
        model: Optional model name to include.

    Returns:
        The formatted experiment name.

    Raises:
        ValueError: If script is empty or not a string.
    """
    import datetime

    if not isinstance(script, str) or not script.strip():
        raise ValueError("script must be a non-empty string")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if model:
        return f"{script}_{model}_{timestamp}"
    return f"{script}_{timestamp}"


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
