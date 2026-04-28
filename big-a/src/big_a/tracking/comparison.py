"""Experiment comparison and query utilities."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from loguru import logger
from qlib.workflow import R


def query_experiments(name_pattern: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Query MLflow experiments, optionally filtering by name pattern.

    Returns list of dicts with keys: experiment_name, run_id, params, metrics, start_time.
    """
    try:
        experiments = R.list_experiments()
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        return []

    results: list[dict[str, Any]] = []

    for exp_name, experiment in experiments.items():
        if name_pattern and not re.search(name_pattern, exp_name):
            continue

        try:
            recorders = R.list_recorders(experiment_name=exp_name)
        except Exception as e:
            logger.warning(f"Failed to list recorders for experiment {exp_name}: {e}")
            continue

        for recorder_id, recorder in recorders.items():
            try:
                info = recorder.info
                params = recorder.list_params()
                metrics = recorder.list_metrics()

                for key, value in metrics.items():
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        metrics[key] = value[-1]

                results.append(
                    {
                        "experiment_name": exp_name,
                        "run_id": recorder_id,
                        "name": info.get("name", ""),
                        "start_time": info.get("start_time", ""),
                        "end_time": info.get("end_time", ""),
                        "status": info.get("status", ""),
                        "params": params,
                        "metrics": metrics,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract data from recorder {recorder_id}: {e}")
                continue

    results.sort(key=lambda x: x.get("start_time", ""), reverse=True)
    return results[:limit]


def compare_by_params(experiment_names: list[str]) -> pd.DataFrame:
    """Build a comparison DataFrame of params across experiments.

    Columns: run_id + all param keys found across experiments.
    Rows: one per experiment run.
    """
    rows = []
    all_param_keys: set[str] = set()

    for exp_name in experiment_names:
        try:
            recorders = R.list_recorders(experiment_name=exp_name)
        except Exception as e:
            logger.warning(f"Failed to list recorders for experiment {exp_name}: {e}")
            continue

        for recorder_id, recorder in recorders.items():
            try:
                params = recorder.list_params()
                info = recorder.info
                row: dict[str, Any] = {"run_id": recorder_id, "experiment": exp_name}
                row.update(params)
                rows.append(row)
                all_param_keys.update(params.keys())
            except Exception as e:
                logger.warning(f"Failed to extract params from recorder {recorder_id}: {e}")
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols = ["run_id", "experiment"] + sorted(all_param_keys)
    existing_cols = [c for c in cols if c in df.columns]
    return pd.DataFrame(df[existing_cols])


def compare_by_metrics(experiment_names: list[str]) -> pd.DataFrame:
    """Build a comparison DataFrame of metrics across experiments.

    Columns: run_id + all metric keys.
    Rows: one per experiment run.
    """
    rows = []
    all_metric_keys: set[str] = set()

    for exp_name in experiment_names:
        try:
            recorders = R.list_recorders(experiment_name=exp_name)
        except Exception as e:
            logger.warning(f"Failed to list recorders for experiment {exp_name}: {e}")
            continue

        for recorder_id, recorder in recorders.items():
            try:
                metrics = recorder.list_metrics()
                info = recorder.info

                for key, value in metrics.items():
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        metrics[key] = value[-1]

                row: dict[str, Any] = {"run_id": recorder_id, "experiment": exp_name}
                row.update(metrics)
                rows.append(row)
                all_metric_keys.update(metrics.keys())
            except Exception as e:
                logger.warning(f"Failed to extract metrics from recorder {recorder_id}: {e}")
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols = ["run_id", "experiment"] + sorted(all_metric_keys)
    existing_cols = [c for c in cols if c in df.columns]
    return pd.DataFrame(df[existing_cols])


def get_rolling_history(experiment_name: str) -> pd.DataFrame:
    """Extract per-window metrics from a rolling backtest experiment.

    Returns DataFrame with columns: window, metric_name, value.
    """
    try:
        recorders = R.list_recorders(experiment_name=experiment_name)
    except Exception as e:
        logger.error(f"Failed to list recorders for experiment {experiment_name}: {e}")
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    for recorder_id, recorder in recorders.items():
        try:
            metrics = recorder.list_metrics()

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (list, tuple)):
                    for step, value in enumerate(metric_value):
                        records.append(
                            {
                                "window": step,
                                "metric_name": metric_name,
                                "value": value,
                            }
                        )
                else:
                    records.append(
                        {
                            "window": 0,
                            "metric_name": metric_name,
                            "value": metric_value,
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to extract rolling history from recorder {recorder_id}: {e}")
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["window", "metric_name"])
    return df.reset_index(drop=True)
