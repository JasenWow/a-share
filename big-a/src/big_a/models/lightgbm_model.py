"""LightGBM model wrapper using Qlib's built-in LGBModel."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from big_a.config import PROJECT_ROOT, load_config
from big_a.qlib_config import init_qlib


def _build_full_config(
    model_config_path: str = "configs/model/lightgbm.yaml",
    data_config_path: str = "configs/data/handler_alpha158.yaml",
) -> dict[str, Any]:
    """Load and merge model + dataset configs."""
    return load_config(model_config_path, data_config_path)


def create_model(config: dict[str, Any] | None = None) -> Any:
    """Create a LGBModel instance from config.

    Parameters
    ----------
    config : dict or None
        Full config dict with 'model' key. If None, loads defaults.

    Returns
    -------
    LGBModel
    """
    from qlib.utils import init_instance_by_config

    if config is None:
        config = _build_full_config()

    model_config = config["model"]
    model = init_instance_by_config(model_config)
    logger.info(f"Created model: {model_config['class']}")
    return model


def create_dataset(config: dict[str, Any] | None = None) -> Any:
    """Create a DatasetH from config.

    Parameters
    ----------
    config : dict or None
        Full config dict with 'dataset' key. If None, loads defaults.

    Returns
    -------
    DatasetH
    """
    from qlib.utils import init_instance_by_config

    if config is None:
        config = _build_full_config()

    dataset_config = config["dataset"]
    dataset = init_instance_by_config(dataset_config)
    logger.info("Dataset created")
    return dataset


def train(
    model_config_path: str = "configs/model/lightgbm.yaml",
    data_config_path: str = "configs/data/handler_alpha158.yaml",
) -> tuple[Any, Any, dict[str, Any]]:
    """Train a LightGBM model on Alpha158 features.

    Parameters
    ----------
    model_config_path : str
        Path to model YAML config (relative to project root).
    data_config_path : str
        Path to dataset YAML config (relative to project root).

    Returns
    -------
    tuple of (model, dataset, config)
        The trained model, the dataset used, and the full merged config.
    """
    init_qlib()

    config = _build_full_config(model_config_path, data_config_path)
    dataset = create_dataset(config)
    model = create_model(config)

    logger.info("Training LightGBM model...")
    model.fit(dataset)
    logger.info("Training complete")

    return model, dataset, config


def predict(
    model: Any,
    dataset: Any,
    segment: str = "test",
) -> pd.Series:
    """Generate predictions from a trained model.

    Parameters
    ----------
    model : LGBModel
        A fitted model.
    dataset : DatasetH
        The dataset to predict on.
    segment : str
        Dataset segment: 'train', 'valid', or 'test'.

    Returns
    -------
    pd.Series
        Predictions with MultiIndex (datetime, instrument), name='score'.
    """
    preds = model.predict(dataset, segment=segment)
    preds.name = "score"
    logger.info(f"Predictions generated for segment '{segment}': {len(preds)} rows")
    return preds


def predict_to_dataframe(
    model: Any,
    dataset: Any,
    segment: str = "test",
) -> pd.DataFrame:
    """Generate predictions as a DataFrame with 'score' column.

    Compatible with Qlib backtest signal format.

    Parameters
    ----------
    model : LGBModel
        A fitted model.
    dataset : DatasetH
        The dataset to predict on.
    segment : str
        Dataset segment.

    Returns
    -------
    pd.DataFrame
        Index=(datetime, instrument), columns=['score'].
    """
    preds = predict(model, dataset, segment=segment)
    return preds.to_frame("score")


def save_model(model: Any, path: str | Path) -> None:
    """Save a trained model to disk.

    Parameters
    ----------
    model : LGBModel
        A fitted model.
    path : str or Path
        File path to save to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path: str | Path) -> Any:
    """Load a saved model from disk.

    Parameters
    ----------
    path : str or Path
        File path to load from.

    Returns
    -------
    LGBModel
    """
    path = Path(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model
