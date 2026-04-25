"""Alpha158 feature handler for creating Qlib DatasetH objects."""
from __future__ import annotations

import pandas as pd
from loguru import logger
from qlib.data.dataset import DatasetH

from big_a.config import load_config
from big_a.qlib_config import init_qlib


def create_alpha158_dataset(config_path: str | None = None) -> DatasetH:
    """Create a Qlib DatasetH with Alpha158 handler.

    Parameters
    ----------
    config_path : str or None
        Path to a YAML config (relative to project root).
        Defaults to ``configs/data/handler_alpha158.yaml``.

    Returns
    -------
    DatasetH
        The initialized Qlib dataset.
    """
    from qlib.utils import init_instance_by_config

    init_qlib()

    path = config_path or "configs/data/handler_alpha158.yaml"
    config = load_config(path)
    dataset = init_instance_by_config(config["dataset"])
    logger.info("Alpha158 dataset created successfully")
    return dataset


def get_segment_data(
    dataset: DatasetH,
    segment: str = "train",
    col_set: str = "feature",
) -> pd.DataFrame:
    """Retrieve a data segment from a DatasetH object.

    Parameters
    ----------
    dataset : DatasetH
        The Qlib dataset.
    segment : str
        One of ``'train'``, ``'valid'``, ``'test'``.
    col_set : str
        Column set to return, e.g. ``'feature'`` or ``'label'``.

    Returns
    -------
    pd.DataFrame
    """
    df = dataset.prepare(segment, col_set=col_set, data_key=dataset.DK_L)
    return df


def get_train_data(dataset: DatasetH, col_set: str = "feature") -> pd.DataFrame:
    """Get training features (or labels) from the dataset."""
    return get_segment_data(dataset, segment="train", col_set=col_set)


def get_valid_data(dataset: DatasetH, col_set: str = "feature") -> pd.DataFrame:
    """Get validation features (or labels) from the dataset."""
    return get_segment_data(dataset, segment="valid", col_set=col_set)


def get_test_data(dataset: DatasetH, col_set: str = "feature") -> pd.DataFrame:
    """Get test features (or labels) from the dataset."""
    return get_segment_data(dataset, segment="test", col_set=col_set)
