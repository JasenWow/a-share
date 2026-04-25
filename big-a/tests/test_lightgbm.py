"""Tests for LightGBM model wrapper."""
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from big_a.models.lightgbm_model import (
    create_model,
    predict,
    predict_to_dataframe,
    save_model,
    load_model,
    _build_full_config,
)


@pytest.fixture
def mock_config():
    """Config with model and dataset sections."""
    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"loss": "mse", "num_leaves": 31, "num_threads": 1},
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler"},
                "segments": {
                    "train": ["2010-01-01", "2018-12-31"],
                    "valid": ["2019-01-01", "2021-12-31"],
                    "test": ["2022-01-01", "2024-12-31"],
                },
            },
        },
    }


@pytest.fixture
def mock_dataset():
    """Fake dataset with predictable index for predictions."""
    dates = pd.date_range("2022-01-04", periods=10, freq="B")
    instruments = [f"SH60000{i}" for i in range(5)]
    multi_index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    features = pd.DataFrame(
        np.random.randn(len(multi_index), 10),
        index=multi_index,
    )
    ds = MagicMock()
    ds.prepare.return_value = features
    ds.segments = {"train": ..., "valid": ..., "test": ...}
    return ds


@pytest.fixture
def mock_model(mock_dataset):
    """Fake fitted model that returns predictions."""
    dates = pd.date_range("2022-01-04", periods=10, freq="B")
    instruments = [f"SH60000{i}" for i in range(5)]
    multi_index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    preds = pd.Series(np.random.randn(len(multi_index)), index=multi_index)

    model = MagicMock()
    model.predict.return_value = preds
    return model


class TestBuildFullConfig:
    def test_merges_model_and_data_configs(self):
        config = _build_full_config()
        assert "model" in config
        assert "dataset" in config
        assert config["model"]["class"] == "LGBModel"
        assert config["dataset"]["class"] == "DatasetH"

    def test_model_params_present(self):
        config = _build_full_config()
        kwargs = config["model"]["kwargs"]
        assert "loss" in kwargs
        assert "num_leaves" in kwargs
        assert "learning_rate" in kwargs


class TestCreateModel:
    @patch("qlib.utils.init_instance_by_config")
    def test_creates_lgb_model(self, mock_init):
        mock_init.return_value = MagicMock()
        config = {
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {"loss": "mse"},
            }
        }
        model = create_model(config)
        mock_init.assert_called_once_with(config["model"])
        assert model is mock_init.return_value


class TestPredict:
    def test_returns_series_with_score_name(self, mock_model, mock_dataset):
        result = predict(mock_model, mock_dataset, segment="test")
        assert isinstance(result, pd.Series)
        assert result.name == "score"
        mock_model.predict.assert_called_once_with(mock_dataset, segment="test")

    def test_predict_to_dataframe(self, mock_model, mock_dataset):
        result = predict_to_dataframe(mock_model, mock_dataset, segment="test")
        assert isinstance(result, pd.DataFrame)
        assert "score" in result.columns
        assert result.index.names == ["datetime", "instrument"]


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, tmp_path):
        original = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}
        model_path = tmp_path / "model.pkl"
        save_model(original, model_path)
        assert model_path.exists()

        loaded = load_model(model_path)
        assert loaded == original

    def test_save_creates_parent_dirs(self, tmp_path):
        model_path = tmp_path / "nested" / "dir" / "model.pkl"
        save_model({"data": True}, model_path)
        assert model_path.exists()

        loaded = load_model(model_path)
        assert loaded == {"data": True}

    def test_loaded_model_preserves_types(self, tmp_path):
        original = {"array": np.array([1, 2, 3]), "label": "test"}
        model_path = tmp_path / "model.pkl"
        save_model(original, model_path)
        loaded = load_model(model_path)
        np.testing.assert_array_equal(loaded["array"], original["array"])
        assert loaded["label"] == "test"


class TestConfigLoading:
    def test_build_full_config_uses_default_paths(self):
        config = _build_full_config()
        assert "model" in config
        assert "dataset" in config

    def test_build_full_config_custom_paths(self):
        config = _build_full_config(
            model_config_path="configs/model/lightgbm.yaml",
            data_config_path="configs/data/handler_alpha158.yaml",
        )
        assert config["model"]["class"] == "LGBModel"
        assert config["dataset"]["kwargs"]["segments"]["train"] == ["2010-01-01", "2018-12-31"]
