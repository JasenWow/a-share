"""Tests for KronosSignalGenerator — model loading mocked to avoid HuggingFace downloads."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from big_a.models.kronos import (
    LOOKBACK,
    KronosSignalGenerator,
    _prepare_stock_sequence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_rows)
    close = 10.0 + np.cumsum(rng.standard_normal(n_rows) * 0.2)
    df = pd.DataFrame({
        "open": close + rng.standard_normal(n_rows) * 0.1,
        "high": close + np.abs(rng.standard_normal(n_rows)) * 0.2,
        "low": close - np.abs(rng.standard_normal(n_rows)) * 0.2,
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, size=n_rows).astype(float),
    }, index=dates)
    return df


@pytest.fixture
def sample_stock_df():
    return _make_ohlcv(120)


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    pred_len = 10
    future_dates = pd.bdate_range("2024-06-20", periods=pred_len)
    pred_df = pd.DataFrame({
        "open": np.full(pred_len, 12.0),
        "high": np.full(pred_len, 12.5),
        "low": np.full(pred_len, 11.5),
        "close": np.linspace(12.0, 13.0, pred_len),
        "volume": np.full(pred_len, 500000.0),
        "amount": np.full(pred_len, 6e6),
    }, index=future_dates)
    predictor.predict.return_value = pred_df
    return predictor


# ---------------------------------------------------------------------------
# Tests: preprocess / _prepare_stock_sequence
# ---------------------------------------------------------------------------

class TestPrepareStockSequence:
    def test_sufficient_data_returns_tail(self, sample_stock_df):
        result = _prepare_stock_sequence(sample_stock_df, lookback=90)
        assert result is not None
        assert len(result) == 90

    def test_insufficient_data_returns_none(self, sample_stock_df):
        result = _prepare_stock_sequence(sample_stock_df, lookback=200)
        assert result is None

    def test_nan_data_returns_none(self):
        df = _make_ohlcv(100)
        df.iloc[50, 0] = np.nan
        result = _prepare_stock_sequence(df, lookback=90)
        assert result is None

    def test_exact_length_returns_df(self):
        df = _make_ohlcv(90)
        result = _prepare_stock_sequence(df, lookback=90)
        assert result is not None
        assert len(result) == 90


# ---------------------------------------------------------------------------
# Tests: KronosSignalGenerator.preprocess (static method)
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_delegates_to_prepare_sequence(self, sample_stock_df):
        result = KronosSignalGenerator.preprocess(sample_stock_df, lookback=90)
        assert result is not None
        assert len(result) == 90

    def test_returns_none_for_short_data(self):
        df = _make_ohlcv(50)
        assert KronosSignalGenerator.preprocess(df, lookback=90) is None


# ---------------------------------------------------------------------------
# Tests: KronosSignalGenerator.load_model (mocked)
# ---------------------------------------------------------------------------

class TestLoadModel:
    @patch("big_a.models.kronos.KronosTokenizer")
    @patch("big_a.models.kronos.Kronos")
    @patch("big_a.models.kronos.KronosPredictor")
    def test_load_model_sets_predictor(self, MockPred, MockKronos, MockTok):
        gen = KronosSignalGenerator()
        gen.load_model()
        assert gen._predictor is not None

    @patch("big_a.models.kronos.KronosTokenizer")
    @patch("big_a.models.kronos.Kronos")
    @patch("big_a.models.kronos.KronosPredictor")
    def test_predictor_property_raises_before_load(self, MockPred, MockKronos, MockTok):
        gen = KronosSignalGenerator()
        with pytest.raises(RuntimeError, match="load_model"):
            _ = gen.predictor


# ---------------------------------------------------------------------------
# Tests: predict (single stock)
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_dataframe(self, sample_stock_df, mock_predictor):
        gen = KronosSignalGenerator()
        gen._predictor = mock_predictor
        result = gen.predict(sample_stock_df)
        assert result is not None
        assert "close" in result.columns
        assert len(result) == 10

    def test_predict_returns_none_for_short_data(self, mock_predictor):
        gen = KronosSignalGenerator()
        gen._predictor = mock_predictor
        short_df = _make_ohlcv(50)
        result = gen.predict(short_df)
        assert result is None

    def test_predict_calls_predictor_with_correct_args(self, sample_stock_df, mock_predictor):
        gen = KronosSignalGenerator(pred_len=10)
        gen._predictor = mock_predictor
        gen.predict(sample_stock_df)
        mock_predictor.predict.assert_called_once()
        call_kwargs = mock_predictor.predict.call_args
        assert call_kwargs.kwargs.get("pred_len") == 10 or call_kwargs[1].get("pred_len") == 10


# ---------------------------------------------------------------------------
# Tests: generate_signals (batch)
# ---------------------------------------------------------------------------

class TestGenerateSignals:
    def test_empty_result_for_no_instruments(self, mock_predictor):
        gen = KronosSignalGenerator()
        gen._predictor = mock_predictor
        with patch.object(gen, "load_data", return_value=pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
        )):
            signals = gen.generate_signals(
                instruments=[],
                start_date="2024-01-01",
                end_date="2024-06-30",
            )
        assert signals.empty

    def test_signal_format_is_correct(self, mock_predictor):
        gen = KronosSignalGenerator(pred_len=10, signal_mode="mean")
        gen._predictor = mock_predictor

        n_rows = 120
        stock_a = _make_ohlcv(n_rows, seed=1)
        stock_b = _make_ohlcv(n_rows, seed=2)

        data = pd.concat([
            stock_a.assign(instrument="SH600000"),
            stock_b.assign(instrument="SZ000001"),
        ]).set_index(["instrument", stock_a.index.append(stock_b.index).to_series().repeat(1).values[:len(stock_a) + len(stock_b)]])

        # Build proper MultiIndex data
        idx_tuples = (
            [("SH600000", t) for t in stock_a.index]
            + [("SZ000001", t) for t in stock_b.index]
        )
        multi_idx = pd.MultiIndex.from_tuples(idx_tuples, names=["instrument", "date"])
        combined = pd.concat([stock_a, stock_b])
        combined.index = multi_idx
        combined.columns = [c.lstrip("$") for c in combined.columns]

        with patch.object(gen, "load_data", return_value=combined):
            signals = gen.generate_signals(
                instruments=["SH600000", "SZ000001"],
                start_date="2024-01-01",
                end_date="2024-06-30",
            )

        assert "score" in signals.columns
        assert signals.index.names == ["datetime", "instrument"]
        assert len(signals) == 2

    def test_signal_score_is_float(self, mock_predictor):
        gen = KronosSignalGenerator(signal_mode="last")
        gen._predictor = mock_predictor

        stock_df = _make_ohlcv(120)
        idx = pd.MultiIndex.from_tuples(
            [("SH600000", t) for t in stock_df.index],
            names=["instrument", "date"],
        )
        stock_df.index = idx

        with patch.object(gen, "load_data", return_value=stock_df):
            signals = gen.generate_signals(
                instruments=["SH600000"],
                start_date="2024-01-01",
                end_date="2024-06-30",
            )

        assert len(signals) == 1
        assert isinstance(signals["score"].iloc[0], (float, np.floating))

    def test_signal_mode_last(self, mock_predictor):
        gen = KronosSignalGenerator(signal_mode="last")
        gen._predictor = mock_predictor

        stock_df = _make_ohlcv(120)
        idx = pd.MultiIndex.from_tuples(
            [("SH600000", t) for t in stock_df.index],
            names=["instrument", "date"],
        )
        stock_df.index = idx

        with patch.object(gen, "load_data", return_value=stock_df):
            signals = gen.generate_signals(
                instruments=["SH600000"],
                start_date="2024-01-01",
                end_date="2024-06-30",
            )
        assert len(signals) == 1


# ---------------------------------------------------------------------------
# Tests: constructor defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_device_is_cpu(self):
        gen = KronosSignalGenerator()
        assert gen.device == "cpu"

    def test_default_lookback(self):
        gen = KronosSignalGenerator()
        assert gen.lookback == LOOKBACK

    def test_custom_params(self):
        gen = KronosSignalGenerator(
            device="cpu", lookback=60, pred_len=5, signal_mode="last",
        )
        assert gen.lookback == 60
        assert gen.pred_len == 5
        assert gen.signal_mode == "last"


# ---------------------------------------------------------------------------
# Tests: config loading
# ---------------------------------------------------------------------------

class TestConfig:
    def test_kronos_config_loads(self):
        from big_a.config import load_config
        config = load_config("configs/model/kronos.yaml")
        assert "kronos" in config
        assert config["kronos"]["device"] == "cpu"
        assert config["kronos"]["lookback"] == 90

    def test_config_signal_mode(self):
        from big_a.config import load_config
        config = load_config("configs/model/kronos.yaml")
        assert config["kronos"]["signal_mode"] == "mean"
