"""Kronos pre-trained model inference for A-share price prediction.

Loads the Kronos base model from HuggingFace and generates trading signals
compatible with Qlib backtest. CPU-only, no fine-tuning.

Model: NeoQuasar/Kronos-base (BSQ tokenizer + DualHead predictor)
Tokenizer: NeoQuasar/Kronos-Tokenizer-base
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger

from big_a.models.kronos_model import Kronos, KronosPredictor, KronosTokenizer

from big_a.config import load_config
from big_a.qlib_config import init_qlib

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID = "NeoQuasar/Kronos-base"
MAX_CONTEXT = 512
LOOKBACK = 90          # trading days of history per stock
PRED_LEN = 10          # trading days to predict
SIGNAL_MODE: Literal["last", "mean"] = "mean"  # how to aggregate multi-step pred


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_ohlcv(
    instruments: list[str],
    start_date: str,
    end_date: str,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Qlib for the given instruments and date range.

    Returns DataFrame with MultiIndex (instrument, datetime) and columns
    ``$open, $high, $low, $close, $volume``.
    """
    import qlib
    from qlib.data import D

    fields = fields or ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(
        instruments,
        fields=fields,
        start_time=start_date,
        end_time=end_date,
    )
    return df


def _prepare_stock_sequence(
    stock_df: pd.DataFrame,
    lookback: int = LOOKBACK,
) -> pd.DataFrame | None:
    """Slice the last *lookback* rows for a single stock.

    Returns None if the stock has fewer than *lookback* rows or contains NaN.
    """
    if len(stock_df) < lookback:
        return None
    tail = stock_df.iloc[-lookback:]
    if tail.isnull().any().any():
        return None
    return tail


# ---------------------------------------------------------------------------
# Kronos wrapper
# ---------------------------------------------------------------------------

class KronosSignalGenerator:
    """Load Kronos pre-trained model and produce Qlib-compatible signals.

    Usage::

        gen = KronosSignalGenerator()
        gen.load_model()
        signals = gen.generate_signals(
            instruments=["SH600000", "SZ000001"],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        # signals: DataFrame(index=[datetime, instrument], columns=["score"])

    Parameters
    ----------
    tokenizer_id : str
        HuggingFace model ID for the Kronos tokenizer.
    model_id : str
        HuggingFace model ID for the Kronos predictor.
    device : str
        Torch device string.  Defaults to ``"cpu"``.
    lookback : int
        Number of trading days used as input context per stock.
    pred_len : int
        Number of future trading days Kronos predicts.
    max_context : int
        Max context length for the Kronos predictor.
    signal_mode : str
        ``"last"`` uses last predicted close minus last actual close;
        ``"mean"`` uses the mean of all predicted closes minus last actual close.
    """

    def __init__(
        self,
        tokenizer_id: str = TOKENIZER_ID,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        lookback: int = LOOKBACK,
        pred_len: int = PRED_LEN,
        max_context: int = MAX_CONTEXT,
        signal_mode: Literal["last", "mean"] = SIGNAL_MODE,
    ) -> None:
        self.tokenizer_id = tokenizer_id
        self.model_id = model_id
        self.device = device
        self.lookback = lookback
        self.pred_len = pred_len
        self.max_context = max_context
        self.signal_mode = signal_mode
        self._predictor: KronosPredictor | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, local_files_only: bool = False) -> None:
        """Download (if needed) and load the Kronos tokenizer + model.

        Parameters
        ----------
        local_files_only : bool
            If True, skip download and rely on the HuggingFace cache.
            Raises if the cache is empty.
        """
        logger.info(
            "Loading Kronos tokenizer={} model={} device={}",
            self.tokenizer_id,
            self.model_id,
            self.device,
        )
        try:
            tokenizer = KronosTokenizer.from_pretrained(
                self.tokenizer_id, local_files_only=local_files_only,
            )
            model = Kronos.from_pretrained(
                self.model_id, local_files_only=local_files_only,
            )
        except Exception:
            if local_files_only:
                raise
            logger.warning("Local cache miss — downloading from HuggingFace …")
            tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
            model = Kronos.from_pretrained(self.model_id)

        self._predictor = KronosPredictor(
            model, tokenizer, device=self.device, max_context=self.max_context,
        )
        logger.info("Kronos model loaded successfully")

    @property
    def predictor(self) -> KronosPredictor:
        if self._predictor is None:
            raise RuntimeError("Model not loaded — call load_model() first")
        return self._predictor

    # ------------------------------------------------------------------
    # Data loading (from Qlib)
    # ------------------------------------------------------------------

    def load_data(
        self,
        instruments: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data via Qlib.

        Returns a DataFrame with MultiIndex (instrument, datetime) and columns
        ``open, high, low, close, volume`` (Qlib ``$`` prefix stripped).
        """
        init_qlib()
        raw = _fetch_ohlcv(instruments, start_date, end_date)
        # Strip $ prefix for Kronos compatibility
        raw.columns = [c.lstrip("$") for c in raw.columns]
        logger.info(
            "Loaded OHLCV data: {} rows, {} instruments",
            len(raw),
            len(raw.index.get_level_values(0).unique()),
        )
        return raw

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(
        stock_df: pd.DataFrame,
        lookback: int = LOOKBACK,
    ) -> pd.DataFrame | None:
        """Prepare a single stock's OHLCV for Kronos inference.

        Takes the last *lookback* rows and validates no NaN remain.
        Returns None if the stock cannot be used.
        """
        return _prepare_stock_sequence(stock_df, lookback=lookback)

    # ------------------------------------------------------------------
    # Prediction (single stock)
    # ------------------------------------------------------------------

    def predict(self, stock_df: pd.DataFrame) -> pd.DataFrame | None:
        """Run Kronos inference on a single prepared stock DataFrame.

        Parameters
        ----------
        stock_df : pd.DataFrame
            OHLCV data for one stock with at least *lookback* rows.
            Index must be DatetimeIndex.

        Returns
        -------
        pd.DataFrame or None
            Kronos prediction DataFrame (columns: open, high, low, close,
            volume, amount; index: predicted business days), or None on error.
        """
        tail = self.preprocess(stock_df, lookback=self.lookback)
        if tail is None:
            return None

        x_df = tail[["open", "high", "low", "close", "volume"]].copy()
        # Kronos expects an 'amount' column; approximate as volume * avg_price
        x_df["amount"] = x_df["volume"] * x_df[["open", "high", "low", "close"]].mean(axis=1)

        x_timestamp = tail.index
        y_timestamp = pd.bdate_range(
            start=x_timestamp[-1] + pd.Timedelta(days=1),
            periods=self.pred_len,
        )

        try:
            with torch.no_grad():
                pred_df = self.predictor.predict(
                    df=x_df,
                    x_timestamp=pd.Series(x_timestamp),
                    y_timestamp=pd.Series(y_timestamp),
                    pred_len=self.pred_len,
                    T=1.0,
                    top_k=0,
                    top_p=0.9,
                    sample_count=1,
                    verbose=False,
                )
            return pred_df
        except Exception as exc:
            logger.warning("Kronos prediction failed: {}", exc)
            return None

    # ------------------------------------------------------------------
    # Signal generation (batch across instruments)
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        instruments: list[str],
        start_date: str,
        end_date: str,
        lookback_override: int | None = None,
    ) -> pd.DataFrame:
        """Generate trading signals for multiple instruments.

        Workflow:
        1. Load OHLCV data from Qlib for all instruments.
        2. For each instrument, take the last *lookback* days.
        3. Run Kronos inference → get predicted close prices.
        4. Compute ``score = predicted_close - last_actual_close``.
        5. Return DataFrame(index=[datetime, instrument], columns=["score"]).

        The *datetime* in the output is the last actual date in the input
        window — i.e., the date the signal is available for trading.

        Parameters
        ----------
        instruments : list[str]
            Qlib instrument codes (e.g. ``["SH600000"]``).
        start_date, end_date : str
            Date range for fetching data.  Must extend far enough before the
            desired signal dates to cover the lookback window.
        lookback_override : int or None
            Override the default lookback for this call.

        Returns
        -------
        pd.DataFrame
            MultiIndex (datetime, instrument), column ``score``.
        """
        if self._predictor is None:
            self.load_model()

        data = self.load_data(instruments, start_date, end_date)
        lookback = lookback_override or self.lookback

        records: list[tuple] = []

        # Group by instrument (level 0 of MultiIndex)
        for instrument, stock_df in data.groupby(level=0):
            # Drop the instrument level for processing
            stock_df = stock_df.droplevel(0).sort_index()
            if len(stock_df) < lookback:
                logger.debug("Skipping {}: only {} rows (< {})", instrument, len(stock_df), lookback)
                continue

            tail = stock_df.iloc[-lookback:]
            if tail.isnull().any().any():
                logger.debug("Skipping {}: NaN in input window", instrument)
                continue

            pred_df = self.predict(tail)
            if pred_df is None or pred_df.empty:
                continue

            last_close = tail["close"].iloc[-1]
            pred_closes = pred_df["close"].values

            if self.signal_mode == "last":
                score = float(pred_closes[-1] - last_close)
            else:
                score = float(np.mean(pred_closes) - last_close)

            # Signal date = last date in the input window (when signal is known)
            signal_date = tail.index[-1]
            records.append((signal_date, instrument, score))

        if not records:
            logger.warning("No signals generated")
            return pd.DataFrame(
                columns=["score"],
                index=pd.MultiIndex.from_tuples([], names=["datetime", "instrument"]),
            )

        signal_df = pd.DataFrame(records, columns=["datetime", "instrument", "score"])
        result = signal_df.set_index(["datetime", "instrument"])[["score"]]
        # Ensure index names for Qlib compatibility
        result.index.names = ["datetime", "instrument"]
        logger.info("Generated {} signals", len(result))
        return result
