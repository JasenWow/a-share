"""Custom Qlib operators for A-share market microstructure.

Operators:
    - LimitStatus: detect 涨跌停 (price limit hit) status
    - VWAP: volume-weighted average price proxy from daily OHLCV
    - VolumeRatio: relative volume vs. rolling mean

All operators subclass qlib.data.base.ExpressionOps so they integrate
with the Qlib Expression Engine via ``qlib.init(custom_ops=[...])``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from qlib.data.base import Expression, ExpressionOps


class _MultiFeatureOps(ExpressionOps):
    """Base class for operators that consume multiple feature expressions.

    Subclasses must set ``self._features`` to the list of child expressions
    and override ``_compute`` with the actual logic.
    """

    def _load_internal(self, instrument, start_index, end_index, *args):
        loaded = [f.load(instrument, start_index, end_index, *args) for f in self._features]
        return self._compute(loaded)

    def _compute(self, series_list: list[pd.Series]) -> pd.Series:
        raise NotImplementedError

    def get_longest_back_rolling(self):
        return max(
            (f.get_longest_back_rolling() for f in self._features),
            default=0,
        )

    def get_extended_window_size(self):
        lefts, rights = zip(
            *(f.get_extended_window_size() for f in self._features),
        )
        return max(lefts), max(rights)


# ---------------------------------------------------------------------------
# LimitStatus
# ---------------------------------------------------------------------------
class LimitStatus(_MultiFeatureOps):
    """Detect 涨跌停 (price-limit) status for A-share stocks.

    Usage in Qlib expression::

        LimitStatus($close, $high, $low)

    Returns
    -------
    pd.Series
        +1  涨停  (close ≈ high, within *tolerance*)
        -1  跌停  (close ≈ low,  within *tolerance*)
         0  otherwise
    """

    def __init__(self, close, high, low, tolerance: float = 0.001):
        self._features = [close, high, low]
        self.tolerance = tolerance

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, *self._features)

    def _compute(self, series_list: list[pd.Series]) -> pd.Series:
        close, high, low = series_list
        # Avoid division by zero when close == 0
        tol = self.tolerance
        is_up = np.isclose(close, high, rtol=tol, atol=0) & (close > 0)
        is_down = np.isclose(close, low, rtol=tol, atol=0) & (close > 0)
        return pd.Series(
            np.where(is_up, 1, np.where(is_down, -1, 0)),
            index=close.index,
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------
class VWAP(_MultiFeatureOps):
    """Volume-weighted average price proxy from daily OHLCV bars.

    Standard A-share daily VWAP proxy::

        ($open + $high + $low + $close * 2) / 5

    Usage in Qlib expression::

        VWAP($open, $high, $low, $close, $volume)

    Note: *volume* is accepted for API consistency but not used in the
    simplified proxy formula (we only have daily bars, not tick data).
    """

    def __init__(self, open_price, high, low, close, volume):
        self._features = [open_price, high, low, close, volume]

    def __str__(self):
        return "{}({},{},{},{},{})".format(type(self).__name__, *self._features)

    def _compute(self, series_list: list[pd.Series]) -> pd.Series:
        open_p, high, low, close, _volume = series_list
        return (open_p + high + low + close * 2) / 5


# ---------------------------------------------------------------------------
# VolumeRatio
# ---------------------------------------------------------------------------
class VolumeRatio(ExpressionOps):
    """Relative volume: current volume divided by its rolling mean.

    Usage in Qlib expression::

        VolumeRatio($volume, 20)

    This is equivalent to ``$volume / Mean($volume, 20)`` but offered as a
    single operator for convenience and to keep factor expressions readable.
    """

    def __init__(self, volume, window: int = 20):
        self.volume = volume
        self.window = int(window)

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.volume, self.window)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.volume.load(instrument, start_index, end_index, *args)
        rolling_mean = series.rolling(self.window, min_periods=1).mean()
        return series / rolling_mean

    def get_longest_back_rolling(self):
        return self.volume.get_longest_back_rolling() + self.window - 1

    def get_extended_window_size(self):
        ll, lr = self.volume.get_extended_window_size()
        return max(ll + self.window - 1, ll), lr
