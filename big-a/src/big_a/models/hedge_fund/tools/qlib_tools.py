"""Qlib data tools for hedge fund agents — replaces financial-datasets API."""
from __future__ import annotations

import pandas as pd
from loguru import logger


def get_prices(
    instruments: list[str],
    start_date: str,
    end_date: str,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Qlib.

    Returns DataFrame with MultiIndex (instrument, datetime) and columns open, high, low, close, volume.
    """
    from qlib.data import D

    _fields = fields or ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(instruments, fields=_fields, start_time=start_date, end_time=end_date)
    df.columns = [c.lstrip("$") for c in df.columns]
    return df


def get_technical_indicators(
    instruments: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Compute technical indicators using Qlib expressions.

    Returns DataFrame with MultiIndex (instrument, datetime) and indicator columns.
    """
    from qlib.data import D

    # Qlib expression-based indicators
    fields = [
        "Mean($close, 5)",  # MA5
        "Mean($close, 10)",  # MA10
        "Mean($close, 20)",  # MA20
        "Std($change, 20)",  # 20-day volatility
        "Rank($volume)",  # Volume rank
        "Mean($volume, 5)",  # Average volume 5d
        "Mean($volume, 20)",  # Average volume 20d
        "$volume / Mean($volume, 20)",  # Volume ratio
    ]
    df = D.features(instruments, fields=fields, start_time=start_date, end_time=end_date)
    return df


def get_market_data(
    instruments: list[str],
    date: str,
) -> dict[str, object]:
    """Get market overview data for given instruments on a date.

    Returns dict with instrument -> {close, volume, change}.
    """
    from qlib.data import D

    fields = ["$close", "$volume", "$change"]
    df = D.features(instruments, fields=fields, start_time=date, end_time=date)
    df.columns = [c.lstrip("$") for c in df.columns]
    result: dict[str, object] = {}
    for inst in instruments:
        try:
            row = df.loc[inst]
            # If multiple rows (MultiIndex with datetime), take the last row
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            # Ensure we have a scalar Series, not a nested structure
            row = row.squeeze()
            if isinstance(row, pd.Series):
                result[inst] = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            else:
                result[inst] = row.to_dict()
        except KeyError:
            result[inst] = {"close": None, "volume": None, "change": None}
    return result
