"""
Data validation module for the Big-A quant system.

This module provides lightweight validation utilities built on top of
QLib data access (via the D object). It is designed to fail gracefully
when data is unavailable and to be easily testable with mock data.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Any, Dict, List

import numpy as np
from loguru import logger

# Try to import Qlib's data interface. Tests can patch this object to mock data.
try:  # pragma: no cover
    from qlib.data import D  # type: ignore
except Exception:  # pragma: no cover
    D = None  # type: ignore

# Import init_qlib helper if available in the project
try:  # pragma: no cover
    from big_a.qlib_config import init_qlib  # type: ignore
except Exception:  # pragma: no cover
    init_qlib = None  # type: ignore

_qlib_initialized: bool = False


def _ensure_qlib() -> None:
    """Initialize Qlib if possible. Safe to call multiple times."""
    global _qlib_initialized
    if _qlib_initialized:
        return
    if callable(init_qlib):
        try:
            init_qlib()
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to initialize Qlib: {}", e)
    _qlib_initialized = True


def _to_date(x: Any) -> date | None:
    if isinstance(x, date):
        return x
    if isinstance(x, datetime):
        return x.date()
    try:
        return datetime.strptime(str(x), "%Y-%m-%d").date()
    except Exception:
        return None


def check_calendar_integrity(start_date: str | None = None, end_date: str | None = None) -> Dict[str, Any]:
    """Verify trading calendar continuity.

    Returns a dict: { 'valid': bool, 'total_days': int, 'missing_days': List[str] }
    Missing days are days between the first and last trading day that are not in the
    trading calendar (i.e., potential holidays).
    """
    _ensure_qlib()
    if D is None:
        return {"valid": False, "total_days": 0, "missing_days": []}
    try:
        days = D.calendar(start_time=start_date, end_time=end_date)
    except Exception as e:  # pragma: no cover
        logger.exception("Error fetching trading calendar: {}", e)
        days = []

    if not isinstance(days, (list, tuple)):
        days = list(days)  # type: ignore

    parsed: List[date] = []
    for d in days:
        dt = _to_date(d)
        if dt is not None:
            parsed.append(dt)
    parsed.sort()

    missing_days: List[str] = []
    for i in range(1, len(parsed)):
        diff = (parsed[i] - parsed[i - 1]).days
        for delta in range(1, diff):
            missing = (parsed[i - 1] + timedelta(days=delta)).strftime("%Y-%m-%d")
            missing_days.append(missing)

    return {"valid": len(missing_days) == 0, "total_days": len(parsed), "missing_days": missing_days}


def check_price_continuity(market: str = "csi300", start_date: str | None = None, end_date: str | None = None) -> Dict[str, Any]:
    """Detect abnormal price gaps (>10% change) in non-limit moves.

    Returns { 'valid': bool, 'anomalies': List[dict] }
    Each anomaly contains: instrument, index, prev_close, cur_close, percent_change.
    """
    _ensure_qlib()
    if D is None:
        return {"valid": False, "anomalies": []}
    try:
        if hasattr(D, "list_instruments"):
            insts = D.list_instruments(market=market)
        else:
            insts = []
    except Exception:  # pragma: no cover
        insts = []

    try:
        if hasattr(D, "features") and insts:
            data = D.features(insts, fields=["close"], start_time=start_date, end_time=end_date)
        else:
            data = {}
    except Exception:  # pragma: no cover
        data = {}

    anomalies: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for inst, fields in data.items():
            if not isinstance(fields, dict):
                continue
            closes = fields.get("close")
            if closes is None:
                continue
            try:
                arr = np.asarray(closes, dtype=float)
            except Exception:
                continue
            for i in range(1, len(arr)):
                prev = float(arr[i - 1])
                cur = float(arr[i])
                if prev == 0:
                    continue
                pct = (cur - prev) / prev
                if abs(pct) > 0.1:
                    anomalies.append(
                        {
                            "instrument": inst,
                            "index": i,
                            "prev_close": prev,
                            "cur_close": cur,
                            "percent_change": float(pct),
                        }
                    )
    valid = len(anomalies) == 0
    return {"valid": valid, "anomalies": anomalies}


def check_nan_ratio(market: str = "csi300", start_date: str | None = None, end_date: str | None = None) -> Dict[str, float]:
    """Calculate NaN ratio per field across the market data.

    Returns a mapping: { 'open': 0.01, 'high': 0.0, ... }
    """
    _ensure_qlib()
    fields = ["open", "high", "low", "close", "volume"]
    # Initialize counters
    nan_counts: Dict[str, int] = {f: 0 for f in fields}
    total: int = 0

    try:
        if hasattr(D, "list_instruments"):
            insts = D.list_instruments(market=market)
        else:
            insts = []
    except Exception:  # pragma: no cover
        insts = []

    try:
        if hasattr(D, "features"):
            data = D.features(insts, fields=fields, start_time=start_date, end_time=end_date)
        else:
            data = {}
    except Exception:  # pragma: no cover
        data = {}

    # Handle common shapes: dict of instrument -> {field: values}
    if isinstance(data, dict):
        for inst, feats in data.items():
            if not isinstance(feats, dict):
                continue
            for f in fields:
                arr = feats.get(f)
                if arr is None:
                    continue
                try:
                    arr_np = np.asarray(arr, dtype=float)
                    total += arr_np.size
                    nan_counts[f] += int(np.isnan(arr_np).sum())
                except Exception:
                    continue
    else:
        # If data is a pandas DataFrame, use its columns
        try:
            import pandas as pd  # type: ignore
            if isinstance(data, pd.DataFrame):
                for f in fields:
                    if f in data.columns:
                        col = data[f]
                        total += len(col)
                        nan_counts[f] += int(col.isna().sum())
        except Exception:
            pass

    ratios: Dict[str, float] = {}
    if total > 0:
        for f in fields:
            ratios[f] = nan_counts[f] / float(total)
    else:
        for f in fields:
            ratios[f] = 0.0
    return ratios


def check_stock_coverage(market: str = "csi300") -> Dict[str, Any]:
    """Verify stock pool completeness for a market (e.g., CSI300 ~300 stocks)."""
    _ensure_qlib()
    stock_count: int = 0
    try:
        if hasattr(D, "list_instruments"):
            insts = D.list_instruments(market=market)
        else:
            insts = []
        if isinstance(insts, list):
            stock_count = len(insts)
        else:
            stock_count = 0
    except Exception:  # pragma: no cover
        stock_count = 0

    # Allow a tolerance around 300; CSI300 historically ~300 stocks
    valid = 290 <= stock_count <= 310
    return {"stock_count": stock_count, "valid": valid}


def generate_data_report(market: str = "csi300", start_date: str | None = None, end_date: str | None = None) -> Dict[str, Any]:
    """Produce a comprehensive data quality report for a market and date window."""
    report = {
        "market": market,
        "start_date": start_date,
        "end_date": end_date,
        "calendar": check_calendar_integrity(start_date, end_date),
        "price_continuity": check_price_continuity(market, start_date, end_date),
        "nan_ratio": check_nan_ratio(market, start_date, end_date),
        "stock_coverage": check_stock_coverage(market),
    }
    return report


__all__ = [
    "check_calendar_integrity",
    "check_price_continuity",
    "check_nan_ratio",
    "check_stock_coverage",
    "generate_data_report",
]
