"""Qlib initialization for A-share quantitative trading."""
from __future__ import annotations

from pathlib import Path

import qlib
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # big-a/
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "qlib_data" / "cn_data"


def _get_custom_ops() -> list:
    from big_a.factors.custom_ops import LimitStatus, VWAP, VolumeRatio

    return [LimitStatus, VWAP, VolumeRatio]


def init_qlib(provider_uri: str | Path | None = None, **kwargs) -> None:
    """Initialize Qlib with A-share config and custom operators."""
    uri = Path(provider_uri) if provider_uri else DEFAULT_DATA_DIR
    if not uri.exists():
        raise FileNotFoundError(
            f"Qlib data not found at {uri}. "
            f"Download: https://github.com/chenditc/investment_data/releases/latest "
            f"Extract to: {uri} with --strip-components=1"
        )
    kwargs.setdefault("custom_ops", _get_custom_ops())
    logger.info(f"Initializing Qlib with provider_uri={uri}")
    qlib.init(provider_uri=str(uri), region="cn", **kwargs)
