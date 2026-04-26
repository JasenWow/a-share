"""股票池构建器模块.

合并 CSI300 + 自选股 + 板块轮动过滤，生成最终股票池。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

# Logging: prefer loguru if available, otherwise fallback to standard logging
try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("screener")

# Project layout assumptions
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # big-a/
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'qlib_data' / 'cn_data'
FEATURES_DIR = DEFAULT_DATA_DIR / 'features'

if TYPE_CHECKING:
    from typing import List, Dict, Any


def load_base_pool(pool_name: str) -> List[str]:
    """Load base pool instruments from Qlib.

    Args:
        pool_name: Pool name like "csi300", "csi500", "csi800", "csi1000", "all"

    Returns:
        List of Qlib-format instrument codes

    Examples:
        >>> instruments = load_base_pool("csi300")
        >>> len(instruments) > 0
        True
    """
    from big_a.qlib_config import init_qlib
    from qlib.data import D

    init_qlib()

    instruments = D.instruments(pool_name)
    return D.list_instruments(instruments=instruments, as_list=True)


def load_watchlist(path: str | Path) -> List[str]:
    """Load watchlist from YAML config file.

    Args:
        path: Path to watchlist YAML file (relative to big-a/ or absolute)

    Returns:
        List of Qlib-format instrument codes

    Examples:
        >>> codes = load_watchlist("configs/watchlist.yaml")
        >>> "SH600519" in codes
        True
    """
    import yaml

    from big_a.config import load_config

    config = load_config(path)
    watchlist = config.get("watchlist", {})

    if not isinstance(watchlist, dict):
        logger.warning("watchlist is not a dict in config: %s", path)
        return []

    # Extract keys (stock codes) from the watchlist dict
    return list(watchlist.keys())


def validate_instruments(codes: List[str]) -> List[str]:
    """Validate instruments by checking if feature directories exist.

    Args:
        codes: List of Qlib-format instrument codes

    Returns:
        Filtered list of codes that have feature data available

    Examples:
        >>> codes = ["SH600000", "SH600001", "INVALID"]
        >>> valid = validate_instruments(codes)
        >>> "INVALID" not in valid
        True
    """
    if not FEATURES_DIR.exists():
        logger.warning("Features directory does not exist: %s", FEATURES_DIR)
        return []

    valid_codes = []
    for code in codes:
        feature_dir = FEATURES_DIR / code.lower()
        if feature_dir.exists() and feature_dir.is_dir():
            valid_codes.append(code)
        else:
            logger.warning("Instrument %s has no feature data, excluding", code)

    logger.info("Validated %d/%d instruments", len(valid_codes), len(codes))
    return valid_codes


def filter_by_sectors(codes: List[str], active_sectors: List[str]) -> List[str]:
    """Filter instruments to only those in active sectors.

    Args:
        codes: List of Qlib-format instrument codes
        active_sectors: List of sector names (e.g., ["银行", "医药生物"])

    Returns:
        Filtered list of codes that belong to active sectors

    Examples:
        >>> codes = ["SH600000", "SH600519"]  # 银行, 食品饮料
        >>> active = ["银行"]
        >>> filtered = filter_by_sectors(codes, active)
        >>> "SH600000" in filtered and "SH600519" not in filtered
        True
    """
    from big_a.data.sector import get_sector_stocks

    if not active_sectors:
        logger.info("No active sectors specified, returning all codes")
        return codes

    # Build set of all stocks in active sectors
    sector_stock_set = set()
    for sector_name in active_sectors:
        try:
            stocks = get_sector_stocks(sector_name)
            sector_stock_set.update(stocks)
            logger.debug("Sector %s has %d stocks", sector_name, len(stocks))
        except Exception as e:
            logger.warning("Failed to get stocks for sector %s: %s", sector_name, e)

    # Filter codes to intersection
    code_set = set(codes)
    filtered = sorted(code_set & sector_stock_set)

    logger.info(
        "Filtered %d codes by %d active sectors: %d -> %d",
        len(codes),
        len(active_sectors),
        len(codes),
        len(filtered),
    )
    return filtered


def build_universe(config_path: str | None = None) -> List[str]:
    """Build stock universe by merging base pool + watchlist + sector rotation filter.

    Args:
        config_path: Path to universe config YAML. If None, uses default
                     "configs/data/universe.yaml"

    Returns:
        Sorted, deduplicated list of Qlib-format instrument codes

    Examples:
        >>> universe = build_universe()
        >>> len(universe) > 0
        True
        >>> isinstance(universe[0], str)
        True
    """
    import yaml

    from big_a.config import load_config

    # Load universe config
    if config_path is None:
        config_path = "configs/data/universe.yaml"

    config = load_config(config_path)
    universe_cfg = config.get("universe", {})

    # Step 1: Load base pool
    base_pool_name = universe_cfg.get("base_pool", "csi300")
    logger.info("Loading base pool: %s", base_pool_name)
    base_pool = load_base_pool(base_pool_name)
    logger.info("Base pool has %d instruments", len(base_pool))

    # Step 2: Load watchlist
    watchlist_path = universe_cfg.get("watchlist", "configs/watchlist.yaml")
    logger.info("Loading watchlist from: %s", watchlist_path)
    watchlist_codes = load_watchlist(watchlist_path)
    logger.info("Watchlist has %d instruments", len(watchlist_codes))

    # Step 3: Merge and deduplicate
    merged = sorted(set(base_pool) | set(watchlist_codes))
    logger.info("Merged universe has %d instruments", len(merged))

    # Step 4: Validate instruments (check feature data availability)
    validated = validate_instruments(merged)

    # Step 5: Apply sector rotation filter if enabled
    sector_rotation_cfg = universe_cfg.get("sector_rotation", {})
    if sector_rotation_cfg.get("enabled", False):
        try:
            # Import rotation module conditionally (it may not exist yet)
            try:
                from big_a.data.rotation import get_top_sectors
            except ImportError:
                logger.warning(
                    "rotation module not available, skipping sector rotation filter"
                )
                get_top_sectors = None

            if get_top_sectors is not None:
                lookback_days = sector_rotation_cfg.get("lookback_days", 20)
                top_k = sector_rotation_cfg.get("top_k_sectors", 5)

                logger.info(
                    "Getting top %d sectors from last %d days",
                    top_k,
                    lookback_days,
                )
                active_sectors = get_top_sectors(
                    lookback_days=lookback_days,
                    top_k=top_k,
                )
                logger.info("Active sectors: %s", active_sectors)

                validated = filter_by_sectors(validated, active_sectors)
            else:
                logger.info("Sector rotation enabled but rotation module not available")
        except Exception as e:
            logger.warning(
                "Failed to apply sector rotation filter: %s. Continuing without filter.",
                e,
            )
    else:
        logger.info("Sector rotation filter disabled")

    logger.info("Final universe has %d instruments", len(validated))
    return validated


__all__ = [
    'load_base_pool',
    'load_watchlist',
    'validate_instruments',
    'filter_by_sectors',
    'build_universe',
]
