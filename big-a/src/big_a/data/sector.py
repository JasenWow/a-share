"""申万一级行业分类集成模块.

提供申万一级行业分类数据的获取、缓存和查询功能。
数据来源：AKShare stock_industry_clf_hist_sw() (官方申万XLS，最稳定)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sector")

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # big-a/
CACHE_DIR = PROJECT_ROOT / 'data' / 'sector_data'
CACHE_FILE = CACHE_DIR / 'sw_classification.parquet'


def _to_qlib_code(raw_code: str) -> str:
    """Convert AKShare stock code to Qlib format.

    Args:
        raw_code: Stock code as pure digits (e.g., "600000", "000001", "300001")

    Returns:
        Qlib format code with exchange prefix:
        - Shanghai (6xxxxx): "SH" prefix → "SH600000"
        - Shenzhen (0xxxxx, 3xxxxx): "SZ" prefix → "SZ000001"
        - Beijing (4xxxxx, 8xxxxx): "BJ" prefix → "BJ430047"

    Examples:
        >>> _to_qlib_code("600000")
        'SH600000'
        >>> _to_qlib_code("000001")
        'SZ000001'
        >>> _to_qlib_code("300001")
        'SZ300001'
        >>> _to_qlib_code("430047")
        'BJ430047'
    """
    if raw_code.startswith('6'):
        return f'SH{raw_code}'
    elif raw_code.startswith('0') or raw_code.startswith('3'):
        return f'SZ{raw_code}'
    elif raw_code.startswith('4') or raw_code.startswith('8'):
        return f'BJ{raw_code}'
    else:
        logger.warning("Unknown stock code prefix for %s, returning as-is", raw_code)
        return raw_code


def _to_raw_code(qlib_code: str) -> str:
    """Convert Qlib format code back to AKShare format.

    Args:
        qlib_code: Qlib format code with exchange prefix (e.g., "SH600000", "SZ000001")

    Returns:
        Raw code without exchange prefix (e.g., "600000", "000001")

    Examples:
        >>> _to_raw_code("SH600000")
        '600000'
        >>> _to_raw_code("SZ000001")
        '000001'
    """
    if len(qlib_code) > 6 and qlib_code[:2] in ('SH', 'SZ', 'BJ'):
        return qlib_code[2:]
    return qlib_code


def _fetch_from_akshare(max_retries: int = 3, retry_delay: int = 5) -> Optional[Dict[str, str]]:
    """Fetch Shenwan industry classification from AKShare with retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries

    Returns:
        Dictionary mapping stock_code (Qlib format) to sector_name,
        or None if all retries fail
    """
    try:
        import akshare as ak
    except ImportError as e:
        logger.error("akshare not installed: %s", e)
        return None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching SW industry classification from AKShare (attempt %d/%d)",
                        attempt, max_retries)
            df = ak.stock_industry_clf_hist_sw()

            if df is None or df.empty:
                logger.warning("AKShare returned empty DataFrame")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                return None

            # Validate required columns
            # The actual column names may vary, check for common patterns
            stock_code_col = None
            sector_name_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'code' in col_lower and 'stock' in col_lower:
                    stock_code_col = col
                elif 'industry' in col_lower or 'sector' in col_lower:
                    sector_name_col = col

            if stock_code_col is None or sector_name_col is None:
                logger.error("Cannot find required columns in AKShare response. Columns: %s", list(df.columns))
                return None

            result = {}
            for _, row in df.iterrows():
                raw_code = str(row[stock_code_col]).strip()
                sector = str(row[sector_name_col]).strip()

                if not raw_code or not sector:
                    continue

                qlib_code = _to_qlib_code(raw_code)
                result[qlib_code] = sector

            logger.info("Successfully fetched %d stock-sector mappings from AKShare", len(result))
            return result

        except Exception as e:
            logger.error("AKShare fetch attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                logger.info("Retrying in %d seconds...", retry_delay)
                time.sleep(retry_delay)
            else:
                logger.error("All %d retries exhausted", max_retries)
                return None

    return None


def _save_to_cache(data: Dict[str, str]) -> None:
    """Save industry classification data to parquet cache.

    Args:
        data: Dictionary mapping stock_code to sector_name
    """
    try:
        import pandas as pd
    except ImportError as e:
        logger.error("pandas not installed: %s", e)
        return

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([
            {'stock_code': code, 'sector_name': sector}
            for code, sector in data.items()
        ])

        if df.empty:
            logger.warning("No data to save to cache")
            return

        if 'stock_code' not in df.columns or 'sector_name' not in df.columns:
            logger.error("Invalid data format for caching")
            return

        df.to_parquet(CACHE_FILE, index=False)
        logger.info("Saved %d stock-sector mappings to cache: %s", len(df), CACHE_FILE)

    except Exception as e:
        logger.error("Failed to save cache: %s", e)


def _load_from_cache() -> Optional[Dict[str, str]]:
    """Load industry classification data from parquet cache.

    Returns:
        Dictionary mapping stock_code to sector_name,
        or None if cache doesn't exist or is invalid
    """
    try:
        import pandas as pd
    except ImportError as e:
        logger.error("pandas not installed: %s", e)
        return None

    if not CACHE_FILE.exists():
        logger.info("Cache file does not exist: %s", CACHE_FILE)
        return None

    try:
        df = pd.read_parquet(CACHE_FILE)

        if df.empty:
            logger.warning("Cache file is empty: %s", CACHE_FILE)
            return None

        if 'stock_code' not in df.columns or 'sector_name' not in df.columns:
            logger.error("Cache file has invalid columns: %s", list(df.columns))
            return None

        result = dict(zip(df['stock_code'], df['sector_name']))
        logger.info("Loaded %d stock-sector mappings from cache: %s", len(result), CACHE_FILE)
        return result

    except Exception as e:
        logger.error("Failed to load cache: %s", e)
        return None


def fetch_sw_classification(force_refresh: bool = False) -> Dict[str, str]:
    """Fetch Shenwan industry classification data.

    This function tries to fetch fresh data from AKShare, with the following behavior:
    - If force_refresh=True: Always fetch from AKShare, ignore cache
    - If force_refresh=False: Try to load from cache first, only fetch if cache missing
    - On AKShare failure: Fall back to cache if available
    - On total failure: Return empty dict

    Args:
        force_refresh: If True, force refresh from AKShare and update cache

    Returns:
        Dictionary mapping stock_code (Qlib format like "SH600000") to
        sector_name (申万行业名 like "银行"). Returns empty dict on failure.
    """
    if not force_refresh:
        cached_data = _load_from_cache()
        if cached_data is not None:
            return cached_data

    fresh_data = _fetch_from_akshare()

    if fresh_data is None:
        logger.warning("AKShare fetch failed, attempting cache fallback")
        cached_data = _load_from_cache()
        if cached_data is not None:
            logger.info("Successfully fell back to cached data")
            return cached_data
        else:
            logger.error("Both AKShare and cache failed, returning empty dict")
            return {}

    _save_to_cache(fresh_data)
    return fresh_data


def get_stock_sector(stock_code: str) -> Optional[str]:
    """Get the sector name for a given stock code.

    This function uses cached data. To refresh the cache, call refresh_sector_data().

    Args:
        stock_code: Stock code in either Qlib format (e.g., "SH600000")
                    or raw format (e.g., "600000")

    Returns:
        Sector name (e.g., "银行") or None if stock not found or data unavailable

    Examples:
        >>> get_stock_sector("SH600000")
        '银行'
        >>> get_stock_sector("600000")  # Auto-converts to Qlib format
        '银行'
        >>> get_stock_sector("SZ000001")  # Ping An Bank
        '银行'
    """
    if not stock_code.startswith(('SH', 'SZ', 'BJ')):
        stock_code = _to_qlib_code(stock_code)

    data = fetch_sw_classification(force_refresh=False)

    return data.get(stock_code)


def get_sector_stocks(sector_name: str) -> List[str]:
    """Get all stock codes in a given sector.

    This function uses cached data. To refresh the cache, call refresh_sector_data().

    Args:
        sector_name: Sector name (e.g., "银行")

    Returns:
        List of stock codes in Qlib format (e.g., ["SH600000", "SH601398", ...]).
        Returns empty list if sector not found or data unavailable.

    Examples:
        >>> bank_stocks = get_sector_stocks("银行")
        >>> len(bank_stocks) > 0
        True
        >>> "SH600000" in bank_stocks
        True
    """
    data = fetch_sw_classification(force_refresh=False)

    stocks = [code for code, sector in data.items() if sector == sector_name]
    return stocks


def refresh_sector_data() -> None:
    """Refresh sector classification data from AKShare.

    This function forces a refresh from AKShare and updates the cache.
    It will raise an exception if the refresh fails completely (including cache fallback).

    Raises:
        RuntimeError: If both AKShare fetch and cache fallback fail

    Examples:
        >>> refresh_sector_data()  # Fetches fresh data from AKShare
    """
    logger.info("Refreshing sector data from AKShare...")
    data = fetch_sw_classification(force_refresh=True)

    if not data:
        raise RuntimeError("Failed to refresh sector data: both AKShare and cache failed")

    logger.info("Successfully refreshed %d stock-sector mappings", len(data))


__all__ = [
    'fetch_sw_classification',
    'get_stock_sector',
    'get_sector_stocks',
    'refresh_sector_data',
    '_to_qlib_code',
    '_to_raw_code',
]
