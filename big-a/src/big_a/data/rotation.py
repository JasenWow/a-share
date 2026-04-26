"""板块轮动信号模块，基于申万一级行业 N 日涨幅排名。

通过计算每个申万行业板块的等权重指数在 N 日内的涨跌幅，为板块轮动策略提供信号。
"""

import datetime
from pathlib import Path
from typing import List, Tuple

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("rotation")

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # big-a/


def _calc_sector_momentum(sector_name: str, lookback_days: int) -> float:
    """Calculate momentum for a single sector.

    Args:
        sector_name: Sector name (e.g., "银行")
        lookback_days: Number of days to look back for momentum calculation

    Returns:
        Momentum score as percentage change. Returns -inf if sector has < 5 stocks
        or if data is unavailable.
    """
    from big_a.data.sector import get_sector_stocks

    sector_stocks = get_sector_stocks(sector_name)

    if len(sector_stocks) < 5:
        logger.debug("Skipping sector %s: only %d stocks (minimum 5 required)", sector_name, len(sector_stocks))
        return float('-inf')

    # Use * 2 to account for weekends/holidays
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")

    try:
        from qlib.data import D

        df = D.features(
            sector_stocks,
            fields=["$close"],
            start_time=start_date,
            end_time=end_date
        )

        if df.empty:
            logger.warning("No data returned for sector %s between %s and %s", sector_name, start_date, end_date)
            return float('-inf')

        price_matrix = df["$close"].unstack(level="instrument")
        sector_index = price_matrix.mean(axis=1)

        if len(sector_index) < lookback_days + 1:
            logger.warning("Insufficient data points for sector %s: %d (need at least %d)",
                         sector_name, len(sector_index), lookback_days + 1)
            return float('-inf')

        current_price = sector_index.iloc[-1]
        historical_price = sector_index.iloc[-lookback_days - 1]

        momentum = (current_price / historical_price - 1) * 100

        return momentum

    except Exception as e:
        logger.error("Error calculating momentum for sector %s: %s", sector_name, e)
        return float('-inf')


def rank_sectors(lookback_days: int = 20) -> List[Tuple[str, float]]:
    """Rank all Shenwan level-1 sectors by momentum.

    Args:
        lookback_days: Number of days to look back for momentum calculation (default: 20)

    Returns:
        List of tuples [(sector_name, momentum_score), ...] sorted by momentum descending.
        Returns empty list if no sector data is available.
    """
    from big_a.data.sector import fetch_sw_classification
    from big_a.qlib_config import init_qlib

    try:
        init_qlib()
    except Exception as e:
        logger.error("Failed to initialize Qlib: %s", e)
        return []

    classification = fetch_sw_classification(force_refresh=False)

    if not classification:
        logger.warning("No sector classification data available")
        return []

    sector_names = set(classification.values())

    if not sector_names:
        logger.warning("No sectors found in classification data")
        return []

    logger.info("Calculating momentum for %d sectors with lookback_days=%d", len(sector_names), lookback_days)

    results = []
    for sector_name in sector_names:
        momentum = _calc_sector_momentum(sector_name, lookback_days)
        if momentum != float('-inf'):
            results.append((sector_name, momentum))
        else:
            logger.debug("Skipping sector %s due to data issues", sector_name)

    if not results:
        logger.warning("No valid sector momentum data available")
        return []

    results.sort(key=lambda x: x[1], reverse=True)

    logger.info("Successfully ranked %d sectors", len(results))
    return results


def get_top_sectors(top_k: int = 5, lookback_days: int = 20) -> List[str]:
    """Get the top-k sectors by momentum.

    Args:
        top_k: Number of top sectors to return (default: 5)
        lookback_days: Number of days to look back for momentum calculation (default: 20)

    Returns:
        List of top-k sector names sorted by momentum descending.
        Returns empty list if no sector data is available.
    """
    ranked_sectors = rank_sectors(lookback_days)

    if not ranked_sectors:
        return []

    top_k = min(top_k, len(ranked_sectors))
    return [sector[0] for sector in ranked_sectors[:top_k]]


__all__ = [
    'rank_sectors',
    'get_top_sectors',
    '_calc_sector_momentum',
]
