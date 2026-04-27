"""每日增量数据更新模块.

使用 akshare 获取最新日线数据，追加到 Qlib 二进制格式文件。
仅更新增量部分，避免每天下载 524MB 全量 tarball。

Qlib 二进制格式说明 (已通过实验验证):
    - .bin 文件 = numpy float32 (little-endian) 数组
    - 第一个元素 = start_index (该股票在 calendar 中的起始索引)
    - 后续元素 = 每日特征值，与 calendar 日期一一对应

归一化公式 (已验证):
    - stored_close  = raw_close  * factor
    - stored_open   = raw_open   * factor
    - stored_high   = raw_high   * factor
    - stored_low    = raw_low    * factor
    - stored_volume = raw_volume / factor   (注意: 与价格方向相反!)
    - stored_amount = raw_amount / 1000     (千元单位)
    - stored_vwap   = raw_vwap  * factor    (raw_vwap = raw_amount / (raw_volume * 100))
    - stored_change = close[t] / close[t-1] - 1  (stored 单位下计算)
    - stored_factor = factor (不变, 除非除权除息)
    - stored_adjclose = close * adjclose_scale  (adjclose_scale = adjclose[1] / close[1], 常数)

Usage:
    from big_a.data.daily_updater import update_daily
    update_daily()  # 更新到最新交易日
"""
import time
from pathlib import Path
from typing import Optional

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("daily_updater")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # big-a/
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'qlib_data' / 'cn_data'


def _data_dir(data_dir: Optional[str | Path]) -> Path:
    """Resolve data directory path."""
    return Path(data_dir) if data_dir else DEFAULT_DATA_DIR


def _to_raw_code(qlib_code: str) -> str:
    """Convert Qlib format code to akshare format (digits only)."""
    if len(qlib_code) > 6 and qlib_code[:2] in ('SH', 'SZ', 'BJ'):
        return qlib_code[2:]
    return qlib_code


def _get_missing_dates(data_dir: Path) -> list[str]:
    """Find trading dates that need to be added after last calendar date.

    Uses akshare trading calendar to determine valid trading days,
    excluding weekends and holidays.

    Returns:
        List of dates in YYYY-MM-DD format that are missing from calendar.
    """
    import akshare as ak

    # Get last date from calendar
    day_file = data_dir / 'calendars' / 'day.txt'
    last_date = None
    if day_file.exists():
        lines = []
        with day_file.open('r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s:
                    lines.append(s)
        if lines:
            last_date = lines[-1].split()[0]

    if not last_date:
        logger.warning("No calendar found, cannot determine missing dates")
        return []

    # Get trading calendar from akshare
    try:
        df_dates = ak.tool_trade_date_hist_sina()
        all_dates = set()
        for d in df_dates['trade_date']:
            all_dates.add(str(d)[:10])
    except Exception as e:
        logger.error("Failed to fetch trading dates from akshare: %s", e)
        return []

    # Find missing dates: trading dates after last_date, up to and including today
    import datetime
    today = datetime.date.today().strftime('%Y-%m-%d')
    missing_dates = sorted(d for d in all_dates if last_date < d <= today)

    if not missing_dates:
        logger.info("Calendar is up to date: %s", last_date)

    return missing_dates


def _get_stock_list(data_dir: Path) -> list[str]:
    """Read unique stock codes from instruments/all.txt.

    Returns:
        Sorted, deduplicated list of Qlib-format codes (e.g., "SH600000").
    """
    instr_file = data_dir / 'instruments' / 'all.txt'
    if not instr_file.exists():
        logger.error("Instruments file not found: %s", instr_file)
        return []

    stocks = set()
    with instr_file.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0].strip():
                code = parts[0].strip()
                if code and not code.startswith('#'):
                    stocks.add(code)

    stock_list = sorted(stocks)
    logger.info("Loaded %d unique stocks from instruments file", len(stock_list))
    return stock_list


def _fetch_daily_data(stock_code: str, date: str) -> Optional[dict]:
    """Fetch one day's OHLCV data from akshare.

    Args:
        stock_code: Qlib format code (e.g., "SH600000")
        date: Date in YYYY-MM-DD format

    Returns:
        dict with keys: open, close, high, low, volume, amount
        or None if no data (suspended, delisted, etc.)
    """
    import akshare as ak

    raw_code = _to_raw_code(stock_code)
    try:
        df = ak.stock_zh_a_hist(
            symbol=raw_code,
            period='daily',
            start_date=date.replace('-', ''),
            end_date=date.replace('-', ''),
            adjust=''  # 不复权: raw prices
        )
        if df is None or df.empty:
            return None

        row = df.iloc[0]
        return {
            'open': float(row['开盘']),
            'close': float(row['收盘']),
            'high': float(row['最高']),
            'low': float(row['最低']),
            'volume': float(row['成交量']),
            'amount': float(row['成交额']),
        }

    except Exception as e:
        logger.debug("Failed to fetch data for %s on %s: %s", stock_code, date, e)
        return None


def _read_bin(path: Path) -> np.ndarray:
    """Read numpy float32 array from binary file."""
    if not path.exists():
        raise FileNotFoundError(f"Bin file not found: {path}")
    return np.fromfile(path, dtype='<f')


def _atomic_write_bin(path: Path, arr: np.ndarray) -> None:
    """Atomically write numpy array to binary file (write to .tmp then rename)."""
    tmp = path.with_suffix('.bin.tmp')
    arr.astype('<f').tofile(tmp)
    tmp.rename(path)


def _append_to_bin(bin_path: Path, value: float) -> np.ndarray:
    """Append a float32 value to existing bin file.

    Returns the new array.
    """
    if bin_path.exists():
        arr = _read_bin(bin_path)
    else:
        arr = np.array([], dtype='<f')
    new_arr = np.append(arr, np.float32(value))
    _atomic_write_bin(bin_path, new_arr)
    return new_arr


def _get_adjclose_scale(stock_dir: Path) -> Optional[float]:
    """Get the adjclose scaling constant for a stock.

    adjclose = close * adjclose_scale (constant per stock).
    Derived from: adjclose_scale = adjclose[1] / close[1].
    """
    adjclose_path = stock_dir / 'adjclose.day.bin'
    close_path = stock_dir / 'close.day.bin'
    if not adjclose_path.exists() or not close_path.exists():
        return None
    try:
        adjclose = _read_bin(adjclose_path)
        close = _read_bin(close_path)
        # Index 0 is start_index, index 1 is first data point
        if adjclose.size > 1 and close.size > 1 and close[1] != 0:
            return float(adjclose[1] / close[1])
    except Exception:
        pass
    return None


def _update_stock_features(
    stock_code: str,
    data_dir: Path,
    ohlcv: dict,
    prev_close_stored: Optional[float],
    factor: float,
    adjclose_scale: Optional[float],
) -> dict:
    """Update all feature bin files for a single stock on a single day.

    Args:
        stock_code: e.g., "SH600000"
        data_dir: Qlib data directory
        ohlcv: dict with open, close, high, low, volume, amount (raw values from akshare)
        prev_close_stored: previous day's stored close (for change calculation)
        factor: adjustment factor (复权因子), read from existing factor.day.bin
        adjclose_scale: constant adjclose[1]/close[1] for this stock

    Returns:
        dict with prev_close_stored for next iteration
    """
    code_lower = stock_code.lower()
    stock_dir = data_dir / 'features' / code_lower

    if not stock_dir.exists():
        logger.warning("Feature directory not found for %s, skipping", stock_code)
        return {'prev_close_stored': None}

    # Price features: stored = raw * factor
    close_stored = ohlcv['close'] * factor
    open_stored = ohlcv['open'] * factor
    high_stored = ohlcv['high'] * factor
    low_stored = ohlcv['low'] * factor

    # Volume: stored = raw / factor (inverse of prices!)
    volume_stored = ohlcv['volume'] / factor if factor != 0 else 0.0

    # Amount: stored = raw / 1000 (千元单位)
    amount_stored = ohlcv['amount'] / 1000.0

    # VWAP: stored = raw_vwap * factor
    # raw_vwap = raw_amount / (raw_volume * 100)  (akshare volume is in 手, 100股/手)
    raw_vwap = ohlcv['amount'] / (ohlcv['volume'] * 100) if ohlcv['volume'] != 0 else 0.0
    vwap_stored = raw_vwap * factor

    # Change: stored close return
    if prev_close_stored is not None and prev_close_stored != 0:
        change_stored = (close_stored / prev_close_stored) - 1.0
    else:
        change_stored = 0.0

    # Factor: reuse latest (only changes on ex-dividend dates)
    factor_stored = factor

    # Adjclose: close * adjclose_scale (constant scale per stock)
    if adjclose_scale is not None:
        adjclose_stored = close_stored * adjclose_scale
    else:
        adjclose_stored = 0.0

    features = {
        'open.day.bin': open_stored,
        'high.day.bin': high_stored,
        'low.day.bin': low_stored,
        'close.day.bin': close_stored,
        'volume.day.bin': volume_stored,
        'amount.day.bin': amount_stored,
        'vwap.day.bin': vwap_stored,
        'change.day.bin': change_stored,
        'factor.day.bin': factor_stored,
        'adjclose.day.bin': adjclose_stored,
    }

    for fname, value in features.items():
        bin_path = stock_dir / fname
        try:
            _append_to_bin(bin_path, value)
        except FileNotFoundError:
            logger.debug("Feature bin not found, skipping: %s", bin_path)
        except Exception as e:
            logger.warning("Failed to update %s for %s: %s", fname, stock_code, e)

    return {'prev_close_stored': close_stored}


def _update_calendar(data_dir: Path, date: str) -> None:
    """Append date to calendars/day.txt."""
    day_file = data_dir / 'calendars' / 'day.txt'
    with day_file.open('a', encoding='utf-8') as f:
        f.write(date + '\n')


def update_daily(
    data_dir: Optional[str | Path] = None,
    stock_list: Optional[list[str]] = None,
) -> dict:
    """Main entry point: fetch latest day data from akshare and append to Qlib bins.

    Args:
        data_dir: Qlib data directory (default: DEFAULT_DATA_DIR)
        stock_list: List of stock codes to update (default: all from instruments/all.txt)

    Returns:
        dict with keys: dates, stocks_updated, stocks_skipped, errors
    """
    dirp = _data_dir(data_dir)

    missing_dates = _get_missing_dates(dirp)
    if not missing_dates:
        return {'dates': [], 'stocks_updated': 0, 'stocks_skipped': 0, 'errors': []}

    if stock_list is None:
        stock_list = _get_stock_list(dirp)

    logger.info("Updating %d stocks for %d dates: %s ... %s",
                len(stock_list), len(missing_dates), missing_dates[0], missing_dates[-1])

    errors = []
    stocks_updated = 0
    stocks_skipped = 0

    # Cache per-stock state across dates
    prev_closes: dict[str, Optional[float]] = {}

    for date in missing_dates:
        logger.info("Processing date %s (%d stocks)...", date, len(stock_list))

        for i, stock_code in enumerate(stock_list):
            if (i + 1) % 500 == 0:
                logger.info("  [%s] Progress: %d/%d stocks", date, i + 1, len(stock_list))

            # Rate limiting: 0.3s between akshare calls
            time.sleep(0.3)

            ohlcv = _fetch_daily_data(stock_code, date)
            if ohlcv is None:
                stocks_skipped += 1
                continue

            stock_dir = dirp / 'features' / stock_code.lower()
            if not stock_dir.exists():
                stocks_skipped += 1
                continue

            # Read factor from existing bin (last value)
            factor = 1.0
            factor_path = stock_dir / 'factor.day.bin'
            if factor_path.exists():
                try:
                    arr = _read_bin(factor_path)
                    if arr.size > 1:
                        factor = float(arr[-1])
                except Exception:
                    pass

            # Read adjclose scale (constant per stock)
            adjclose_scale = _get_adjclose_scale(stock_dir)

            prev_close = prev_closes.get(stock_code)

            result = _update_stock_features(
                stock_code, dirp, ohlcv, prev_close, factor, adjclose_scale
            )
            prev_closes[stock_code] = result['prev_close_stored']

            stocks_updated += 1

        # Append this date to calendar after processing all stocks
        try:
            _update_calendar(dirp, date)
        except Exception as e:
            logger.error("Failed to update calendar for %s: %s", date, e)
            errors.append(f"calendar:{date}:{e}")

    logger.info(
        "Daily update complete. Updated: %d, Skipped: %d, Errors: %d",
        stocks_updated, stocks_skipped, len(errors)
    )

    return {
        'dates': missing_dates,
        'stocks_updated': stocks_updated,
        'stocks_skipped': stocks_skipped,
        'errors': errors,
    }


__all__ = ['update_daily', '_get_missing_dates', '_get_stock_list', '_fetch_daily_data']