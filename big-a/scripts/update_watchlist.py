#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional

import os
import sys as _sys
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / 'src'
CONFIG_DIR = SCRIPT_DIR.parent / 'configs'
_sys.path.insert(0, str(SRC_DIR))

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("update-watchlist")

import yaml
import typer

from big_a.data.daily_updater import (
    update_daily,
    _get_missing_dates,
    _data_dir,
)

app = typer.Typer()


def _load_watchlist() -> list[str]:
    watchlist_path = CONFIG_DIR / 'watchlist.yaml'
    if not watchlist_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {watchlist_path}")

    with watchlist_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    watchlist = data.get('watchlist', {})
    if not watchlist:
        raise ValueError("No 'watchlist' key found in watchlist.yaml")

    codes = list(watchlist.keys())
    logger.info("Loaded %d stocks from watchlist: %s", len(codes), codes)
    return codes


@app.command()
def update(
    data_dir: Path = typer.Option(None, "--data-dir", help="Data directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show planned updates without applying"),
):
    watchlist_codes = _load_watchlist()

    dirp = _data_dir(str(data_dir) if data_dir else None)

    if dry_run:
        print("DRY-RUN: Watchlist update plan")

        try:
            missing_dates = _get_missing_dates(dirp)
            num_dates = len(missing_dates)

            if missing_dates:
                date_list_str = ', '.join(missing_dates[:5])
                if num_dates > 5:
                    date_list_str += f', ... ({num_dates - 5} more)'
                print(f"Would update {len(watchlist_codes)} watchlist stocks for {num_dates} dates: [{date_list_str}]")
                print(f"Watchlist stocks: {watchlist_codes}")
            else:
                print("Calendar is up to date, no missing dates.")
                print(f"Watchlist stocks: {watchlist_codes}")
        except Exception as e:
            print(f"Could not determine update plan: {e}")
        return

    logger.info("Starting watchlist data update for %d stocks.", len(watchlist_codes))
    result = update_daily(str(dirp), stock_list=watchlist_codes)
    logger.info(
        "Watchlist update complete. Updated: %d, Skipped: %d, Errors: %d",
        result['stocks_updated'], result['stocks_skipped'], len(result['errors'])
    )


if __name__ == "__main__":
    app()
