#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional

# Ensure src import path for local package imports
import os
import sys as _sys
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / 'src'
_sys.path.insert(0, str(SRC_DIR))

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("update-daily")

from big_a.data.daily_updater import (
    update_daily,
    _get_missing_dates,
    _get_stock_list,
    _data_dir,
)

import typer

app = typer.Typer()


@app.command()
def update(
    data_dir: Path = typer.Option(None, "--data-dir", help="Data directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show planned updates without applying"),
):
    """Incrementally update daily CN stock data from akshare."""
    dirp = _data_dir(str(data_dir) if data_dir else None)

    if dry_run:
        print("DRY-RUN: Daily update plan")

        try:
            missing_dates = _get_missing_dates(dirp)
            stock_list = _get_stock_list(dirp)

            num_dates = len(missing_dates)
            num_stocks = len(stock_list)

            if missing_dates:
                date_list_str = ', '.join(missing_dates[:5])
                if num_dates > 5:
                    date_list_str += f', ... ({num_dates - 5} more)'
                print(f"Would update {num_stocks} stocks for {num_dates} dates: [{date_list_str}]")
            else:
                print("Calendar is up to date, no missing dates.")
        except Exception as e:
            print(f"Could not determine update plan: {e}")
        return

    logger.info("Starting daily data update.")
    result = update_daily(str(dirp), None)
    logger.info(
        "Daily update complete. Updated: %d, Skipped: %d, Errors: %d",
        result['stocks_updated'], result['stocks_skipped'], len(result['errors'])
    )


if __name__ == "__main__":
    app()
