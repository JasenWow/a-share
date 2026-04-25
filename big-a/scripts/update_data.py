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
    logger = logging.getLogger("update-data")

from big_a.data.updater import (
    get_last_update_date,
    update_incremental,
    verify_update,
)

import typer

app = typer.Typer()


def _data_dir_path(data_dir: Optional[Path]) -> Optional[str]:
    if data_dir is None:
        return None
    return str(data_dir)


@app.command()
def update(
    data_dir: Path = typer.Option(None, "--data-dir", help="Data directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show planned updates without applying"),
):
    """Incrementally update Qlib CN data for Big-A quant system."""
    if dry_run:
        print("DRY-RUN: Data update plan")
        try:
            last = get_last_update_date(str(data_dir) if data_dir else None)
            print(f"Current last update date: {last}")
        except Exception as e:
            print(f"Could not determine last update date: {e}")
        print("This dry-run would fetch the latest qlib_bin tarball from chenditc/investment_data releases and extract to the data directory.")
        print("It would then verify calendars and a sample instrument/feature presence.")
        return

    logger.info("Starting data update (incremental).")
    update_incremental(str(data_dir) if data_dir else None, None)
    ok = verify_update(str(data_dir) if data_dir else None)
    if ok:
        logger.info("Data update completed and verified.")
    else:
        logger.error("Data update verification failed.")


if __name__ == "__main__":
    app()
