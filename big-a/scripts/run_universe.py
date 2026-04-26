#!/usr/bin/env python3
"""CLI tool for building and inspecting stock universe."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger
from typing import Optional

app = typer.Typer(help="Build and inspect stock universe")


def load_universe_config(config_path: str) -> dict:
    """Load universe configuration from YAML file."""
    from big_a.config import load_config
    return load_config(config_path).get("universe", {})


def get_sector_distribution(universe_cfg: dict, watchlist_codes: list) -> dict:
    """Get sector distribution for the universe."""
    from big_a.data.sector import get_stock_sector

    sector_dist = {}
    for code in watchlist_codes:
        try:
            sector = get_stock_sector(code)
            if sector:
                sector_dist[sector] = sector_dist.get(sector, 0) + 1
        except Exception as e:
            logger.warning(f"Failed to get sector for {code}: {e}")

    return sector_dist


def print_statistics(
    universe_cfg: dict,
    watchlist_codes: list,
    sector_dist: dict,
    dry_run: bool = False
):
    """Print universe statistics to stdout."""
    print("Universe Statistics")
    print("==================")
    print(f"Total stocks: {len(watchlist_codes)}")

    watchlist_path = universe_cfg.get("watchlist", "configs/watchlist.yaml")
    print(f"Watchlist stocks: {len(watchlist_codes)}")

    base_pool = universe_cfg.get("base_pool", "csi300")
    pool_sizes = {
        "csi300": 300,
        "csi500": 500,
        "csi800": 800,
        "csi1000": 1000,
        "all": 5000,
    }
    print(f"Base pool: {base_pool} ({pool_sizes.get(base_pool, '?')} stocks)")

    sector_rotation = universe_cfg.get("sector_rotation", {})
    if sector_rotation.get("enabled", False):
        lookback = sector_rotation.get("lookback_days", 20)
        top_k = sector_rotation.get("top_k_sectors", 5)
        print(f"Sector rotation: enabled (top {top_k} sectors, {lookback}-day lookback)")
    else:
        print("Sector rotation: disabled")

    print()

    print("Sector Distribution:")
    sorted_sectors = sorted(sector_dist.items(), key=lambda x: x[1], reverse=True)
    for sector, count in sorted_sectors:
        print(f"- {sector}: {count} stocks")

    if not sorted_sectors:
        print("(No sector data available)")

    print()

    if dry_run:
        print("[DRY RUN MODE] - No actual universe building was performed.")
        print("Config loaded and statistics shown without calling Qlib.")


@app.command()
def run(
    config: str = typer.Option(
        "configs/data/universe.yaml",
        "--config",
        "-c",
        help="Path to universe config file"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show statistics without executing full pipeline"
    ),
    show_sectors: bool = typer.Option(
        False,
        "--show-sectors",
        help="Show sector distribution of the universe"
    ),
):
    """Build universe and output statistics to stdout."""
    try:
        logger.info(f"Loading universe config from: {config}")
        universe_cfg = load_universe_config(config)

        watchlist_path = universe_cfg.get("watchlist", "configs/watchlist.yaml")
        from big_a.data.screener import load_watchlist
        watchlist_codes = load_watchlist(watchlist_path)

        sector_dist = {}
        if show_sectors:
            sector_dist = get_sector_distribution(universe_cfg, watchlist_codes)

        print_statistics(universe_cfg, watchlist_codes, sector_dist, dry_run)

        if not dry_run:
            from big_a.data.screener import build_universe
            logger.info("Building universe...")
            universe = build_universe(config)
            print(f"Final universe: {len(universe)} instruments")
            logger.success("Universe built successfully")

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to build universe: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
