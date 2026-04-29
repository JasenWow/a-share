"""Prefect daily pipeline for data updates."""

from datetime import datetime
from pathlib import Path

try:
    from loguru import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("flows")

from prefect import flow, task

from big_a.simulation.config import load_simulation_config
from big_a.simulation.engine import SimulationEngine
from big_a.simulation.storage import SimulationStorage
from big_a.broker.in_memory import InMemoryBroker
from big_a.simulation.types import StockSignal, SignalStrength, SignalSource


@task
def update_market_data() -> None:
    """Update market data incrementally."""
    from big_a.data.updater import update_incremental

    logger.info("Starting market data update...")
    update_incremental()
    logger.info("Market data update completed successfully.")


@task
def update_sector_data() -> None:
    """Update sector classification data."""
    from big_a.data.sector import refresh_sector_data

    logger.info("Starting sector data update...")
    refresh_sector_data()
    logger.info("Sector data update completed successfully.")


@task
def run_simulation_daily() -> dict:
    """Execute one simulation trading day with mock signals."""
    config = load_simulation_config("configs/simulation/default.yaml")
    storage = SimulationStorage(
        base_dir=config.storage_base_dir,
        trades_dir=config.storage_trades_dir,
        decisions_dir=config.storage_decisions_dir,
        snapshots_dir=config.storage_snapshots_dir,
    )

    # Load latest state
    latest = storage.load_latest_snapshot()
    broker = InMemoryBroker(
        initial_cash=config.initial_capital,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_commission=config.min_commission,
        limit_threshold=config.limit_threshold,
    )

    if latest:
        broker._cash = latest.cash
        for code, pos in latest.positions.items():
            broker._positions[code] = pos
            broker._prices[code] = pos.current_price

    engine = SimulationEngine(config=config, broker=broker)
    engine.initialize()

    # Generate mock signals (placeholder - real signals from models later)
    mock_signals = [
        StockSignal(stock_code="600519.SH", score=0.8, signal=SignalStrength.BUY, source=SignalSource.fused),
        StockSignal(stock_code="000858.SZ", score=0.6, signal=SignalStrength.BUY, source=SignalSource.fused),
        StockSignal(stock_code="600036.SH", score=0.4, signal=SignalStrength.BUY, source=SignalSource.fused),
    ]

    mock_prices = {s.stock_code: 100.0 for s in mock_signals}

    portfolio = engine.run_daily(
        trading_date=datetime.now().strftime("%Y-%m-%d"),
        signals=mock_signals,
        prices=mock_prices,
    )

    # Save state
    storage.save_snapshot(portfolio, datetime.now().strftime("%Y-%m-%d"))

    return {"total_value": portfolio.total_value, "cash": portfolio.cash}


@flow
def daily_pipeline() -> None:
    """Daily data update pipeline that runs market and sector updates sequentially."""
    logger.info("Starting daily data pipeline...")

    try:
        update_market_data()
        logger.info("Market data update task succeeded.")
    except Exception as e:
        logger.error(f"Market data update task failed: {e}")

    try:
        update_sector_data()
        logger.info("Sector data update task succeeded.")
    except Exception as e:
        logger.error(f"Sector data update task failed: {e}")

    logger.info("Daily data pipeline completed.")


@flow(name="simulation_trading_pipeline")
def simulation_pipeline() -> None:
    """AI-driven simulated trading pipeline - runs after market close."""
    run_simulation_daily()


if __name__ == "__main__":
    # Deploy the flow with a cron schedule (Mon-Fri 19:00 - after market close)
    daily_pipeline.serve(cron="0 19 * * 1-5")