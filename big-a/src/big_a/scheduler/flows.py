"""Prefect daily pipeline for data updates."""

try:
    from loguru import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("flows")

from prefect import flow, task


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


if __name__ == "__main__":
    # Deploy the flow with a cron schedule (Mon-Fri 19:00 - after market close)
    daily_pipeline.serve(cron="0 19 * * 1-5")
