#!/usr/bin/env python3
"""Run rolling walk-forward backtests over sliding windows."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    model_type: str = typer.Option(
        "lightgbm", help="Model type: lightgbm or kronos"
    ),
    config_path: str = typer.Option(
        "configs/backtest/rolling_csi300.yaml", help="Rolling config YAML path"
    ),
    output: Path = typer.Option(
        "output/rolling_report.parquet", help="Path to save aggregated report"
    ),
    summary_output: Path = typer.Option(
        "output/rolling_summary.parquet", help="Path to save per-window summary"
    ),
):
    """Run rolling walk-forward backtest over sliding time windows."""
    from big_a.backtest.rolling import run_rolling

    results = run_rolling(model_type=model_type, config_path=config_path)

    summary_df = results["summary_df"]
    combined_report = results["combined_report"]

    output.parent.mkdir(parents=True, exist_ok=True)

    if combined_report is not None:
        combined_report.to_parquet(output)
        logger.info(f"Combined report saved to {output} ({len(combined_report)} rows)")

    summary_df.to_parquet(summary_output)
    logger.info(f"Summary saved to {summary_output} ({len(summary_df)} windows)")

    logger.info(f"\n{'='*60}")
    logger.info("Rolling Backtest Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Windows: {len(summary_df)}")
    logger.info(f"Mean IC: {results['mean_ic']:.4f}")
    logger.info(f"Mean Rank IC: {results['mean_rank_ic']:.4f}")
    logger.info(f"Mean ICIR: {results['mean_icir']:.4f}")
    logger.info(f"Mean Sharpe: {results['mean_sharpe']:.4f}")
    logger.info(f"Mean Max Drawdown: {results['mean_max_drawdown']:.4f}")
    logger.info(f"\nPer-window details:\n{summary_df.to_string()}")


if __name__ == "__main__":
    app()
