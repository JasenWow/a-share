#!/usr/bin/env python3
"""Run backtest on saved signal predictions."""
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
    signal_path: Path = typer.Option(
        "output/predictions.parquet", help="Path to signal predictions (parquet)"
    ),
    config_path: str = typer.Option(
        "configs/backtest/topk_csi300.yaml", help="Backtest config YAML path"
    ),
    output: Path = typer.Option(
        "output/backtest_report.parquet", help="Path to save backtest report"
    ),
    analysis_output: Path = typer.Option(
        "output/backtest_analysis.parquet", help="Path to save risk analysis"
    ),
):
    """Run backtest using saved signal predictions."""
    import pandas as pd

    from big_a.backtest.engine import compute_analysis, load_backtest_config, run_backtest
    from big_a.qlib_config import init_qlib

    init_qlib()

    logger.info(f"Loading signal from {signal_path}")
    signal = pd.read_parquet(signal_path)

    config = load_backtest_config(config_path)

    report, positions = run_backtest(signal, config)

    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(output)
    logger.info(f"Report saved to {output} ({len(report)} rows)")

    analysis = compute_analysis(report)
    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_parquet(analysis_output)
    logger.info(f"Analysis saved to {analysis_output}")

    logger.info(f"\n{analysis.to_string()}")


if __name__ == "__main__":
    app()
