#!/usr/bin/env python3
"""Analyze a backtest report and generate performance charts."""
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
    report_path: Path = typer.Option(
        "output/backtest_report.parquet", help="Path to backtest report parquet"
    ),
    output_dir: Path = typer.Option(
        "output/analysis", help="Directory to save analysis results"
    ),
):
    import pandas as pd

    from big_a.backtest.analysis import analyze_backtest, generate_report

    logger.info("Loading report from {}", report_path)
    report_df = pd.read_parquet(report_path)

    analysis = analyze_backtest(report_df)
    analysis["_report_df"] = report_df

    generate_report(analysis, output_dir)

    print(generate_report.__module__)

    from big_a.backtest.analysis import _format_summary
    print(_format_summary(analysis))


if __name__ == "__main__":
    app()
