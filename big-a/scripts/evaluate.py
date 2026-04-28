#!/usr/bin/env python3
"""CLI for model evaluation: compare Kronos vs LightGBM predictions."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger

from big_a.experiment import (
    experiment_context,
    log_metrics,
    log_params,
    make_experiment_name,
)

app = typer.Typer()


@app.command()
def main(
    kronos_predictions: Path = typer.Option(..., help="Path to Kronos predictions CSV"),
    lightgbm_predictions: Path = typer.Option(..., help="Path to LightGBM predictions CSV"),
    actual_returns: Path = typer.Option(..., help="Path to actual returns CSV"),
    output_dir: Path = typer.Option("output/evaluation", help="Directory for output plots and tables"),
    no_track: bool = typer.Option(False, "--no-track", help="Disable experiment tracking"),
):
    """Compare Kronos and LightGBM model predictions."""
    import pandas as pd

    from big_a.backtest.evaluation import (
        calc_ic,
        calc_rank_ic,
        compare_models,
        plot_ic_series,
        plot_model_comparison,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    kronos_df = pd.read_csv(kronos_predictions, index_col=[0, 1])
    lgbm_df = pd.read_csv(lightgbm_predictions, index_col=[0, 1])
    actual_s = pd.read_csv(actual_returns, index_col=[0, 1]).squeeze("columns")

    predictions = {"Kronos": kronos_df, "LightGBM": lgbm_df}

    comparison = compare_models(predictions, actual_s)
    logger.info("\n{}", comparison.to_string())
    comparison.to_csv(output_dir / "comparison.csv")

    for name, pred_df in predictions.items():
        ic = calc_ic(pred_df, actual_s)
        rank_ic = calc_rank_ic(pred_df, actual_s)
        if len(ic) > 0:
            plot_ic_series(ic, title=f"{name} IC Series", save_path=str(output_dir / f"{name.lower()}_ic.png"))
        if len(rank_ic) > 0:
            plot_ic_series(rank_ic, title=f"{name} Rank IC Series", save_path=str(output_dir / f"{name.lower()}_rank_ic.png"))

    for metric in ["mean_ic", "mean_rank_ic", "icir"]:
        plot_model_comparison(comparison, metric=metric, save_path=str(output_dir / f"compare_{metric}.png"))

    logger.info("Evaluation complete. Results saved to {}", output_dir)

    if not no_track:
        exp_name = make_experiment_name("evaluate")
        with experiment_context(exp_name) as _recorder:
            log_params({"kronos": str(kronos_predictions), "lightgbm": str(lightgbm_predictions)})
            for _, row in comparison.iterrows():
                model_name = row.name
                for col in comparison.columns:
                    val = row[col]
                    if pd.notna(val):
                        log_metrics({f"{model_name}_{col}": float(val)})


if __name__ == "__main__":
    app()
