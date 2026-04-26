#!/usr/bin/env python3
"""One-click end-to-end LightGBM pipeline: data validation → train → predict → backtest → analysis."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger

app = typer.Typer(help="End-to-end LightGBM pipeline for A-share trading signals")


@app.command()
def main(
    output_dir: Path = typer.Option("output", "--output-dir", help="Output directory"),
    model_config: str = typer.Option(
        "configs/model/lightgbm.yaml", "--model-config", help="Model config path"
    ),
    data_config: str = typer.Option(
        "configs/data/handler_alpha158.yaml", "--data-config", help="Dataset config path"
    ),
    backtest_config: str = typer.Option(
        "configs/backtest/topk_csi300.yaml", "--backtest-config", help="Backtest config path"
    ),
    skip_train: bool = typer.Option(
        False, "--skip-train", help="Reuse existing model instead of retraining"
    ),
):
    """Run the complete LightGBM pipeline from data validation to analysis report."""
    try:
        from big_a.qlib_config import DEFAULT_DATA_DIR, init_qlib

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "lightgbm_model.pkl"
        preds_path = output_dir / "lightgbm_predictions.parquet"
        report_path = output_dir / "backtest_report.parquet"
        analysis_dir = output_dir / "analysis"

        # ── Step 1: Data validation ───────────────────────────────────────
        logger.info("Step 1/6: Validating data")
        if not DEFAULT_DATA_DIR.exists():
            logger.error(
                "Qlib data not found at {}. "
                "Download: python scripts/update_data.py update",
                DEFAULT_DATA_DIR,
            )
            raise typer.Exit(1)
        logger.info("Data directory found: {}", DEFAULT_DATA_DIR)

        # ── Step 2: Initialize Qlib ───────────────────────────────────────
        logger.info("Step 2/6: Initializing Qlib")
        init_qlib()

        # ── Step 3: Train LightGBM ────────────────────────────────────────
        if skip_train and model_path.exists():
            logger.info("Step 3/6: Skipping training — loading existing model from {}", model_path)
            from big_a.models.lightgbm_model import load_model, create_dataset
            from big_a.config import load_config

            config = load_config(model_config, data_config)
            dataset = create_dataset(config)
            model = load_model(model_path)
        else:
            logger.info("Step 3/6: Training LightGBM model")
            from big_a.models.lightgbm_model import train as train_lgb, save_model

            model, dataset, _config = train_lgb(
                model_config_path=model_config,
                data_config_path=data_config,
            )
            save_model(model, model_path)
            logger.info("Model saved to {}", model_path)

        # ── Step 4: Generate predictions ──────────────────────────────────
        logger.info("Step 4/6: Generating predictions on test segment")
        from big_a.models.lightgbm_model import predict_to_dataframe

        predictions = predict_to_dataframe(model, dataset, segment="test")
        if predictions.empty:
            logger.error("Predictions are empty — check data config and date ranges")
            raise typer.Exit(1)
        predictions.to_parquet(preds_path)
        logger.info("Predictions saved to {} ({} rows)", preds_path, len(predictions))

        # ── Step 5: Run backtest ──────────────────────────────────────────
        logger.info("Step 5/6: Running backtest")
        from big_a.backtest.engine import load_backtest_config, run_backtest

        bt_config = load_backtest_config(backtest_config)
        report, _positions = run_backtest(predictions, bt_config)
        report.to_parquet(report_path)
        logger.info("Backtest report saved to {} ({} days)", report_path, len(report))

        # ── Step 6: Generate analysis report ──────────────────────────────
        logger.info("Step 6/6: Generating analysis report")
        from big_a.backtest.analysis import analyze_backtest, generate_report, _format_summary

        analysis = analyze_backtest(report)
        analysis["_report_df"] = report
        generate_report(analysis, analysis_dir)

        # ── Final summary ─────────────────────────────────────────────────
        print()
        print(_format_summary(analysis))
        print()
        logger.info("Pipeline complete. Output files:")
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                logger.info("  {}", p.relative_to(output_dir.parent))

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
