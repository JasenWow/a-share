#!/usr/bin/env python3
"""One-click end-to-end pipeline: LightGBM and Kronos prediction → backtest → analysis."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger

app = typer.Typer(help="End-to-end pipeline: LightGBM + Kronos model comparison")


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
    skip_kronos: bool = typer.Option(False, "--skip-kronos", help="Skip Kronos prediction and comparison steps"),
    kronos_config: str = typer.Option("configs/model/kronos.yaml", "--kronos-config", help="Kronos model config path"),
):
    """Run the complete LightGBM + Kronos pipeline from data validation to analysis report."""
    try:
        from big_a.qlib_config import DEFAULT_DATA_DIR, init_qlib

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / "lightgbm_model.pkl"
        preds_path = output_dir / "lightgbm_predictions.parquet"
        report_path = output_dir / "backtest_report.parquet"
        analysis_dir = output_dir / "analysis"
        kronos_preds_path = output_dir / "kronos_predictions.parquet"
        kronos_report_path = output_dir / "kronos_backtest_report.parquet"
        kronos_analysis_dir = output_dir / "kronos_analysis"

        # ── Step 1: Data validation ───────────────────────────────────────
        logger.info("Step 1/9: Validating data")
        if not DEFAULT_DATA_DIR.exists():
            logger.error(
                "Qlib data not found at {}. "
                "Download: python scripts/update_data.py update",
                DEFAULT_DATA_DIR,
            )
            raise typer.Exit(1)
        logger.info("Data directory found: {}", DEFAULT_DATA_DIR)

        # ── Step 2: Initialize Qlib ───────────────────────────────────────
        logger.info("Step 2/9: Initializing Qlib")
        init_qlib()

        # ── Step 3: Train LightGBM ────────────────────────────────────────
        if skip_train and model_path.exists():
            logger.info("Step 3/9: Skipping training — loading existing model from {}", model_path)
            from big_a.models.lightgbm_model import load_model, create_dataset
            from big_a.config import load_config

            config = load_config(model_config, data_config)
            dataset = create_dataset(config)
            model = load_model(model_path)
        else:
            logger.info("Step 3/9: Training LightGBM model")
            from big_a.models.lightgbm_model import train as train_lgb, save_model

            model, dataset, _config = train_lgb(
                model_config_path=model_config,
                data_config_path=data_config,
            )
            save_model(model, model_path)
            logger.info("Model saved to {}", model_path)

        # ── Step 4: Generate predictions ──────────────────────────────────
        logger.info("Step 4/9: Generating predictions on test segment")
        from big_a.models.lightgbm_model import predict_to_dataframe

        predictions = predict_to_dataframe(model, dataset, segment="test")
        if predictions.empty:
            logger.error("Predictions are empty — check data config and date ranges")
            raise typer.Exit(1)
        predictions.to_parquet(preds_path)
        logger.info("Predictions saved to {} ({} rows)", preds_path, len(predictions))

        # ── Step 5: Run backtest ──────────────────────────────────────────
        logger.info("Step 5/9: Running backtest")
        from big_a.backtest.engine import load_backtest_config, run_backtest

        bt_config = load_backtest_config(backtest_config)
        report, _positions = run_backtest(predictions, bt_config)
        report.to_parquet(report_path)
        logger.info("Backtest report saved to {} ({} days)", report_path, len(report))

        # ── Step 6: Generate analysis report ──────────────────────────────
        logger.info("Step 6/9: Generating analysis report")
        from big_a.backtest.analysis import analyze_backtest, generate_report, _format_summary

        analysis = analyze_backtest(report)
        analysis["_report_df"] = report
        generate_report(analysis, analysis_dir)

        # ── Final summary ─────────────────────────────────────────────────
        print()
        print(_format_summary(analysis))
        print()

        # ── Step 7: Rolling Kronos signal generation ────────────────────────
        if not skip_kronos:
            logger.info("Step 7/9: Generating rolling Kronos signals")
            from big_a.models.kronos import KronosSignalGenerator
            from big_a.config import load_config as load_kronos_config
            from qlib.data import D
            import tqdm
            import numpy as np
            import pandas as pd

            # Load Kronos config
            kronos_cfg = load_kronos_config(kronos_config).get("kronos", {})

            # Initialize generator (CPU by default)
            gen = KronosSignalGenerator(
                tokenizer_id=kronos_cfg.get("tokenizer_id", "NeoQuasar/Kronos-Tokenizer-base"),
                model_id=kronos_cfg.get("model_id", "NeoQuasar/Kronos-base"),
                device=kronos_cfg.get("device", "cpu"),
                lookback=kronos_cfg.get("lookback", 90),
                pred_len=kronos_cfg.get("pred_len", 10),
                max_context=kronos_cfg.get("max_context", 512),
                signal_mode=kronos_cfg.get("signal_mode", "mean"),
            )
            gen.load_model()

            # Derive test period and instruments from LightGBM predictions
            test_dates = sorted(predictions.index.get_level_values("datetime").unique())
            instruments = sorted(predictions.index.get_level_values("instrument").unique())

            # Fetch all OHLCV data ONCE with lookback buffer
            lookback_buffer = pd.DateOffset(months=6)
            data_start = str((test_dates[0] - lookback_buffer).date())
            data_end = str(test_dates[-1].date())
            all_data = gen.load_data(instruments, data_start, data_end)

            # Rolling loop: for each test date, generate signals for all stocks
            all_signals = []
            for date in tqdm.tqdm(test_dates, desc="Kronos rolling"):
                rows = []
                for instrument, stock_df in all_data.groupby(level=0):
                    stock_df = stock_df.droplevel(0).sort_index()
                    stock_df = stock_df[stock_df.index <= date]
                    if len(stock_df) < gen.lookback:
                        continue
                    tail = stock_df.iloc[-gen.lookback:]
                    if tail.isnull().any().any():
                        continue
                    pred_df = gen.predict(tail)
                    if pred_df is None or pred_df.empty:
                        continue
                    last_close = tail["close"].iloc[-1]
                    pred_closes = pred_df["close"].values
                    if gen.signal_mode == "last":
                        score = float(pred_closes[-1] - last_close)
                    else:
                        score = float(np.mean(pred_closes) - last_close)
                    rows.append((date, instrument, score))
                if rows:
                    all_signals.extend(rows)

            # Build predictions DataFrame
            if all_signals:
                signal_df = pd.DataFrame(all_signals, columns=["datetime", "instrument", "score"])
                kronos_predictions = signal_df.set_index(["datetime", "instrument"])[["score"]]
                kronos_predictions.index.names = ["datetime", "instrument"]
            else:
                kronos_predictions = pd.DataFrame(
                    columns=["score"],
                    index=pd.MultiIndex.from_tuples([], names=["datetime", "instrument"]),
                )

            if kronos_predictions.empty:
                logger.error("No Kronos signals generated")
                raise typer.Exit(1)
            kronos_predictions.to_parquet(kronos_preds_path)
            logger.info("Kronos predictions saved to {} ({} rows)", kronos_preds_path, len(kronos_predictions))

        # ── Step 8: Kronos backtest ──────────────────────────────────────────
        if not skip_kronos:
            logger.info("Step 8/9: Running Kronos backtest")
            from big_a.backtest.engine import run_backtest

            kronos_report, _kronos_positions = run_backtest(kronos_predictions, bt_config)
            kronos_report.to_parquet(kronos_report_path)
            logger.info("Kronos backtest saved to {} ({} days)", kronos_report_path, len(kronos_report))

        # ── Step 9: Model comparison ────────────────────────────────────────
        if not skip_kronos:
            logger.info("Step 9/9: Comparing models")
            from big_a.backtest.evaluation import compare_models

            # Compute actual returns via Qlib
            label_expr = ["Ref($close, -2) / Ref($close, -1) - 1"]
            actual_returns = D.features(
                instruments,
                fields=label_expr,
                start_time=str(test_dates[0].date()) if hasattr(test_dates[0], 'date') else str(test_dates[0]),
                end_time=str(test_dates[-1].date()) if hasattr(test_dates[-1], 'date') else str(test_dates[-1]),
            )
            actual_returns.columns = ["score"]

            comparison = compare_models(
                {"LightGBM": predictions, "Kronos": kronos_predictions},
                actual_returns,
            )
            print()
            print("── Model Comparison ──")
            print(comparison.to_string())
            print()

            # Generate Kronos analysis report
            kronos_analysis = analyze_backtest(kronos_report)
            kronos_analysis["_report_df"] = kronos_report
            generate_report(kronos_analysis, kronos_analysis_dir)
            print("── Kronos Backtest Summary ──")
            print(_format_summary(kronos_analysis))
            print()

        # ── Final summary ───────────────────────────────────────────────────
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
