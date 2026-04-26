#!/usr/bin/env python3
"""Small-capital real trading backtest with Kronos + RealTradingStrategy."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger

from big_a.experiment import start_experiment, log_params, log_metrics, log_artifact, end_experiment

app = typer.Typer(help="Small-capital real trading backtest with Kronos + risk controls")


@app.command()
def main(
    output_dir: Path = typer.Option("output/real_trading", "--output-dir", help="Output directory"),
    backtest_config: str = typer.Option("configs/backtest/real_trading.yaml", "--config", help="Backtest config path"),
    kronos_config: str = typer.Option("configs/model/kronos.yaml", "--kronos-config", help="Kronos model config path"),
    skip_kronos_download: bool = typer.Option(False, "--skip-kronos-download", help="Use cached Kronos model"),
):
    """Run small-capital real trading backtest with Kronos signals and risk controls."""
    try:
        from big_a.qlib_config import DEFAULT_DATA_DIR, init_qlib

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start_experiment("real_trading_backtest")

        kronos_preds_path = output_dir / "kronos_predictions.parquet"
        report_path = output_dir / "backtest_report.parquet"
        analysis_dir = output_dir / "analysis"

        # ── Step 1: Validate data ───────────────────────────────────────────────
        logger.info("Step 1/6: Validating data")
        if not DEFAULT_DATA_DIR.exists():
            logger.error(
                "Qlib data not found at {}. "
                "Download: python scripts/update_data.py update",
                DEFAULT_DATA_DIR,
            )
            raise typer.Exit(1)
        logger.info("Data directory found: {}", DEFAULT_DATA_DIR)

        # ── Step 2: Initialize Qlib ───────────────────────────────────────────────
        logger.info("Step 2/6: Initializing Qlib")
        init_qlib()

        # ── Step 3: Load backtest config ───────────────────────────────────────────
        logger.info("Step 3/6: Loading backtest config")
        from big_a.backtest.engine import load_backtest_config

        config = load_backtest_config(backtest_config)
        logger.info("Backtest config loaded from {}", backtest_config)

        log_params({
            "backtest_config": backtest_config,
            "kronos_config": kronos_config,
            "skip_kronos_download": skip_kronos_download,
        })

        # ── Step 4: Generate Kronos signals ─────────────────────────────────────────
        logger.info("Step 4/6: Generating rolling Kronos signals")
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
        gen.load_model(local_files_only=skip_kronos_download)

        # Get test period from backtest config
        bt_cfg = config.get("backtest", {})
        start_time = bt_cfg.get("start_time", "2022-01-01")
        end_time = bt_cfg.get("end_time", "2024-12-31")

        # Get instruments from market (CSI 300)
        instruments = D.list_instruments(instruments=D.instruments("csi300"), as_list=True)
        logger.info("Loaded {} instruments from CSI300", len(instruments))

        # Fetch all OHLCV data ONCE with lookback buffer
        lookback_buffer = pd.DateOffset(months=6)
        data_start = str((pd.to_datetime(start_time) - lookback_buffer).date())
        data_end = str(pd.to_datetime(end_time).date())
        all_data = gen.load_data(instruments, data_start, data_end)

        # Get trading dates for the test period
        trading_days = D.calendar(start_time=start_time, end_time=end_time)

        # Rolling loop: for each test date, generate signals for all stocks
        all_signals = []
        for date in tqdm.tqdm(trading_days, desc="Kronos rolling"):
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

        log_metrics({"prediction_count": float(len(kronos_predictions))})

        # ── Step 5: Create RealTradingStrategy and run backtest ──────────────────────
        logger.info("Step 5/6: Creating RealTradingStrategy and running backtest")
        from big_a.strategy.real_trading import RealTradingStrategy
        from big_a.backtest.engine import run_backtest_with_strategy

        # Merge risk controls from config into strategy kwargs
        risk_controls = config.get("risk_controls", {})
        strat_kwargs = {**config.get("strategy", {}).get("kwargs", {}), **risk_controls}
        # Remove 'signal' key since it's passed separately
        strat_kwargs.pop("signal", None)

        strategy = RealTradingStrategy(signal=kronos_predictions, **strat_kwargs)

        report, _positions = run_backtest_with_strategy(strategy, config)
        report.to_parquet(report_path)
        logger.info("Backtest report saved to {} ({} days)", report_path, len(report))

        # Persist positions (pickle for full Qlib Position objects, parquet for flat analysis)
        import pickle
        positions_pkl_path = output_dir / "positions.pkl"
        with open(positions_pkl_path, "wb") as f:
            pickle.dump(_positions, f)
        logger.info("Positions saved to {}", positions_pkl_path)

        def _flatten_positions(positions: dict) -> pd.DataFrame:
            """Extract holding weights from Qlib Position objects into a DataFrame."""
            rows = []
            for date, pos in positions.items():
                for stock, weight in pos.stock_weight.items():
                    rows.append({"datetime": date, "instrument": stock, "weight": weight})
            return pd.DataFrame(rows)

        if _positions:
            positions_flat_path = output_dir / "positions_flat.parquet"
            _flat_df = _flatten_positions(_positions)
            if not _flat_df.empty:
                _flat_df.to_parquet(positions_flat_path)
                logger.info("Flat positions saved to {} ({} rows)", positions_flat_path, len(_flat_df))

        # ── Step 6: Generate analysis report ─────────────────────────────────────────
        logger.info("Step 6/6: Generating analysis report")
        from big_a.backtest.analysis import analyze_backtest, generate_report, _format_summary

        analysis = analyze_backtest(report)
        analysis["_report_df"] = report
        generate_report(analysis, analysis_dir)

        log_metrics({
            "sharpe_ratio": analysis.get("sharpe_ratio", 0),
            "annualized_return": analysis.get("annualized_return", 0),
            "max_drawdown": analysis.get("max_drawdown", 0),
            "information_ratio": analysis.get("information_ratio", 0),
            "excess_return": analysis.get("excess_return", 0),
        })
        log_artifact(report_path)

        # ── Final summary ───────────────────────────────────────────────────────────
        print()
        print(_format_summary(analysis))
        print()

        logger.info("Backtest complete. Output files:")
        for p in sorted(output_dir.rglob("*")):
            if p.is_file():
                logger.info("  {}", p.relative_to(output_dir.parent))

        end_experiment("FINISHED")

    except typer.Exit:
        end_experiment("KILLED")
        raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        end_experiment("FAILED")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
