#!/usr/bin/env python3
"""Unified CLI for the Big-A quant system.

Subcommands:
    train     — Train a LightGBM model
    predict   — Generate predictions (LightGBM or Kronos)
    backtest  — Run backtest on saved signals
    evaluate  — Compare models (IC, Rank IC, ICIR)
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger

from big_a.experiment import (
    experiment_context,
    log_hyperparams_from_config,
    log_data_version,
    log_model_artifact,
    make_experiment_name,
    log_metrics,
    log_params,
)

app = typer.Typer(help="Big-A quant system CLI")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    model_config: str = typer.Option(
        "configs/model/lightgbm.yaml", help="Model config path (relative to project root)"
    ),
    data_config: str = typer.Option(
        "configs/data/handler_alpha158.yaml", help="Dataset config path (relative to project root)"
    ),
    output: Path = typer.Option(
        "output/lightgbm_model.pkl", help="Path to save the trained model"
    ),
    no_track: bool = typer.Option(False, "--no-track", help="Disable experiment tracking"),
) -> None:
    """Train a LightGBM model on Alpha158 features."""
    from big_a.models.lightgbm_model import train as _train, save_model, predict_to_dataframe

    if no_track:
        model, dataset, config = _train(
            model_config_path=model_config,
            data_config_path=data_config,
        )
        save_model(model, output)
        preds = predict_to_dataframe(model, dataset, segment="test")
        logger.info(f"Test predictions shape: {preds.shape}")
        logger.info(f"Score stats:\n{preds['score'].describe()}")
    else:
        exp_name = make_experiment_name("train", "lightgbm")
        with experiment_context(exp_name) as _recorder:
            model, dataset, config = _train(
                model_config_path=model_config,
                data_config_path=data_config,
            )
            log_hyperparams_from_config(config)
            log_data_version()
            save_model(model, output)
            log_model_artifact(output)
            preds = predict_to_dataframe(model, dataset, segment="test")
            log_metrics({"prediction_count": float(len(preds))})
            logger.info(f"Test predictions shape: {preds.shape}")
            logger.info(f"Score stats:\n{preds['score'].describe()}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@app.command()
def predict(
    model: str = typer.Option(
        "lightgbm", help="Model type: lightgbm or kronos"
    ),
    model_path: Path = typer.Option(
        "output/lightgbm_model.pkl", help="Path to saved model (LightGBM)"
    ),
    data_config: str = typer.Option(
        "configs/data/handler_alpha158.yaml", help="Dataset config path"
    ),
    model_config: str = typer.Option(
        "configs/model/lightgbm.yaml", help="Model config path (for dataset creation)"
    ),
    kronos_config: str = typer.Option(
        "configs/model/kronos.yaml", help="Kronos config path"
    ),
    market: str = typer.Option(
        "csi300", help="Market for instrument resolution (Kronos)"
    ),
    segment: str = typer.Option(
        "test", help="Dataset segment: train/valid/test (LightGBM)"
    ),
    output: Path = typer.Option(
        None, help="Output file path (default: output/<model>_predictions)"
    ),
    local_only: bool = typer.Option(
        False, "--local-only", help="Kronos: use cached model only"
    ),
    no_track: bool = typer.Option(False, "--no-track", help="Disable experiment tracking"),
) -> None:
    """Generate predictions from a trained model."""
    if no_track:
        if model == "lightgbm":
            _predict_lightgbm(model_path, data_config, model_config, segment, output)
        elif model == "kronos":
            _predict_kronos(kronos_config, market, output, local_only)
        else:
            logger.error(f"Unknown model type: {model}. Use 'lightgbm' or 'kronos'.")
            raise typer.Exit(1)
    else:
        exp_name = make_experiment_name("predict", model)
        with experiment_context(exp_name) as _recorder:
            if model == "lightgbm":
                _predict_lightgbm_track(model_path, data_config, model_config, segment, output)
            elif model == "kronos":
                _predict_kronos_track(kronos_config, market, output, local_only)
            else:
                logger.error(f"Unknown model type: {model}. Use 'lightgbm' or 'kronos'.")
                raise typer.Exit(1)


def _predict_lightgbm(
    model_path: Path,
    data_config: str,
    model_config: str,
    segment: str,
    output: Path | None,
) -> None:
    from big_a.models.lightgbm_model import load_model, create_dataset, predict_to_dataframe
    from big_a.config import load_config
    from big_a.qlib_config import init_qlib

    init_qlib()

    mdl = load_model(model_path)
    config = load_config(model_config, data_config)
    dataset = create_dataset(config)

    preds = predict_to_dataframe(mdl, dataset, segment=segment)

    out = output or Path("output/lightgbm_predictions.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out)
    logger.info(f"Predictions saved to {out} ({len(preds)} rows)")


def _predict_kronos(
    config_path: str,
    market: str,
    output: Path | None,
    local_only: bool,
) -> None:
    from big_a.config import load_config
    from big_a.models.kronos import KronosSignalGenerator

    config = load_config(config_path)
    kconf = config.get("kronos", {})
    iconf = config.get("inference", {})

    gen = KronosSignalGenerator(
        tokenizer_id=kconf.get("tokenizer_id", "NeoQuasar/Kronos-Tokenizer-base"),
        model_id=kconf.get("model_id", "NeoQuasar/Kronos-base"),
        device=kconf.get("device", "cpu"),
        lookback=kconf.get("lookback", 90),
        pred_len=kconf.get("pred_len", 10),
        max_context=kconf.get("max_context", 512),
        signal_mode=kconf.get("signal_mode", "mean"),
    )
    gen.load_model(local_files_only=local_only)

    from qlib.data import D
    from big_a.qlib_config import init_qlib

    init_qlib()
    instruments = D.list_instruments(instruments=D.instruments(market), as_list=True)
    logger.info(f"Loaded {len(instruments)} instruments from market={market}")

    signals = gen.generate_signals(
        instruments=instruments,
        start_date=iconf.get("start_date", "2024-01-01"),
        end_date=iconf.get("end_date", "2024-12-31"),
    )

    out = output or Path("output/kronos_predictions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out)
    logger.info(f"Signals saved to {out} ({len(signals)} rows)")


def _predict_lightgbm_track(
    model_path: Path,
    data_config: str,
    model_config: str,
    segment: str,
    output: Path | None,
) -> None:
    from big_a.models.lightgbm_model import load_model, create_dataset, predict_to_dataframe
    from big_a.config import load_config
    from big_a.qlib_config import init_qlib

    init_qlib()

    config = load_config(model_config, data_config)
    log_hyperparams_from_config(config)

    mdl = load_model(model_path)
    dataset = create_dataset(config)

    preds = predict_to_dataframe(mdl, dataset, segment=segment)

    out = output or Path("output/lightgbm_predictions.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out)
    log_params({"model_type": "lightgbm", "segment": segment, "output_path": str(out)})
    log_metrics({"prediction_count": float(len(preds))})
    logger.info(f"Predictions saved to {out} ({len(preds)} rows)")


def _predict_kronos_track(
    config_path: str,
    market: str,
    output: Path | None,
    local_only: bool,
) -> None:
    from big_a.config import load_config
    from big_a.models.kronos import KronosSignalGenerator

    config = load_config(config_path)
    log_hyperparams_from_config(config)
    kconf = config.get("kronos", {})
    iconf = config.get("inference", {})

    gen = KronosSignalGenerator(
        tokenizer_id=kconf.get("tokenizer_id", "NeoQuasar/Kronos-Tokenizer-base"),
        model_id=kconf.get("model_id", "NeoQuasar/Kronos-base"),
        device=kconf.get("device", "cpu"),
        lookback=kconf.get("lookback", 90),
        pred_len=kconf.get("pred_len", 10),
        max_context=kconf.get("max_context", 512),
        signal_mode=kconf.get("signal_mode", "mean"),
    )
    gen.load_model(local_files_only=local_only)

    from qlib.data import D
    from big_a.qlib_config import init_qlib

    init_qlib()
    instruments = D.list_instruments(instruments=D.instruments(market), as_list=True)
    logger.info(f"Loaded {len(instruments)} instruments from market={market}")

    signals = gen.generate_signals(
        instruments=instruments,
        start_date=iconf.get("start_date", "2024-01-01"),
        end_date=iconf.get("end_date", "2024-12-31"),
    )

    out = output or Path("output/kronos_predictions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out)
    log_params({"output_path": str(out), "model_type": "kronos", "instrument_count": float(len(instruments))})
    log_metrics({"signal_count": float(len(signals))})
    logger.info(f"Signals saved to {out} ({len(signals)} rows)")


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

@app.command()
def backtest(
    signal_file: Path = typer.Option(
        "output/predictions.parquet", help="Path to signal predictions (parquet or CSV)"
    ),
    config: str = typer.Option(
        "configs/backtest/topk_csi300.yaml", help="Backtest config YAML path"
    ),
    output: Path = typer.Option(
        "output/backtest_report.parquet", help="Path to save backtest report"
    ),
    analysis_output: Path = typer.Option(
        "output/backtest_analysis.parquet", help="Path to save risk analysis"
    ),
    no_track: bool = typer.Option(False, "--no-track", help="Disable experiment tracking"),
) -> None:
    """Run backtest using saved signal predictions."""
    import pandas as pd

    from big_a.backtest.engine import compute_analysis, load_backtest_config, run_backtest
    from big_a.qlib_config import init_qlib

    init_qlib()

    logger.info(f"Loading signal from {signal_file}")
    if signal_file.suffix == ".csv":
        signal = pd.read_csv(signal_file, index_col=[0, 1])
    else:
        signal = pd.read_parquet(signal_file)

    bt_config = load_backtest_config(config)
    report, positions = run_backtest(signal, bt_config)

    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(output)
    logger.info(f"Report saved to {output} ({len(report)} rows)")

    analysis = compute_analysis(report)
    analysis_output.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_parquet(analysis_output)
    logger.info(f"Analysis saved to {analysis_output}")
    logger.info(f"\n{analysis.to_string()}")

    if not no_track:
        exp_name = make_experiment_name("backtest")
        with experiment_context(exp_name) as _recorder:
            log_params({"signal_file": str(signal_file)})
            metrics = {}
            for idx in analysis.index:
                for col in analysis.columns:
                    key = f"{idx}_{col}"
                    val = analysis.loc[idx, col]
                    if pd.notna(val):
                        metrics[key] = float(val)
            log_metrics(metrics)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@app.command()
def evaluate(
    kronos: Path = typer.Option(
        ..., "--kronos", help="Path to Kronos predictions (CSV)"
    ),
    lightgbm: Path = typer.Option(
        ..., "--lightgbm", help="Path to LightGBM predictions (CSV or parquet)"
    ),
    actual: Path = typer.Option(
        ..., "--actual", help="Path to actual returns (CSV)"
    ),
    output_dir: Path = typer.Option(
        "output/evaluation", help="Directory for output plots and tables"
    ),
    no_track: bool = typer.Option(False, "--no-track", help="Disable experiment tracking"),
) -> None:
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

    kronos_df = pd.read_csv(kronos, index_col=[0, 1])
    if lightgbm.suffix == ".parquet":
        lgbm_df = pd.read_parquet(lightgbm)
    else:
        lgbm_df = pd.read_csv(lightgbm, index_col=[0, 1])
    actual_s = pd.read_csv(actual, index_col=[0, 1]).squeeze("columns")

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

    from big_a.backtest.metrics import METRIC_LABELS

    for metric in ["mean_ic", "mean_rank_ic", "icir"]:
        plot_model_comparison(comparison, metric=metric, save_path=str(output_dir / f"compare_{metric}.png"))

    logger.info("Evaluation complete. Results saved to {}", output_dir)

    if not no_track:
        exp_name = make_experiment_name("evaluate")
        with experiment_context(exp_name) as _recorder:
            log_params({"kronos": str(kronos), "lightgbm": str(lightgbm)})
            for _, row in comparison.iterrows():
                model_name = row.name
                for col in comparison.columns:
                    key = f"{model_name}_{col}"
                    val = row[col]
                    if pd.notna(val):
                        log_metrics({key: float(val)})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
