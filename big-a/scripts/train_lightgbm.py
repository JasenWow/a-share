#!/usr/bin/env python3
"""Train a LightGBM model on Alpha158 features."""
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
)

app = typer.Typer()


@app.command()
def main(
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
):
    """Train a LightGBM model and save to disk."""
    from big_a.models.lightgbm_model import train, save_model, predict_to_dataframe

    if no_track:
        model, dataset, config = train(
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
            model, dataset, config = train(
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


if __name__ == "__main__":
    app()
