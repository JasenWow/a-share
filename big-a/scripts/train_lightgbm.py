#!/usr/bin/env python3
"""Train a LightGBM model on Alpha158 features."""
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
    model_config: str = typer.Option(
        "configs/model/lightgbm.yaml", help="Model config path (relative to project root)"
    ),
    data_config: str = typer.Option(
        "configs/data/handler_alpha158.yaml", help="Dataset config path (relative to project root)"
    ),
    output: Path = typer.Option(
        "output/lightgbm_model.pkl", help="Path to save the trained model"
    ),
):
    """Train a LightGBM model and save to disk."""
    from big_a.models.lightgbm_model import train, save_model, predict_to_dataframe

    model, dataset, config = train(
        model_config_path=model_config,
        data_config_path=data_config,
    )

    save_model(model, output)

    preds = predict_to_dataframe(model, dataset, segment="test")
    logger.info(f"Test predictions shape: {preds.shape}")
    logger.info(f"Score stats:\n{preds['score'].describe()}")


if __name__ == "__main__":
    app()
