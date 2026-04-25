#!/usr/bin/env python3
"""Generate predictions from a saved LightGBM model."""
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
    model_path: Path = typer.Option(
        "output/lightgbm_model.pkl", help="Path to saved model"
    ),
    data_config: str = typer.Option(
        "configs/data/handler_alpha158.yaml", help="Dataset config path"
    ),
    model_config: str = typer.Option(
        "configs/model/lightgbm.yaml", help="Model config path (for dataset creation)"
    ),
    segment: str = typer.Option("test", help="Dataset segment: train/valid/test"),
    output: Path = typer.Option(
        "output/lightgbm_predictions.parquet", help="Path to save predictions"
    ),
):
    """Load a trained model and generate predictions."""
    from big_a.models.lightgbm_model import load_model, create_dataset, predict_to_dataframe
    from big_a.config import load_config
    from big_a.qlib_config import init_qlib

    init_qlib()

    model = load_model(model_path)

    config = load_config(model_config, data_config)
    dataset = create_dataset(config)

    preds = predict_to_dataframe(model, dataset, segment=segment)

    output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(output)
    logger.info(f"Predictions saved to {output} ({len(preds)} rows)")


if __name__ == "__main__":
    app()
