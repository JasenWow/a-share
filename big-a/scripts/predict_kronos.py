#!/usr/bin/env python3
"""CLI entry point for Kronos signal generation."""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from typing import Optional

import typer
from loguru import logger

app = typer.Typer(help="Kronos pre-trained model inference for A-share signals")


@app.command()
def predict(
    instruments: Optional[list[str]] = typer.Argument(None, help="Instrument codes (e.g. SH600000). If omitted, uses config market."),
    config_path: str = typer.Option("configs/model/kronos.yaml", "--config", help="Path to Kronos config YAML"),
    start_date: Optional[str] = typer.Option(None, "--start-date", help="Override start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="Override end date (YYYY-MM-DD)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output CSV path (default: stdout)"),
    local_only: bool = typer.Option(False, "--local-only", help="Use cached model only, skip download"),
) -> None:
    """Generate Kronos trading signals for the given instruments."""
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

    if not instruments:
        market = iconf.get("instruments", "csi300")
        try:
            from qlib.data import D
            from big_a.qlib_config import init_qlib
            init_qlib()
            instruments = list(D.instruments(market))
            logger.info("Loaded {} instruments from market={}", len(instruments), market)
        except Exception as exc:
            logger.error("Cannot resolve market instruments: {}", exc)
            raise typer.Exit(1)

    signals = gen.generate_signals(
        instruments=instruments,
        start_date=start_date or iconf.get("start_date", "2024-01-01"),
        end_date=end_date or iconf.get("end_date", "2024-12-31"),
    )

    if output:
        signals.to_csv(output)
        logger.info("Signals saved to {}", output)
    else:
        print(signals.to_string())


if __name__ == "__main__":
    app()
