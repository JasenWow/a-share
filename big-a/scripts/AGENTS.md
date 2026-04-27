# SCRIPTS KNOWLEDGE BASE

**Generated:** 2026-04-27

## OVERVIEW
CLI entry points for training, prediction, backtesting, and data management.

## WHERE TO LOOK
| Category | Scripts |
|----------|---------|
| Main CLI | `run.py` (train/predict/backtest subcommands via typer) |
| Pipeline | `e2e.py` (full LightGBM + Kronos pipeline) |
| Backtest | `backtest.py`, `evaluate.py`, `analyze.py`, `roll_backtest.py`, `real_trading_backtest.py` |
| Models | `train_lightgbm.py`, `predict_kronos.py`, `predict_lightgbm.py` |
| Data | `update_data.py`, `validate_data.py` |
| Analysis | `watchlist_report.py` (watchlist scoring), `run_universe.py` |

## CONVENTIONS
- **Invocation**: Always `python scripts/x.py` from `big-a/` directory
- **NOT** `python -m scripts.x` (scripts/ lacks `__init__.py`)
- **Working directory**: Scripts expect to run from `big-a/` with `src/` on path

## ANTI-PATTERNS
- Do not run `update_data.py` for real-time streaming (Phase 1 restriction)
- Do not import scripts as modules (no package structure)
