# Big-A Package Knowledge Base

**Generated:** 2026-04-27

## OVERVIEW

Core trading system implementation with backtest engine, factor definitions, ML models (Kronos/LightGBM), and real trading strategy.

## STRUCTURE

```
big_a/src/big_a/
├── backtest/       # Backtesting engine, rolling window, evaluation, metrics, plots
├── data/           # Data updater, loader, validator
├── factors/        # Alpha factor definitions
├── models/         # Kronos transformer, LightGBM, hedge fund ensemble
├── report/         # Report formatting and scoring
├── strategy/       # Real trading strategy
├── workflow/       # Prefect workflow orchestration
├── scheduler/      # Task scheduling
├── config.py       # Configuration management
├── experiment.py   # Experiment tracking
└── qlib_config.py  # Qlib initialization
```

## WHERE TO LOOK

| Task | Location | Key Files |
|------|----------|-----------|
| Backtest | `backtest/` | engine.py (core), rolling.py (rolling window), evaluation.py, metrics.py, plots.py |
| Data | `data/` | updater/, loader/, validator/ |
| Factors | `factors/` | alpha_factors.py |
| Kronos model | `models/kronos_model/` | kronos.py (wrapper), transformer architecture |
| LightGBM model | `models/` | lightgbm_model.py |
| Ensemble | `models/hedge_fund/` | hedge_fund/ multi-model portfolio |
| Reports | `report/` | formatter.py, scorer.py |
| Real trading | `strategy/` | real_trading.py |
| Workflows | `workflow/` | Prefect flow definitions |
| Config | Root | config.py, experiment.py, qlib_config.py |

## CONVENTIONS

- Config uses dataclasses for strong typing
- Backtest returns dict with 'portfolio' and 'positions' keys
- Factor names prefixed with "alpha_" for Qlib compatibility
- Models implement fit/predict interface with sklearn compatibility

## ANTI-PATTERNS

- Do not instantiate Qlib handler directly; use qlib_config.py init functions
- Do not bypass the report/formatter layer for output
- Do not mix pandas DataFrame and Qlib DataHandler interally
