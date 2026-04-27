# Backtest Module

**Generated:** 2026-04-27

## OVERVIEW

Backtesting engine with Qlib integration: single-shot and walk-forward rolling evaluation, IC/sharpe metrics, and interactive plotting.

## STRUCTURE

```
backtest/
├── engine.py        # run_backtest, run_backtest_with_strategy, compute_analysis
├── rolling.py       # RollingBacktester (walk-forward), generate_windows, WindowResult
├── evaluation.py    # calc_ic, calc_rank_ic, calc_sharpe, calc_max_drawdown, calc_turnover
├── analysis.py      # analyze_backtest, generate_report
├── reporting.py     # Plotly: prediction_vs_actual, residual, quantile_returns, turnover
├── plots.py         # Matplotlib: nav, drawdown, monthly_returns, ic_series
├── metrics.py       # SUCCESS_IC=0.03, SUCCESS_SHARPE=1.0, MAX_DRAWDOWN_THRESHOLD=0.20
└── __init__.py
```

## WHERE TO LOOK

| Task | Location | Key Functions |
|------|----------|---------------|
| Run backtest | `engine.py` | `run_backtest(signal, config)` returns (report, positions) |
| Walk-forward | `rolling.py` | `RollingBacktester.run_rolling()` |
| IC metrics | `evaluation.py` | `calc_ic()`, `calc_rank_ic()`, `calc_icir()` |
| Risk metrics | `evaluation.py` | `calc_sharpe()`, `calc_max_drawdown()`, `calc_turnover()` |
| Performance report | `analysis.py` | `analyze_backtest(report_df)` returns dict |
| Interactive charts | `reporting.py` | `plot_prediction_vs_actual()`, `plot_quantile_returns()` |
| Static charts | `plots.py` | `plot_nav()`, `plot_drawdown()`, `plot_monthly_returns()` |

## CONVENTIONS

- Signal is MultiIndex (datetime, instrument) with 'score' column
- Report DataFrame columns: [return, bench, cost, turnover]
- A-share defaults: limit_threshold=0.095, open_cost=0.0005, close_cost=0.0015
- Rolling windows: train/valid/test splits by year boundaries
- RollingBacktester supports lightgbm, kronos, and hedge_fund model types

## ANTI-PATTERNS

- Do not use TopkDropoutStrategy for real trading (use RealTradingStrategy in strategy/)
- Do not bypass compute_analysis when you need risk metrics
- Do not hardcode instrument pools; pass via config or instruments parameter
