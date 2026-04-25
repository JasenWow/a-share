"""Backtest analysis: comprehensive performance metrics from a backtest report."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from big_a.backtest.evaluation import calc_max_drawdown, calc_sharpe


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_backtest(report_df: pd.DataFrame) -> dict[str, Any]:
    """Compute comprehensive performance metrics from a backtest report.

    Parameters
    ----------
    report_df : pd.DataFrame
        Backtest report with columns ``[return, bench, cost, turnover]``
        and a ``DatetimeIndex``.

    Returns
    -------
    dict[str, Any]
        Keys:
        - annualized_return, annualized_benchmark
        - excess_return
        - sharpe_ratio, information_ratio
        - max_drawdown, drawdown_duration_days
        - total_cost
        - mean_turnover, max_turnover
        - monthly_return_distribution (Series of year-month → return)
        - n_trading_days, start_date, end_date
    """
    df = report_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    strategy_returns: pd.Series = df["return"]
    bench_returns: pd.Series = df["bench"]
    cost: pd.Series = df["cost"]
    turnover: pd.Series = df["turnover"]

    n_days = len(df)
    trading_days_per_year = 252

    cum_strategy = (1 + strategy_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()

    annualized_return = cum_strategy.iloc[-1] ** (trading_days_per_year / n_days) - 1
    annualized_benchmark = cum_bench.iloc[-1] ** (trading_days_per_year / n_days) - 1
    excess_return = annualized_return - annualized_benchmark

    sharpe_ratio = calc_sharpe(strategy_returns)

    active_return = strategy_returns - bench_returns
    if len(active_return) >= 2 and active_return.std() != 0:
        information_ratio = float(
            np.sqrt(trading_days_per_year) * active_return.mean() / active_return.std()
        )
    else:
        information_ratio = float(np.nan)

    max_dd = calc_max_drawdown(cum_strategy)
    drawdown_duration_days = _max_drawdown_duration(cum_strategy)

    total_cost = float(cost.sum())
    mean_turnover = float(turnover.mean())
    max_turnover = float(turnover.max())

    monthly_returns = strategy_returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly_returns.index = monthly_returns.index.strftime("%Y-%m")

    return {
        "annualized_return": float(annualized_return),
        "annualized_benchmark": float(annualized_benchmark),
        "excess_return": float(excess_return),
        "sharpe_ratio": float(sharpe_ratio),
        "information_ratio": float(information_ratio),
        "max_drawdown": float(max_dd),
        "drawdown_duration_days": int(drawdown_duration_days),
        "total_cost": float(total_cost),
        "mean_turnover": mean_turnover,
        "max_turnover": max_turnover,
        "monthly_return_distribution": monthly_returns,
        "n_trading_days": n_days,
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(analysis: dict[str, Any], output_dir: str | Path) -> Path:
    """Save a text summary and charts from *analysis* to *output_dir*.

    Parameters
    ----------
    analysis : dict
        Output of ``analyze_backtest``.
    output_dir : str or Path
        Directory to write into (created if missing).

    Returns
    -------
    Path
        Path to the output directory.
    """
    from big_a.backtest.plots import (
        plot_drawdown,
        plot_monthly_returns,
        plot_nav,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = _format_summary(analysis)
    summary_path = out / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Summary saved to {}", summary_path)

    report_df = analysis.get("_report_df")
    if report_df is not None:
        cum_strategy = (1 + report_df["return"]).cumprod()
        cum_bench = (1 + report_df["bench"]).cumprod()

        plot_nav(cum_strategy, cum_bench, save_path=str(out / "nav.png"))

        plot_drawdown(cum_strategy, save_path=str(out / "drawdown.png"))

        plot_monthly_returns(
            report_df["return"], save_path=str(out / "monthly_returns.png")
        )

    logger.info("All charts saved to {}", out)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _max_drawdown_duration(cumulative: pd.Series) -> int:
    """Longest drawdown duration in trading days."""
    peak = cumulative.cummax()
    in_drawdown = cumulative < peak
    if not in_drawdown.any():
        return 0

    max_len = 0
    current_len = 0
    for val in in_drawdown:
        if val:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 0
    return max_len


def _format_summary(a: dict[str, Any]) -> str:
    """Pretty-print the analysis dict."""
    lines = [
        "=" * 60,
        "  BACKTEST PERFORMANCE SUMMARY",
        "=" * 60,
        f"  Period       : {a['start_date']} → {a['end_date']} ({a['n_trading_days']} days)",
        "-" * 60,
        f"  Annualized Return   : {a['annualized_return']:>+9.2%}",
        f"  Annualized Benchmark: {a['annualized_benchmark']:>+9.2%}",
        f"  Excess Return       : {a['excess_return']:>+9.2%}",
        "-" * 60,
        f"  Sharpe Ratio        : {a['sharpe_ratio']:>+9.4f}",
        f"  Information Ratio   : {a['information_ratio']:>+9.4f}",
        "-" * 60,
        f"  Max Drawdown        : {a['max_drawdown']:>9.2%}",
        f"  Max DD Duration     : {a['drawdown_duration_days']:>6d} days",
        "-" * 60,
        f"  Total Cost          : {a['total_cost']:>9.4f}",
        f"  Mean Turnover       : {a['mean_turnover']:>9.4f}",
        f"  Max  Turnover       : {a['max_turnover']:>9.4f}",
        "=" * 60,
    ]
    return "\n".join(lines)
