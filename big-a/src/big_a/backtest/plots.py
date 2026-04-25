"""Backtest visualization: NAV, drawdown, monthly returns, IC series."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# NAV curve
# ---------------------------------------------------------------------------

def plot_nav(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    title: str = "NAV Curve",
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strategy_nav.index, strategy_nav.values, label="Strategy", linewidth=1.2)
    ax.plot(benchmark_nav.index, benchmark_nav.values, label="Benchmark", linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_close(fig, save_path, "NAV")


# ---------------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------------

def plot_drawdown(
    cumulative_returns: pd.Series,
    title: str = "Drawdown",
    save_path: str | None = None,
) -> None:
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_close(fig, save_path, "Drawdown")


# ---------------------------------------------------------------------------
# Monthly returns heatmap
# ---------------------------------------------------------------------------

def plot_monthly_returns(
    returns: pd.Series,
    title: str = "Monthly Returns",
    save_path: str | None = None,
) -> None:
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    years = monthly.index.year.unique()
    months = range(1, 13)

    data = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            mask = (monthly.index.year == year) & (monthly.index.month == month)
            if mask.any():
                data[i, j] = monthly.loc[mask].iloc[-1]

    fig, ax = plt.subplots(figsize=(10, max(3, len(years) * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years])
    ax.set_title(title)

    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="Return")
    fig.tight_layout()
    _save_or_close(fig, save_path, "Monthly returns")


# ---------------------------------------------------------------------------
# IC series
# ---------------------------------------------------------------------------

def plot_ic_series(
    ic_data: pd.Series,
    title: str = "IC Series",
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(ic_data.index, ic_data.values, width=0.8, alpha=0.7, label="IC")
    mean_ic = ic_data.mean()
    ax.axhline(mean_ic, color="red", linestyle="--", linewidth=1, label=f"Mean={mean_ic:.4f}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("IC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_close(fig, save_path, "IC series")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_or_close(fig: plt.Figure, save_path: str | None, label: str) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("{} plot saved to {}", label, save_path)
    plt.close(fig)
