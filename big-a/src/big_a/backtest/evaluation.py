"""Model evaluation: IC, Rank IC, ICIR, Sharpe, MaxDrawdown, Turnover, and comparison utilities."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr, spearmanr

from big_a.backtest.metrics import (
    MAX_DRAWDOWN_THRESHOLD,
    METRIC_LABELS,
    SUCCESS_IC,
    SUCCESS_SHARPE,
)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def calc_ic(predicted: pd.Series | pd.DataFrame, actual: pd.Series | pd.DataFrame) -> pd.Series:
    """Cross-sectional Pearson IC per date.

    Parameters
    ----------
    predicted : Series or DataFrame with 'score' column, MultiIndex (datetime, instrument).
    actual : Same shape — actual forward returns.

    Returns
    -------
    pd.Series indexed by datetime, values = Pearson correlation.
    """
    pred = _to_series(predicted, "score")
    act = _to_series(actual, "score")

    dates = pred.index.get_level_values(0).unique()
    ic_values = []
    ic_dates = []
    for date in dates:
        p = pred.xs(date, level=0)
        a = act.xs(date, level=0)
        common = p.index.intersection(a.index)
        if len(common) < 3:
            continue
        p_slice = p.loc[common].values
        a_slice = a.loc[common].values
        mask = np.isfinite(p_slice) & np.isfinite(a_slice)
        if mask.sum() < 3:
            continue
        corr, _ = pearsonr(p_slice[mask], a_slice[mask])
        ic_values.append(corr)
        ic_dates.append(date)

    return pd.Series(ic_values, index=ic_dates, name="ic")


def calc_rank_ic(predicted: pd.Series | pd.DataFrame, actual: pd.Series | pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank IC per date."""
    pred = _to_series(predicted, "score")
    act = _to_series(actual, "score")

    dates = pred.index.get_level_values(0).unique()
    ic_values = []
    ic_dates = []
    for date in dates:
        p = pred.xs(date, level=0)
        a = act.xs(date, level=0)
        common = p.index.intersection(a.index)
        if len(common) < 3:
            continue
        p_slice = p.loc[common].values
        a_slice = a.loc[common].values
        mask = np.isfinite(p_slice) & np.isfinite(a_slice)
        if mask.sum() < 3:
            continue
        corr, _ = spearmanr(p_slice[mask], a_slice[mask])
        ic_values.append(corr)
        ic_dates.append(date)

    return pd.Series(ic_values, index=ic_dates, name="rank_ic")


def calc_icir(ic_series: pd.Series) -> float:
    """ICIR = mean(IC) / std(IC)."""
    if len(ic_series) < 2:
        return np.nan
    return float(ic_series.mean() / ic_series.std())


def calc_sharpe(daily_returns: pd.Series) -> float:
    """Annualized Sharpe ratio: sqrt(252) * mean / std."""
    if len(daily_returns) < 2:
        return np.nan
    clean = daily_returns.dropna()
    if clean.std() == 0:
        return np.nan
    return float(np.sqrt(252) * clean.mean() / clean.std())


def calc_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum drawdown as a positive fraction (0.0 = no drawdown)."""
    if len(cumulative_returns) < 2:
        return 0.0
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return float(-drawdown.min())


def calc_turnover(positions: pd.DataFrame) -> float:
    """Average daily turnover across dates.

    Parameters
    ----------
    positions : DataFrame(index=[datetime, instrument], columns=['weight'])
        Portfolio weights per date and instrument.

    Returns
    -------
    float
        Mean turnover (fraction of portfolio changed per day).
    """
    if "weight" not in positions.columns:
        raise ValueError("positions must have a 'weight' column")

    dates = positions.index.get_level_values(0).unique().sort_values()
    if len(dates) < 2:
        return 0.0

    turnovers = []
    for i in range(1, len(dates)):
        prev_w = positions.xs(dates[i - 1], level=0)["weight"]  # type: ignore[arg-type]
        curr_w = positions.xs(dates[i], level=0)["weight"]  # type: ignore[arg-type]
        common = prev_w.index.union(curr_w.index)
        prev_full = prev_w.reindex(common, fill_value=0.0)
        curr_full = curr_w.reindex(common, fill_value=0.0)
        turnovers.append(float(np.abs(curr_full - prev_full).sum() / 2))

    return float(np.mean(turnovers))


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    predictions: dict[str, pd.DataFrame],
    actual_returns: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Produce a comparison table for multiple models.

    Parameters
    ----------
    predictions : dict mapping model name → DataFrame(index=[datetime, instrument], columns=['score'])
    actual_returns : actual forward returns in same format.

    Returns
    -------
    pd.DataFrame with columns: mean_ic, mean_rank_ic, icir, and rows = model names.
    """
    actual = _to_series(actual_returns, "score")
    rows = []
    for name, pred_df in predictions.items():
        pred_s = _to_series(pred_df, "score")
        ic = calc_ic(pred_s, actual)
        rank_ic = calc_rank_ic(pred_s, actual)
        icir = calc_icir(ic)
        rows.append({
            "model": name,
            "mean_ic": ic.mean() if len(ic) > 0 else np.nan,
            "mean_rank_ic": rank_ic.mean() if len(rank_ic) > 0 else np.nan,
            "icir": icir,
        })
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ic_series(
    ic_series: pd.Series,
    title: str = "IC Series",
    save_path: str | None = None,
) -> None:
    """Plot IC time series with mean line."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(ic_series.index, ic_series.values, width=0.8, alpha=0.7, label="IC")
    ax.axhline(ic_series.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean={ic_series.mean():.4f}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("IC")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("IC plot saved to {}", save_path)
    plt.close(fig)


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "mean_ic",
    save_path: str | None = None,
) -> None:
    """Bar chart comparing models on a given metric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    values = comparison_df[metric]
    ax.bar(comparison_df.index, values, alpha=0.7)
    ax.set_title(f"Model Comparison — {METRIC_LABELS.get(metric, metric)}")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Model comparison plot saved to {}", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_series(
    data: pd.Series | pd.DataFrame,
    col: str = "score",
) -> pd.Series:
    """Convert DataFrame to Series by extracting *col*, or pass Series through."""
    if isinstance(data, pd.DataFrame):
        if col not in data.columns:
            raise ValueError(f"DataFrame must contain column '{col}', got {data.columns.tolist()}")
        return data[col]  # type: ignore[return-value]
    return data  # type: ignore[return-value]
