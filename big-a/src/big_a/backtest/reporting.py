from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger
import plotly.graph_objects as go
from plotly.graph_objects import Figure as goFigure
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from collections.abc import Sequence


def plot_rolling_metrics(
    results: list,
    metric_names: list[str] | None = None,
) -> goFigure:
    """Plot rolling backtest metrics over time.

    Args:
        results: List of WindowResult-like objects with attributes:
            window_idx, test_start, test_end, ic, rank_ic, icir, sharpe, max_drawdown
        metric_names: List of metric names to plot. Defaults to ["ic", "sharpe", "max_drawdown"].

    Returns:
        Plotly Figure with rolling metric trends.
    """
    if metric_names is None:
        metric_names = ["ic", "sharpe", "max_drawdown"]

    fig = go.Figure()

    for metric in metric_names:
        dates = []
        values = []

        for result in results:
            try:
                dates.append(result.test_end)
                values.append(getattr(result, metric, np.nan))
            except AttributeError:
                logger.warning(f"Result missing attribute: {metric}")
                continue

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines+markers",
                name=metric.upper(),
                marker=dict(size=6),
                line=dict(width=1.5),
            )
        )

    fig.update_layout(
        title="Rolling Backtest Metric Trends",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig


def plot_factor_distribution(features: pd.DataFrame, n_cols: int = 4) -> goFigure:
    """Plot feature distributions as histograms in a grid.

    Args:
        features: DataFrame with columns = feature names.
        n_cols: Number of columns in subplot grid.

    Returns:
        Plotly Figure with feature distribution histograms.
    """
    feature_names = features.columns.tolist()
    # Limit to ~16 features for readability
    feature_names = feature_names[:16]

    n_features = len(feature_names)
    n_rows = (n_features + n_cols - 1) // n_cols

    subplot_titles = feature_names
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for idx, feature in enumerate(feature_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        values = features[feature].dropna().values

        fig.add_trace(
            go.Histogram(
                x=values,
                name=feature,
                showlegend=False,
                nbinsx=30,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Feature Distribution",
        height=300 * n_rows,
        showlegend=False,
    )

    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    return fig


def plot_factor_correlation(features: pd.DataFrame) -> goFigure:
    """Plot feature correlation matrix as a heatmap.

    Args:
        features: DataFrame with columns = feature names.

    Returns:
        Plotly Figure with correlation heatmap.
    """
    corr_matrix = features.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        width=max(600, len(corr_matrix) * 40),
        height=max(600, len(corr_matrix) * 40),
    )

    return fig


def plot_factor_ic_decay(ic_by_lag: pd.DataFrame) -> goFigure:
    """Plot factor IC decay over lag periods.

    Args:
        ic_by_lag: DataFrame with columns = factor names, index = lag periods.

    Returns:
        Plotly Figure with IC decay lines for each factor.
    """
    fig = go.Figure()

    for factor in ic_by_lag.columns:
        fig.add_trace(
            go.Scatter(
                x=ic_by_lag.index,
                y=ic_by_lag[factor].values,
                mode="lines+markers",
                name=factor,
                marker=dict(size=6),
                line=dict(width=1.5),
            )
        )

    fig.update_layout(
        title="Factor IC Decay",
        xaxis_title="Lag Periods",
        yaxis_title="IC Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig


def plot_prediction_vs_actual(
    predicted: pd.Series,
    actual: pd.Series,
    title: str = "Prediction vs Actual",
) -> goFigure:
    """Plot predicted vs actual values with regression line.

    Args:
        predicted: Series of predicted values.
        actual: Series of actual values.
        title: Chart title.

    Returns:
        Plotly Figure with scatter plot and regression line.
    """
    common_index = predicted.index.intersection(actual.index)
    pred = predicted.loc[common_index].values
    act = actual.loc[common_index].values

    valid_mask = ~(np.isnan(pred) | np.isnan(act))
    pred = pred[valid_mask]
    act = act[valid_mask]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pred,
            y=act,
            mode="markers",
            name="Predictions",
            marker=dict(size=5, opacity=0.6),
        )
    )

    if len(pred) > 1:
        coeffs = np.polyfit(pred, act, 1)
        reg_line = np.poly1d(coeffs)
        x_line = np.array([pred.min(), pred.max()])
        y_line = reg_line(x_line)

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Regression",
                line=dict(color="red", width=2, dash="dash"),
        )
    )

    min_val = min(pred.min(), act.min())
    max_val = max(pred.max(), act.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="y = x",
            line=dict(color="gray", width=1, dash="dot"),
        )
    )

    r_squared = np.corrcoef(pred, act)[0, 1] ** 2 if len(pred) > 1 else np.nan
    pearson_r = np.corrcoef(pred, act)[0, 1] if len(pred) > 1 else np.nan

    annotation_text = f"R² = {r_squared:.4f}<br>Pearson r = {pearson_r:.4f}"

    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )

    return fig


def plot_residual_analysis(residuals: pd.Series) -> goFigure:
    """Plot residual analysis with histogram and box plot.

    Args:
        residuals: Series of residual values.

    Returns:
        Plotly Figure with residual histogram and box plot.
    """
    values = np.asarray(residuals.dropna().values)
    mean_val = np.mean(values)
    std_val = np.std(values)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.05,
    )

    fig.add_trace(
        go.Histogram(
            x=values,
            name="Residuals",
            nbinsx=30,
        ),
        row=1,
        col=1,
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        row="1",
        col="1",
    )

    fig.add_trace(
        go.Box(
            y=values,
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Residual Value", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Residual Value", row=1, col=2)

    fig.update_layout(
        title="Residual Analysis",
        height=400,
    )

    annotation_text = f"Mean: {mean_val:.4f}<br>Std: {std_val:.4f}"

    fig.add_annotation(
        x=0.95,
        y=0.95,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    return fig


def plot_quantile_returns(quantile_returns: pd.DataFrame) -> goFigure:
    """Plot mean returns by prediction quantile.

    Args:
        quantile_returns: DataFrame with columns = quantile labels,
                         index = periods or single row.

    Returns:
        Plotly Figure with bar chart of quantile returns.
    """
    # Calculate mean return per quantile (handle both single row and multi-row)
    if len(quantile_returns) == 1:
        returns = np.asarray(quantile_returns.iloc[0])
    else:
        returns = np.asarray(quantile_returns.mean())

    quantile_labels = quantile_returns.columns.tolist()

    fig = go.Figure(
        data=[
            go.Bar(
                x=quantile_labels,
                y=returns,
                marker_color=returns,
                marker_colorscale="RdYlGn",
            )
        ]
    )

    fig.update_layout(
        title="Returns by Prediction Quantile",
        xaxis_title="Quantile",
        yaxis_title="Mean Return",
    )

    return fig


def plot_holding_concentration(positions_df: pd.DataFrame) -> goFigure:
    """Plot top-10 holdings weight over time as stacked area chart.

    Args:
        positions_df: DataFrame with columns [datetime, instrument, weight].

    Returns:
        Plotly Figure with stacked area chart of holdings.
    """
    pivoted = positions_df.pivot(index="datetime", columns="instrument", values="weight")

    top_instruments = pivoted.sum().nlargest(10).index.tolist()

    # Keep only top-10 and aggregate others as "Others"
    top_df = pivoted[top_instruments].copy()
    others_weight = pivoted.drop(columns=top_instruments).sum(axis=1)
    if others_weight.sum() > 0:
        top_df["Others"] = others_weight

    fig = go.Figure()

    for instrument in top_df.columns:
        fig.add_trace(
            go.Scatter(
                x=top_df.index,
                y=np.asarray(top_df[instrument]),
                mode="none",
                fill="tonexty",
                name=instrument,
                stackgroup="one",
            )
        )

    fig.update_layout(
        title="Top-10 Holdings Weight Over Time",
        xaxis_title="Date",
        yaxis_title="Weight",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig


def plot_turnover_analysis(report: pd.DataFrame) -> goFigure:
    """Plot daily turnover with rolling mean.

    Args:
        report: DataFrame with 'turnover' column and DatetimeIndex.

    Returns:
        Plotly Figure with turnover line and rolling mean.
    """
    if "turnover" not in report.columns:
        logger.error("Report must contain 'turnover' column")
        raise ValueError("Report must contain 'turnover' column")

    turnover = report["turnover"]
    rolling_mean = turnover.rolling(window=20).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=report.index,
            y=np.asarray(turnover),
            mode="lines",
            name="Daily Turnover",
            line=dict(width=1),
            opacity=0.7,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=report.index,
            y=np.asarray(rolling_mean),
            mode="lines",
            name="20-Day Rolling Mean",
            line=dict(width=2, color="red"),
        )
    )

    fig.update_layout(
        title="Turnover Analysis",
        xaxis_title="Date",
        yaxis_title="Turnover",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig
