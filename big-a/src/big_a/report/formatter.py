"""Rich terminal report formatter for watchlist scoring results.

Provides beautiful terminal output with colors, tables, and panels for
displaying watchlist scoring analysis.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from typing import Any


def _format_score(value: float) -> Text:
    """Format score with sign and color.

    Parameters
    ----------
    value : float
        Score value.

    Returns
    -------
    Text
        Formatted text with color.
    """
    text = f"{value:+.2f}"
    if value > 0:
        return Text(text, style="green")
    elif value < 0:
        return Text(text, style="red")
    return Text(text)


def _format_pct(value: float) -> Text:
    """Format percentage with sign and color.

    Parameters
    ----------
    value : float
        Percentage value. If NaN, returns "—"

    Returns
    -------
    Text
        Formatted text with color.
    """
    import math
    if math.isnan(value):
        return Text("—")
    text = f"{value:+.1f}%"
    if value > 0:
        return Text(text, style="green")
    elif value < 0:
        return Text(text, style="red")
    return Text(text)


def _format_amount(value: float) -> str:
    """Format monetary amount with smart unit.

    Parameters
    ----------
    value : float
        Amount in yuan.

    Returns
    -------
    str
        Formatted amount string.
    """
    if value >= 1e8:
        return f"¥{value/1e8:.1f}亿"
    elif value >= 1e4:
        return f"¥{value/1e4:.1f}万"
    return f"¥{value:,.0f}"


def _format_volume(value: float) -> str:
    """Format volume in shares with smart unit."""
    if value >= 1e8:
        return f"{value/1e8:.1f}亿股"
    elif value >= 1e4:
        return f"{value/1e4:.1f}万股"
    return f"{value:,.0f}股"


def _get_signal_label(score: float) -> Text:
    """Get signal label with emoji and color.

    Parameters
    ----------
    score : float
        Combined score in -1 to 1 range.

    Returns
    -------
    Text
        Signal label with emoji and color.
    """
    if score > 0.5:
        return Text("🟢 强烈看涨", style="green")
    elif score > 0:
        return Text("🟢 看涨", style="green")
    elif score > -0.5:
        return Text("🔴 看跌", style="red")
    else:
        return Text("🔴 强烈看跌", style="red")


def format_qualitative_analysis(results: dict[str, Any], console: Console) -> None:
    """Print the AI qualitative analysis section.

    Only renders when results contains non-empty hedge_fund_analysis.
    """
    hedge_fund = results.get("hedge_fund_analysis", {})
    if not hedge_fund or not hedge_fund.get("details"):
        return

    details = hedge_fund["details"]
    watchlist = results.get("watchlist", {})

    header = Panel(
        Text("🤖 AI 定性分析", style="bold white"),
        border_style="bright_blue",
    )
    console.print(header)
    console.print()

    AGENT_NAMES = {
        "technicals_agent": "📊 技术分析",
        "valuation_agent": "💰 估值分析",
        "warren_buffett_agent": "🏦 巴菲特",
        "risk_manager_agent": "⚠️ 风控",
        "portfolio_manager_agent": "📋 投资组合",
    }

    for instrument, agent_data in details.items():
        name = watchlist.get(instrument, instrument)
        lines = []

        for agent_name, signal_data in agent_data.items():
            display_name = AGENT_NAMES.get(agent_name, agent_name.replace("_agent", "").replace("_", " ").title())
            signal = signal_data.get("signal", "neutral")
            confidence = signal_data.get("confidence", 0.0)
            reasoning = signal_data.get("reasoning", "")

            if signal == "bullish":
                signal_icon = "🟢 看涨"
            elif signal == "bearish":
                signal_icon = "🔴 看跌"
            else:
                signal_icon = "⚪ 中性"

            conf_pct = int(confidence * 100)
            bar_filled = int(confidence * 10)
            bar_empty = 10 - bar_filled
            conf_bar = "█" * bar_filled + "░" * bar_empty

            if len(reasoning) > 200:
                reasoning = reasoning[:197] + "..."

            lines.append(f"[bold]{display_name}[/bold]  {signal_icon}  置信度: {conf_bar} {conf_pct}%")
            lines.append(f"  {reasoning}")
            lines.append("")

        panel_text = "\n".join(lines).strip()
        stock_panel = Panel(
            Text.from_markup(panel_text),
            title=f"[{instrument}] {name}",
            title_align="left",
            border_style="bright_cyan",
        )
        console.print(stock_panel)
        console.print()


def format_scores_table(results: dict[str, Any], console: Console) -> None:
    """Print just the model scores table.

    Parameters
    ----------
    results : dict
        Output from WatchlistScorer.run().
    console : Console
        Rich Console instance.
    """
    watchlist = results.get("watchlist", {})
    kronos_scores = results.get("kronos_scores", pd.DataFrame())
    lightgbm_scores = results.get("lightgbm_scores", pd.DataFrame())

    if kronos_scores.empty and lightgbm_scores.empty:
        console.print("[yellow]暂无模型打分数据[/yellow]")
        return

    table = Table(title="[1] 模型打分", show_header=True, header_style="bold magenta")
    table.add_column("代码", style="cyan", width=10)
    table.add_column("名称", style="white", width=12)
    table.add_column("Kronos", justify="right", width=10)
    table.add_column("LightGBM", justify="right", width=10)
    table.add_column("信号", style="white", width=14)

    if not kronos_scores.empty:
        latest_kronos = kronos_scores.groupby("instrument").last().reset_index()
    else:
        latest_kronos = pd.DataFrame(columns=["instrument", "score", "score_pct"])

    if not lightgbm_scores.empty:
        latest_lightgbm = lightgbm_scores.groupby("instrument").last().reset_index()
    else:
        latest_lightgbm = pd.DataFrame(columns=["date", "instrument", "score"])

    for instrument, name in watchlist.items():
        kronos_row = latest_kronos[latest_kronos["instrument"] == instrument]
        lgb_row = latest_lightgbm[latest_lightgbm["instrument"] == instrument]

        if not kronos_row.empty:
            kronos_text = _format_pct(kronos_row.iloc[0]["score_pct"])
            kronos_score_val = kronos_row.iloc[0]["score_pct"]
        else:
            kronos_text = Text("N/A", style="dim")
            kronos_score_val = 0

        if not lgb_row.empty:
            lgb_text = _format_score(lgb_row.iloc[0]["score"])
            lgb_score_val = lgb_row.iloc[0]["score"]
        else:
            lgb_text = Text("N/A", style="dim")
            lgb_score_val = 0

        combined_score = 0
        if not kronos_row.empty and not lgb_row.empty:
            combined_score = (kronos_score_val / 10.0 + lgb_score_val / 3.0) / 2.0
            combined_score = max(-1.0, min(1.0, combined_score))
        elif not kronos_row.empty:
            combined_score = max(-1.0, min(1.0, kronos_score_val / 10.0))
        elif not lgb_row.empty:
            combined_score = max(-1.0, min(1.0, lgb_score_val / 3.0))

        signal_text = _get_signal_label(combined_score)

        table.add_row(instrument, name, kronos_text, lgb_text, signal_text)

    console.print(table)


def format_trend_tables(results: dict[str, Any], console: Console) -> None:
    """Print the 10-day trend tables.

    Parameters
    ----------
    results : dict
        Output from WatchlistScorer.run().
    console : Console
        Rich Console instance.
    """
    watchlist = results.get("watchlist", {})
    kronos_trend = results.get("kronos_trend", pd.DataFrame())
    lightgbm_trend = results.get("lightgbm_trend", pd.DataFrame())

    if kronos_trend.empty and lightgbm_trend.empty:
        console.print("[yellow]暂无趋势数据[/yellow]")
        return

    if not kronos_trend.empty:
        table = Table(title="[2] 10日分数走势 (Kronos)", show_header=True, header_style="bold magenta")
        table.add_column("日期", style="cyan", width=10)

        for instrument in watchlist.keys():
            table.add_column(instrument, justify="right", width=10)

        dates = sorted(kronos_trend["date"].unique())[-10:]

        for date in dates:
            row = [date.strftime("%m-%d")]
            for instrument in watchlist.keys():
                score_row = kronos_trend[
                    (kronos_trend["date"] == date) &
                    (kronos_trend["instrument"] == instrument)
                ]
                if not score_row.empty:
                    score_val = score_row.iloc[0]["score_pct"]
                    row.append(_format_pct(score_val))
                else:
                    row.append(Text("—", style="dim"))
            table.add_row(*row)

        console.print(table)
        console.print()

    if not lightgbm_trend.empty:
        table = Table(title="[2] 10日分数走势 (LightGBM)", show_header=True, header_style="bold magenta")
        table.add_column("日期", style="cyan", width=10)

        for instrument in watchlist.keys():
            table.add_column(instrument, justify="right", width=10)

        dates = sorted(lightgbm_trend["date"].unique())[-10:]

        for date in dates:
            row = [date.strftime("%m-%d")]
            for instrument in watchlist.keys():
                score_row = lightgbm_trend[
                    (lightgbm_trend["date"] == date) &
                    (lightgbm_trend["instrument"] == instrument)
                ]
                if not score_row.empty:
                    score_val = score_row.iloc[0]["score"]
                    row.append(_format_score(score_val))
                else:
                    row.append(Text("—", style="dim"))
            table.add_row(*row)

        console.print(table)
        console.print()


def format_market_data(results: dict[str, Any], console: Console) -> None:
    """Print the market data tables.

    Parameters
    ----------
    results : dict
        Output from WatchlistScorer.run().
    console : Console
        Rich Console instance.
    """
    watchlist = results.get("watchlist", {})
    market_data = results.get("market_data", pd.DataFrame())

    if market_data.empty:
        console.print("[yellow]暂无行情数据[/yellow]")
        return

    from big_a.report.scorer import _convert_market_units
    if "factor" in market_data.columns:
        market_data = _convert_market_units(market_data)

    for instrument, name in watchlist.items():
        try:
            stock_data = market_data.xs(instrument, level=1)
        except KeyError:
            console.print(f"[yellow]暂无 {name} ({instrument}) 的行情数据[/yellow]")
            continue

        if stock_data.empty:
            console.print(f"[yellow]暂无 {name} ({instrument}) 的行情数据[/yellow]")
            continue

        stock_data = stock_data.iloc[-10:]

        table = Table(
            title=f"[3] 近10日行情 - {name} ({instrument})",
            show_header=True,
            header_style="bold magenta"
        )
        table.add_column("日期", style="cyan", width=10)
        table.add_column("开盘", justify="right", width=10)
        table.add_column("收盘", justify="right", width=10)
        table.add_column("最高", justify="right", width=10)
        table.add_column("最低", justify="right", width=10)
        table.add_column("成交额", justify="right", width=12)
        table.add_column("成交量", justify="right", width=12)
        table.add_column("涨跌幅", justify="right", width=10)

        for date, row in stock_data.iterrows():
            table.add_row(
                date.strftime("%m-%d"),
                f"{row['open_raw']:.2f}",
                f"{row['close_raw']:.2f}",
                f"{row['high_raw']:.2f}",
                f"{row['low_raw']:.2f}",
                _format_amount(row['amount_yuan']),
                _format_volume(row['volume_shares']),
                _format_pct(row['change_pct']),
            )

        console.print(table)
        console.print()


def format_portfolio(results: dict[str, Any], console: Console) -> None:
    """Print the simulated portfolio table.

    Parameters
    ----------
    results : dict
        Output from WatchlistScorer.run().
    console : Console
        Rich Console instance.
    """
    portfolio = results.get("portfolio", pd.DataFrame())

    if portfolio.empty:
        console.print("[yellow]暂无持仓数据[/yellow]")
        return

    table = Table(title="[4] 模拟持仓", show_header=True, header_style="bold magenta")
    table.add_column("代码", style="cyan", width=10)
    table.add_column("名称", style="white", width=12)
    table.add_column("仓位权重", justify="right", width=12)
    table.add_column("持仓市值", justify="right", width=14)
    table.add_column("信号", style="white", width=14)

    for _, row in portfolio.iterrows():
        instrument = row["instrument"]
        name = row["name"]
        weight = row["weight"]
        allocation = row["allocation"]
        signal = row["signal"]

        if instrument == "CASH":
            weight_text = f"{weight * 100:.1f}%"
        else:
            weight_text = f"{weight * 100:.1f}%"

        table.add_row(
            instrument,
            name,
            weight_text,
            _format_amount(allocation),
            signal,
        )

    console.print(table)


def format_summary(results: dict[str, Any], console: Console) -> None:
    """Print the summary panel.

    Parameters
    ----------
    results : dict
        Output from WatchlistScorer.run().
    console : Console
        Rich Console instance.
    """
    summary = results.get("summary", {})

    if not summary:
        console.print("[yellow]暂无总结数据[/yellow]")
        return

    total_stocks = summary.get("total_stocks", 0)
    bullish_count = summary.get("bullish_count", 0)
    bearish_count = summary.get("bearish_count", 0)
    avg_score = summary.get("avg_score", 0.0)
    best_stock = summary.get("best_stock")
    worst_stock = summary.get("worst_stock")

    summary_parts = [
        f"{bullish_count}/{total_stocks} 看涨",
        f"平均综合得分: {_format_score(avg_score).plain}",
    ]

    if best_stock:
        summary_parts.append(f"最强: {best_stock}")

    summary_text = " | ".join(summary_parts)

    panel = Panel(
        Text(summary_text, style="bold white"),
        title="[总结]",
        title_align="left",
        border_style="bright_yellow",
    )

    console.print(panel)


def format_report(results: dict[str, Any], console: Console | None = None) -> None:
    """Print the full watchlist report to terminal.

    Parameters
    ----------
    results : dict
        Output from WatchlistScorer.run().
    console : Console, optional
        Rich Console instance. Creates one if not provided.
    """
    if console is None:
        console = Console()

    current_date = datetime.now().strftime("%Y-%m-%d")

    header = Panel(
        Text(f"📊 自选股分析报告  {current_date}", style="bold white"),
        border_style="bright_blue",
    )
    console.print(header)
    console.print()

    # [0] AI qualitative analysis (only if available)
    format_qualitative_analysis(results, console)

    format_scores_table(results, console)
    console.print()

    format_trend_tables(results, console)

    format_market_data(results, console)

    format_portfolio(results, console)
    console.print()

    format_summary(results, console)
