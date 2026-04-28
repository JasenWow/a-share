#!/usr/bin/env python3
"""CLI for comparing ML experiment results."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="实验对比工具 - 对比不同实验的超参/指标/策略/滚动回测")


@app.command()
def params(
    experiment: str = typer.Option(None, "--experiment", "-e", help="实验名前缀过滤"),
    limit: int = typer.Option(20, "--limit", "-n", help="最多显示的实验数"),
) -> None:
    """对比不同实验的超参数差异 (高亮不同值)."""
    from big_a.tracking.comparison import query_experiments, compare_by_params

    console = Console()

    try:
        results = query_experiments(name_pattern=experiment, limit=limit)

        if not results:
            console.print("[yellow]未找到匹配的实验[/yellow]")
            return

        exp_names = list({r["experiment_name"] for r in results})
        df = compare_by_params(exp_names)

        if df.empty:
            console.print("[yellow]未找到参数数据[/yellow]")
            return

        table = Table(title="实验超参数对比")
        table.add_column("run_id", style="cyan", no_wrap=True)

        for col in df.columns:
            if col not in ("run_id", "experiment"):
                table.add_column(col, style="green")

        for _, row in df.iterrows():
            table.add_row(*[str(row.get(c, "")) for c in df.columns])

        console.print(table)
        console.print(f"\n[dim]共 {len(df)} 条记录[/dim]")

    except Exception as e:
        logger.exception("Failed to compare params: {}", e)
        console.print(f"[red]对比失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def models(
    experiment: str = typer.Option(None, "--experiment", "-e", help="实验名前缀过滤"),
    limit: int = typer.Option(20, "--limit", "-n", help="最多显示的实验数"),
) -> None:
    """对比不同模型 (Kronos vs LightGBM) 的表现."""
    from big_a.tracking.comparison import query_experiments, compare_by_metrics

    console = Console()

    try:
        results = query_experiments(name_pattern=experiment, limit=limit)

        if not results:
            console.print("[yellow]未找到匹配的实验[/yellow]")
            return

        exp_names = list({r["experiment_name"] for r in results})
        df = compare_by_metrics(exp_names)

        if df.empty:
            console.print("[yellow]未找到指标数据[/yellow]")
            return

        metric_cols = [c for c in df.columns if c not in ("run_id", "experiment")]
        if not metric_cols:
            console.print("[yellow]未找到模型指标[/yellow]")
            return

        table = Table(title="模型指标对比")
        table.add_column("experiment", style="cyan", no_wrap=True)

        for col in metric_cols:
            table.add_column(col, style="magenta")

        for _, row in df.iterrows():
            table.add_row(
                row.get("experiment", ""),
                *[f"{row.get(c, float('nan')):.4f}" if c in row and isinstance(row.get(c), (int, float)) else str(row.get(c, "")) for c in metric_cols],
            )

        console.print(table)
        console.print(f"\n[dim]共 {len(df)} 条记录[/dim]")

    except Exception as e:
        logger.exception("Failed to compare models: {}", e)
        console.print(f"[red]对比失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def strategy(
    experiment: str = typer.Option(None, "--experiment", "-e", help="实验名前缀过滤"),
    limit: int = typer.Option(20, "--limit", "-n", help="最多显示的实验数"),
) -> None:
    """对比不同策略配置的回测结果."""
    from big_a.tracking.comparison import query_experiments, compare_by_metrics

    console = Console()

    try:
        results = query_experiments(name_pattern=experiment, limit=limit)

        if not results:
            console.print("[yellow]未找到匹配的实验[/yellow]")
            return

        exp_names = list({r["experiment_name"] for r in results})
        df = compare_by_metrics(exp_names)

        if df.empty:
            console.print("[yellow]未找到策略回测数据[/yellow]")
            return

        metric_cols = [c for c in df.columns if c not in ("run_id", "experiment")]
        if not metric_cols:
            console.print("[yellow]未找到回测指标[/yellow]")
            return

        table = Table(title="策略回测对比")
        table.add_column("experiment", style="cyan", no_wrap=True)

        for col in metric_cols:
            table.add_column(col, style="yellow")

        for _, row in df.iterrows():
            table.add_row(
                row.get("experiment", ""),
                *[str(row.get(c, "")) for c in metric_cols],
            )

        console.print(table)
        console.print(f"\n[dim]共 {len(df)} 条记录[/dim]")

    except Exception as e:
        logger.exception("Failed to compare strategy: {}", e)
        console.print(f"[red]对比失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def rolling(
    experiment: str = typer.Option(None, "--experiment", "-e", help="实验名前缀"),
) -> None:
    """展示滚动回测的 per-window 指标趋势."""
    from big_a.tracking.comparison import get_rolling_history

    console = Console()

    if not experiment:
        console.print("[red]必须指定实验名 (--experiment / -e)[/red]")
        raise typer.Exit(1)

    try:
        df = get_rolling_history(experiment)

        if df.empty:
            console.print("[yellow]未找到滚动回测数据[/yellow]")
            return

        table = Table(title=f"滚动回测 - {experiment}")
        table.add_column("window", style="cyan", no_wrap=True)
        table.add_column("metric_name", style="green")
        table.add_column("value", style="magenta")

        for _, row in df.iterrows():
            value = row.get("value", "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            table.add_row(str(row.get("window", "")), row.get("metric_name", ""), str(value))

        console.print(table)
        console.print(f"\n[dim]共 {len(df)} 条记录[/dim]")

    except Exception as e:
        logger.exception("Failed to get rolling history: {}", e)
        console.print(f"[red]获取滚动历史失败: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
