#!/usr/bin/env python3
"""CLI for watchlist stock scoring and analysis report."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import typer
from loguru import logger
from rich.console import Console

app = typer.Typer(help="自选股分析报告 - 打分、趋势、模拟持仓")


@app.command()
def report(
    watchlist: str = typer.Option("configs/watchlist.yaml", "--watchlist", "-w", help="自选股配置文件路径"),
    account: float = typer.Option(1_000_000, "--account", "-a", help="模拟持仓资金"),
    model: str = typer.Option("kronos", "--model", "-m", help="使用的模型: all, kronos, lightgbm"),
    no_portfolio: bool = typer.Option(False, "--no-portfolio", help="不显示模拟持仓"),
    output: Path = typer.Option(None, "--output", "-o", help="保存报告到文件 (txt)"),
    local_only: bool = typer.Option(True, "--local-only", help="仅使用本地缓存的Kronos模型"),
    lookback_days: int = typer.Option(5, "--lookback", "-l", help="历史趋势天数 (Kronos滚动推理较慢)"),
    skip_trend: bool = typer.Option(False, "--skip-trend", help="跳过Kronos滚动打分 (大幅加速)"),
) -> None:
    """Generate full watchlist scoring report with trends and portfolio simulation."""
    from big_a.report.scorer import WatchlistScorer
    from big_a.report.formatter import format_report

    console = Console()

    try:
        logger.info("Starting watchlist report generation")
        logger.info("Watchlist: {}", watchlist)
        logger.info("Account: ¥{}", account)
        logger.info("Model: {}", model)
        logger.info("Lookback days: {}", lookback_days)

        scorer = WatchlistScorer(
            watchlist_path=watchlist,
            lookback_days=lookback_days,
            account=account,
            skip_trend=skip_trend,
            skip_lightgbm=(model == "kronos"),
        )

        results = scorer.run()

        if not results.get("watchlist"):
            logger.warning("Watchlist is empty or not found: {}", watchlist)
            console.print("[yellow]自选股为空或配置文件不存在[/yellow]")
            raise typer.Exit(1)

        has_kronos = not results.get("kronos_scores", pd.DataFrame()).empty
        has_lightgbm = not results.get("lightgbm_scores", pd.DataFrame()).empty

        if not has_kronos and not has_lightgbm:
            logger.error("No scores generated from any model")
            console.print("[red]无法生成任何模型的打分[/red]")
            console.print()
            console.print("[yellow]请检查:[/yellow]")
            console.print("  1. Qlib数据是否已安装 (python big-a/scripts/update_data.py)")
            console.print("  2. 模型文件是否存在 (Kronos: ~/.cache/huggingface, LightGBM: output/lightgbm_model.pkl)")
            raise typer.Exit(1)

        if no_portfolio:
            results["portfolio"] = pd.DataFrame()

        if output:
            output_file = open(output, "w", encoding="utf-8")
            output_console = Console(file=output_file)
        else:
            output_console = console

        format_report(results, output_console)

        if output:
            output_file.close()
            logger.info("Report saved to {}", output)
            console.print(f"[green]报告已保存到: {output}[/green]")

    except FileNotFoundError as e:
        logger.error("File not found: {}", e)
        console.print(f"[red]文件不存在: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Failed to generate report: {}", e)
        console.print(f"[red]生成报告失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def score(
    watchlist: str = typer.Option("configs/watchlist.yaml", "--watchlist", "-w", help="自选股配置文件路径"),
    model: str = typer.Option("kronos", "--model", "-m", help="使用的模型: all, kronos, lightgbm"),
    local_only: bool = typer.Option(True, "--local-only", help="仅使用本地缓存的Kronos模型"),
) -> None:
    """Quick scoring only, no full report."""
    from big_a.report.scorer import WatchlistScorer
    from big_a.report.formatter import format_scores_table
    import pandas as pd

    console = Console()

    try:
        logger.info("Starting quick scoring")
        logger.info("Watchlist: {}", watchlist)
        logger.info("Model: {}", model)

        scorer = WatchlistScorer(watchlist_path=watchlist)

        watchlist_data = scorer.load_watchlist()

        if not watchlist_data:
            logger.warning("Watchlist is empty or not found: {}", watchlist)
            console.print("[yellow]自选股为空或配置文件不存在[/yellow]")
            raise typer.Exit(1)

        from big_a.qlib_config import init_qlib
        init_qlib()

        instruments = list(watchlist_data.keys())

        results = {
            "watchlist": watchlist_data,
            "kronos_scores": pd.DataFrame(),
            "lightgbm_scores": pd.DataFrame(),
        }

        if model in ["all", "kronos"]:
            logger.info("Scoring with Kronos...")
            results["kronos_scores"] = scorer.score_kronos(instruments)

        if model in ["all", "lightgbm"]:
            logger.info("Scoring with LightGBM...")
            results["lightgbm_scores"] = scorer.score_lightgbm(instruments)

        format_scores_table(results, console)

        has_kronos = not results["kronos_scores"].empty
        has_lightgbm = not results["lightgbm_scores"].empty

        if not has_kronos and not has_lightgbm:
            logger.error("No scores generated from any model")
            console.print()
            console.print("[yellow]请检查:[/yellow]")
            console.print("  1. Qlib数据是否已安装")
            console.print("  2. 模型文件是否存在")
            raise typer.Exit(1)

    except FileNotFoundError as e:
        logger.error("File not found: {}", e)
        console.print(f"[red]文件不存在: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Failed to score: {}", e)
        console.print(f"[red]打分失败: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
