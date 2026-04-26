"""Backtest engine wrapping Qlib's backtest_daily with A-share parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from qlib.strategy.base import BaseStrategy

# A-share default exchange parameters
DEFAULT_EXCHANGE_KWARGS: dict[str, Any] = {
    "freq": "day",
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}

DEFAULT_BACKTEST_KWARGS: dict[str, Any] = {
    "account": 100000000,
    "benchmark": "SH000300",
}


def run_backtest(
    signal: pd.DataFrame | pd.Series,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run backtest using Qlib's backtest_daily with TopkDropoutStrategy.

    Parameters
    ----------
    signal : pd.DataFrame or pd.Series
        Prediction signal with MultiIndex (datetime, instrument).
        If DataFrame, must have a 'score' column.
        If Series, will be converted to DataFrame with name 'score'.
    config : dict, optional
        Config dict matching configs/backtest/topk_csi300.yaml structure.
        If None, uses defaults.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - report: DataFrame with columns [return, bench, cost, turnover]
        - positions: dict mapping datetime -> Position object
    """
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy

    if isinstance(signal, pd.Series):
        signal = signal.to_frame("score")
    if "score" not in signal.columns:
        signal = signal.rename(columns={signal.columns[0]: "score"})

    cfg = config or {}
    bt_cfg = cfg.get("backtest", {})
    strat_cfg = cfg.get("strategy", {}).get("kwargs", {})

    start_time = bt_cfg.get("start_time")
    end_time = bt_cfg.get("end_time")

    if start_time is None:
        start_time = signal.index.get_level_values("datetime").min()
    if end_time is None:
        end_time = signal.index.get_level_values("datetime").max()

    account = bt_cfg.get("account", DEFAULT_BACKTEST_KWARGS["account"])
    benchmark = bt_cfg.get("benchmark", DEFAULT_BACKTEST_KWARGS["benchmark"])

    exchange_kwargs = {**DEFAULT_EXCHANGE_KWARGS, **bt_cfg.get("exchange_kwargs", {})}

    topk = strat_cfg.get("topk", 50)
    n_drop = strat_cfg.get("n_drop", 5)

    logger.info(
        f"Running backtest: {start_time} -> {end_time}, "
        f"topk={topk}, n_drop={n_drop}, account={account}, benchmark={benchmark}"
    )

    strategy = TopkDropoutStrategy(signal=signal, topk=topk, n_drop=n_drop)

    report, positions = backtest_daily(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        account=account,
        benchmark=benchmark,
        exchange_kwargs=exchange_kwargs,
    )
    logger.info(f"Backtest complete: {len(report)} trading days")
    return report, positions


def run_backtest_with_strategy(
    strategy: BaseStrategy,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run backtest with a pre-built strategy instance.

    Unlike run_backtest() which hardcodes TopkDropoutStrategy,
    this function accepts any BaseStrategy subclass (e.g. RealTradingStrategy).

    Parameters
    ----------
    strategy : BaseStrategy
        A configured strategy instance (e.g. RealTradingStrategy).
    config : dict, optional
        Config dict with backtest parameters (start_time, end_time, account, etc.).
        If None, uses defaults.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - report: DataFrame with columns [return, bench, cost, turnover]
        - positions: dict mapping datetime -> Position object
    """
    from qlib.contrib.evaluate import backtest_daily

    cfg = config or {}
    bt_cfg = cfg.get("backtest", {})

    start_time = bt_cfg.get("start_time")
    end_time = bt_cfg.get("end_time")

    account = bt_cfg.get("account", DEFAULT_BACKTEST_KWARGS["account"])
    benchmark = bt_cfg.get("benchmark", DEFAULT_BACKTEST_KWARGS["benchmark"])

    exchange_kwargs = {**DEFAULT_EXCHANGE_KWARGS, **bt_cfg.get("exchange_kwargs", {})}

    logger.info(
        f"Running backtest with custom strategy: {start_time} -> {end_time}, "
        f"account={account}, benchmark={benchmark}"
    )

    report, positions = backtest_daily(
        start_time=start_time,
        end_time=end_time,
        strategy=strategy,
        account=account,
        benchmark=benchmark,
        exchange_kwargs=exchange_kwargs,
    )

    logger.info(f"Backtest complete: {len(report)} trading days")
    return report, positions


def compute_analysis(report: pd.DataFrame) -> pd.DataFrame:
    """Compute risk analysis from backtest report.

    Parameters
    ----------
    report : pd.DataFrame
        Report DataFrame from run_backtest with columns [return, bench, cost].

    Returns
    -------
    pd.DataFrame
        Risk metrics including excess return with/without cost.
    """
    from qlib.contrib.evaluate import risk_analysis

    analysis: dict[str, pd.DataFrame] = {}
    analysis["excess_return_without_cost"] = risk_analysis(
        report["return"] - report["bench"]
    )
    analysis["excess_return_with_cost"] = risk_analysis(
        report["return"] - report["bench"] - report["cost"]
    )
    return pd.concat(analysis)


def load_backtest_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load backtest config from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config, relative to project root.
        Defaults to configs/backtest/topk_csi300.yaml.

    Returns
    -------
    dict
        Parsed config dictionary.
    """
    from big_a.config import load_config

    if config_path is None:
        config_path = "configs/backtest/topk_csi300.yaml"
    return load_config(config_path)
