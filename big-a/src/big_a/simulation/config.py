from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

from big_a.simulation.types import SimulationConfig


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Load simulation config from YAML file and return SimulationConfig.

    Flattens nested YAML structure into SimulationConfig flat fields.
    Raises FileNotFoundError with clear message if file missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Simulation config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Flatten nested YAML into SimulationConfig fields
    sim = raw.get("simulation", {})
    cb = sim.get("circuit_breaker", {})
    ex = sim.get("exchange", {})
    uni = sim.get("universe", {})
    llm = raw.get("llm", {})
    fusion = raw.get("fusion", {})
    storage = raw.get("storage", {})

    config = SimulationConfig(
        initial_capital=sim.get("initial_capital", 500000),
        account=sim.get("account", "sim_001"),
        max_weight=sim.get("max_weight", 0.25),
        stop_loss=sim.get("stop_loss", -0.08),
        rebalance_freq=sim.get("rebalance_freq", 5),
        topk=sim.get("topk", 5),
        n_drop=sim.get("n_drop", 1),
        risk_degree=sim.get("risk_degree", 0.95),
        max_total_loss=cb.get("max_total_loss", -0.20),
        min_cash=cb.get("min_cash", 10000),
        open_cost=ex.get("open_cost", 0.0005),
        close_cost=ex.get("close_cost", 0.0015),
        min_commission=ex.get("min_commission", 5.0),
        limit_threshold=ex.get("limit_threshold", 0.095),
        deal_price=ex.get("deal_price", "close"),
        universe_base_pool=uni.get("base_pool", "csi300"),
        universe_watchlist=uni.get("watchlist", "configs/watchlist.yaml"),
        llm_enabled=llm.get("enabled", True),
        llm_api_base=llm.get("api_base", "https://api.minimaxi.com/anthropic"),
        llm_model=llm.get("model", "MiniMax-M2.7"),
        llm_timeout=llm.get("timeout", 30),
        llm_max_retries=llm.get("max_retries", 3),
        llm_temperature=llm.get("temperature", 0.3),
        llm_max_tokens=llm.get("max_tokens", 4096),
        fusion_llm_weight=fusion.get("llm_weight", 0.5),
        fusion_quant_weight=fusion.get("quant_weight", 0.5),
        storage_base_dir=storage.get("base_dir", "data/simulation"),
        storage_trades_dir=storage.get("trades_dir", "data/simulation/trades"),
        storage_decisions_dir=storage.get("decisions_dir", "data/simulation/decisions"),
        storage_snapshots_dir=storage.get("snapshots_dir", "data/simulation/snapshots"),
    )

    logger.info(f"Loaded simulation config from {path}: capital={config.initial_capital}")
    return config