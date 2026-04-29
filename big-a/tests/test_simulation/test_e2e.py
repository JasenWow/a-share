"""End-to-end integration tests for simulated trading system."""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from big_a.broker.in_memory import InMemoryBroker
from big_a.simulation.engine import SimulationEngine
from big_a.simulation.fusion import SignalFusion
from big_a.simulation.storage import SimulationStorage
from big_a.simulation.types import (
    Order,
    OrderSide,
    Portfolio,
    Position,
    SignalSource,
    SignalStrength,
    SimulationConfig,
    StockSignal,
    TradeRecord,
    TradingDecision,
)


def make_signals(data: list[tuple[str, float]]) -> list[StockSignal]:
    return [
        StockSignal(
            stock_code=code,
            score=score,
            signal=(
                SignalStrength.BUY if score > 0.3
                else SignalStrength.HOLD if score > -0.3
                else SignalStrength.SELL
            ),
            source=SignalSource.fused,
        )
        for code, score in data
    ]


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig(
        initial_capital=500000.0,
        max_weight=0.25,
        stop_loss=-0.08,
        rebalance_freq=5,
        topk=3,
        max_total_loss=-0.20,
        open_cost=0.0005,
        close_cost=0.0015,
        min_commission=5.0,
    )


@pytest.fixture
def broker(config: SimulationConfig) -> InMemoryBroker:
    return InMemoryBroker(
        initial_cash=config.initial_capital,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_commission=config.min_commission,
    )


def test_three_day_simulation(config: SimulationConfig, broker: InMemoryBroker):
    engine = SimulationEngine(config=config, broker=broker)
    engine.initialize()

    prices_d1 = {"A": 100.0, "B": 50.0, "C": 80.0}
    signals_d1 = make_signals([("A", 0.8), ("B", 0.6), ("C", 0.4)])
    portfolio_d1 = engine.run_daily("2024-01-02", signals_d1, prices_d1)

    assert engine.get_day_count() == 1
    assert len(portfolio_d1.positions) == 3, f"Expected 3 positions on day 1, got {len(portfolio_d1.positions)}"
    assert portfolio_d1.cash < config.initial_capital
    assert "A" in portfolio_d1.positions
    assert "B" in portfolio_d1.positions
    assert "C" in portfolio_d1.positions

    snapshots_d1 = engine.get_daily_snapshots()
    assert len(snapshots_d1) == 1

    prices_d2 = {"A": 110.0, "B": 45.0, "C": 85.0}
    signals_d2 = make_signals([("A", 0.8), ("B", 0.6), ("C", 0.4)])
    portfolio_d2 = engine.run_daily("2024-01-03", signals_d2, prices_d2)

    assert engine.get_day_count() == 2

    pos_a_d2 = portfolio_d2.positions["A"]
    pos_c_d2 = portfolio_d2.positions["C"]
    assert pos_a_d2.unrealized_pnl > 0, f"A should be profitable, got {pos_a_d2.unrealized_pnl}"
    assert pos_c_d2.unrealized_pnl > 0, f"C should be profitable, got {pos_c_d2.unrealized_pnl}"
    assert "B" not in portfolio_d2.positions, (
        f"B should be stopped out on day 2 (entry=50, price=45, pnl_pct=-10% <= -8%), "
        f"but still in portfolio. Positions: {list(portfolio_d2.positions.keys())}"
    )

    snapshots_d2 = engine.get_daily_snapshots()
    assert len(snapshots_d2) == 2

    prices_d3 = {"A": 115.0, "C": 90.0, "D": 60.0}
    signals_d3 = make_signals([("A", 0.9), ("D", 0.7)])
    portfolio_d3 = engine.run_daily("2024-01-04", signals_d3, prices_d3)

    assert engine.get_day_count() == 3

    snapshots_d3 = engine.get_daily_snapshots()
    assert len(snapshots_d3) == 3

    assert "A" in portfolio_d3.positions
    assert "C" in portfolio_d3.positions


def test_llm_failure_degradation(config: SimulationConfig, broker: InMemoryBroker):
    fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)

    llm_signals: list[StockSignal] = []
    quant_scores = {"A": 0.8, "B": 0.6, "C": 0.4}

    fused = fusion.fuse(llm_signals=llm_signals, quant_scores=quant_scores)

    assert len(fused) == 3, f"Expected 3 fused signals, got {len(fused)}"
    codes = {s.stock_code for s in fused}
    assert codes == {"A", "B", "C"}
    assert all(s.source == SignalSource.fused for s in fused)

    score_map = {s.stock_code: s.score for s in fused}
    assert score_map["A"] > score_map["B"] > score_map["C"]

    engine = SimulationEngine(config=config, broker=broker)
    engine.initialize()

    prices = {"A": 100.0, "B": 50.0, "C": 80.0}
    portfolio = engine.run_daily("2024-01-02", fused, prices)

    assert len(portfolio.positions) == 3
    assert portfolio.cash < config.initial_capital


def test_persistence_recovery(config: SimulationConfig, broker: InMemoryBroker, tmp_path):
    engine1 = SimulationEngine(config=config, broker=broker)
    engine1.initialize()

    prices_d1 = {"A": 100.0, "B": 50.0, "C": 80.0}
    signals_d1 = make_signals([("A", 0.8), ("B", 0.6), ("C", 0.4)])
    portfolio_d1 = engine1.run_daily("2024-01-02", signals_d1, prices_d1)

    assert engine1.get_day_count() == 1
    assert len(portfolio_d1.positions) == 3

    storage = SimulationStorage(
        base_dir=str(tmp_path / "simulation"),
        trades_dir=str(tmp_path / "simulation" / "trades"),
        decisions_dir=str(tmp_path / "simulation" / "decisions"),
        snapshots_dir=str(tmp_path / "simulation" / "snapshots"),
    )

    storage.save_snapshot(portfolio_d1, "2024-01-02")

    decision = TradingDecision(
        date="2024-01-02",
        signals=signals_d1,
        orders=[],
        reasoning="Test decision",
    )
    storage.save_decision(decision, "2024-01-02")

    loaded_snapshot = storage.load_latest_snapshot()
    assert loaded_snapshot is not None
    assert loaded_snapshot.cash == portfolio_d1.cash
    assert len(loaded_snapshot.positions) == len(portfolio_d1.positions)

    decision_path = tmp_path / "simulation" / "decisions" / "2024-01-02.json"
    with open(decision_path, encoding="utf-8") as f:
        loaded_decision_data = json.load(f)
    assert loaded_decision_data["date"] == "2024-01-02"
    assert len(loaded_decision_data["signals"]) == 3

    broker2 = InMemoryBroker(
        initial_cash=config.initial_capital,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_commission=config.min_commission,
    )
    engine2 = SimulationEngine(config=config, broker=broker2)
    engine2.initialize()

    prices_d2 = {"A": 110.0, "B": 55.0, "C": 85.0}
    signals_d2 = make_signals([("A", 0.8), ("B", 0.6), ("C", 0.4)])
    portfolio_d2 = engine2.run_daily("2024-01-03", signals_d2, prices_d2)

    assert engine2.get_day_count() == 1
    assert len(portfolio_d2.positions) == 3


def test_full_pipeline_with_fusion(config: SimulationConfig, broker: InMemoryBroker):
    fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)

    quant_scores = {"A": 0.9, "B": 0.6, "C": 0.4, "D": 0.2}

    llm_signals = [
        StockSignal(stock_code="A", score=0.7, signal=SignalStrength.BUY, source=SignalSource.llm),
        StockSignal(stock_code="B", score=0.5, signal=SignalStrength.BUY, source=SignalSource.llm),
        StockSignal(stock_code="C", score=0.3, signal=SignalStrength.HOLD, source=SignalSource.llm),
    ]

    fused_signals = fusion.fuse(llm_signals=llm_signals, quant_scores=quant_scores)

    assert len(fused_signals) == 4
    assert all(s.source == SignalSource.fused for s in fused_signals)

    engine = SimulationEngine(config=config, broker=broker)
    engine.initialize()

    prices = {"A": 100.0, "B": 50.0, "C": 80.0, "D": 60.0}
    portfolio = engine.run_daily("2024-01-02", fused_signals, prices)

    assert len(portfolio.positions) <= 3
    assert portfolio.cash < config.initial_capital


def test_storage_roundtrip(config: SimulationConfig, broker: InMemoryBroker, tmp_path):
    storage = SimulationStorage(
        base_dir=str(tmp_path / "simulation"),
        trades_dir=str(tmp_path / "simulation" / "trades"),
        decisions_dir=str(tmp_path / "simulation" / "decisions"),
        snapshots_dir=str(tmp_path / "simulation" / "snapshots"),
    )

    engine = SimulationEngine(config=config, broker=broker)
    engine.initialize()

    prices = {"A": 100.0, "B": 50.0, "C": 80.0}
    signals = make_signals([("A", 0.8), ("B", 0.6), ("C", 0.4)])
    portfolio = engine.run_daily("2024-01-02", signals, prices)

    trade = TradeRecord(
        order_id="test_order_001",
        stock_code="A",
        side=OrderSide.BUY,
        quantity=1000,
        fill_price=100.0,
        commission=5.0,
        timestamp=datetime(2024, 1, 2, 10, 0, 0),
    )
    storage.save_trade(trade, "2024-01-02")

    storage.save_snapshot(portfolio, "2024-01-02")

    decision = TradingDecision(
        date="2024-01-02",
        signals=signals,
        orders=[Order(stock_code="A", side=OrderSide.BUY, quantity=1000, price=100.0)],
        reasoning="Test e2e decision",
    )
    storage.save_decision(decision, "2024-01-02")

    loaded_trades = storage.load_trades()
    assert len(loaded_trades) == 1
    assert loaded_trades[0].order_id == "test_order_001"
    assert loaded_trades[0].stock_code == "A"
    assert loaded_trades[0].side == OrderSide.BUY
    assert loaded_trades[0].quantity == 1000

    loaded_snapshot = storage.load_latest_snapshot()
    assert loaded_snapshot is not None
    assert loaded_snapshot.cash == portfolio.cash
    assert len(loaded_snapshot.positions) == len(portfolio.positions)
    for code in portfolio.positions:
        assert code in loaded_snapshot.positions

    decision_path = tmp_path / "simulation" / "decisions" / "2024-01-02.json"
    with open(decision_path, encoding="utf-8") as f:
        loaded_decision_data = json.load(f)
    assert loaded_decision_data["date"] == "2024-01-02"
    assert len(loaded_decision_data["signals"]) == 3
    assert len(loaded_decision_data["orders"]) == 1
    assert loaded_decision_data["orders"][0]["stock_code"] == "A"