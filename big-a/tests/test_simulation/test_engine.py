from __future__ import annotations

from datetime import datetime

import pytest

from big_a.broker.in_memory import InMemoryBroker
from big_a.simulation.engine import SimulationEngine
from big_a.simulation.types import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    SignalStrength,
    SignalSource,
    SimulationConfig,
    StockSignal,
)


def make_signal(stock_code: str, score: float) -> StockSignal:
    return StockSignal(
        stock_code=stock_code,
        score=score,
        signal=SignalStrength.BUY if score > 0 else SignalStrength.SELL,
        source=SignalSource.fused,
    )


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig(
        initial_capital=500000.0,
        max_weight=0.25,
        stop_loss=-0.08,
        rebalance_freq=5,
        topk=5,
        max_total_loss=-0.20,
    )


@pytest.fixture
def broker(config: SimulationConfig) -> InMemoryBroker:
    return InMemoryBroker(
        initial_cash=config.initial_capital,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_commission=config.min_commission,
    )


@pytest.fixture
def engine(config: SimulationConfig, broker: InMemoryBroker) -> SimulationEngine:
    return SimulationEngine(config=config, broker=broker)


def test_initialize(engine: SimulationEngine, config: SimulationConfig):
    engine.initialize()
    assert engine.get_day_count() == 0
    portfolio = engine.get_portfolio()
    assert portfolio.cash == config.initial_capital
    assert len(portfolio.positions) == 0


def test_first_day_buy(engine: SimulationEngine, broker: InMemoryBroker, config: SimulationConfig):
    engine.initialize()

    prices = {"stock_a": 100.0, "stock_b": 50.0, "stock_c": 25.0}
    signals = [
        make_signal("stock_a", 0.8),
        make_signal("stock_b", 0.6),
        make_signal("stock_c", 0.4),
    ]

    portfolio = engine.run_daily("2024-01-02", signals, prices)

    assert engine.get_day_count() == 1
    assert len(portfolio.positions) == 3
    assert portfolio.cash < config.initial_capital


def test_stop_loss_trigger(engine: SimulationEngine, broker: InMemoryBroker, config: SimulationConfig):
    engine.initialize()

    prices_day1 = {"stock_a": 100.0}
    signals_day1 = [make_signal("stock_a", 0.8)]
    engine.run_daily("2024-01-02", signals_day1, prices_day1)

    positions_before = broker.get_all_positions()
    assert "stock_a" in positions_before
    assert positions_before["stock_a"].quantity > 0

    prices_day2 = {"stock_a": 91.0}
    signals_day2 = [make_signal("stock_a", 0.5)]
    portfolio_day2 = engine.run_daily("2024-01-03", signals_day2, prices_day2)

    assert "stock_a" not in portfolio_day2.positions


def test_stop_loss_no_trigger(engine: SimulationEngine, broker: InMemoryBroker, config: SimulationConfig):
    engine.initialize()

    prices_day1 = {"stock_a": 100.0}
    signals_day1 = [make_signal("stock_a", 0.8)]
    engine.run_daily("2024-01-02", signals_day1, prices_day1)

    prices_day2 = {"stock_a": 93.0}
    signals_day2 = [make_signal("stock_a", 0.5)]
    portfolio_day2 = engine.run_daily("2024-01-03", signals_day2, prices_day2)

    assert "stock_a" in portfolio_day2.positions
    assert portfolio_day2.positions["stock_a"].quantity > 0


def test_circuit_breaker(engine: SimulationEngine, broker: InMemoryBroker, config: SimulationConfig):
    engine.initialize()

    prices_day1 = {"stock_a": 100.0, "stock_b": 100.0, "stock_c": 100.0}
    signals_day1 = [
        make_signal("stock_a", 0.8),
        make_signal("stock_b", 0.6),
        make_signal("stock_c", 0.4),
    ]
    engine.run_daily("2024-01-02", signals_day1, prices_day1)

    positions_day1 = broker.get_all_positions()
    assert len(positions_day1) == 3

    prices_day2 = {"stock_a": 93.0, "stock_b": 93.0, "stock_c": 93.0}
    signals_day2 = [make_signal("stock_a", 0.5)]
    portfolio_day2 = engine.run_daily("2024-01-03", signals_day2, prices_day2)

    assert len(portfolio_day2.positions) == 3
    assert all(p.current_price == 93.0 for p in portfolio_day2.positions.values())


def test_rebalance_frequency(engine: SimulationEngine, broker: InMemoryBroker):
    config = SimulationConfig(
        initial_capital=500000.0,
        max_weight=0.25,
        stop_loss=-0.08,
        rebalance_freq=3,
        topk=5,
        max_total_loss=-0.20,
    )
    engine = SimulationEngine(config=config, broker=broker)
    engine.initialize()

    prices = {"stock_a": 100.0, "stock_b": 50.0}
    signals = [make_signal("stock_a", 0.8), make_signal("stock_b", 0.6)]

    engine.run_daily("2024-01-02", signals, prices)
    assert engine.get_day_count() == 1

    engine.run_daily("2024-01-03", signals, prices)
    assert engine.get_day_count() == 2

    engine.run_daily("2024-01-04", signals, prices)
    assert engine.get_day_count() == 3


def test_target_weights_capped(engine: SimulationEngine, broker: InMemoryBroker, config: SimulationConfig):
    engine.initialize()

    prices = {"stock_a": 100.0, "stock_b": 100.0}
    signals = [make_signal("stock_a", 0.8), make_signal("stock_b", 0.6)]

    top_signals = sorted(signals, key=lambda s: s.score, reverse=True)[: config.topk]
    weights = engine._calculate_target_weights(top_signals)

    assert weights["stock_a"] == 0.25
    assert weights["stock_b"] == 0.25


def test_five_day_simulation(engine: SimulationEngine, broker: InMemoryBroker, config: SimulationConfig):
    engine.initialize()

    prices_day1 = {"stock_a": 100.0, "stock_b": 50.0, "stock_c": 25.0}
    signals_day1 = [
        make_signal("stock_a", 0.8),
        make_signal("stock_b", 0.6),
        make_signal("stock_c", 0.4),
    ]
    engine.run_daily("2024-01-02", signals_day1, prices_day1)
    assert engine.get_day_count() == 1

    prices_day2 = {"stock_a": 105.0, "stock_b": 52.0, "stock_c": 26.0}
    signals_day2 = [make_signal("stock_a", 0.7)]
    engine.run_daily("2024-01-03", signals_day2, prices_day2)
    assert engine.get_day_count() == 2

    prices_day3 = {"stock_a": 102.0, "stock_b": 51.0, "stock_c": 24.5}
    signals_day3 = [make_signal("stock_a", 0.6)]
    engine.run_daily("2024-01-04", signals_day3, prices_day3)
    assert engine.get_day_count() == 3

    prices_day4 = {"stock_a": 100.0, "stock_b": 48.0, "stock_c": 25.0}
    signals_day4 = [
        make_signal("stock_a", 0.8),
        make_signal("stock_b", 0.7),
        make_signal("stock_c", 0.5),
    ]
    engine.run_daily("2024-01-05", signals_day4, prices_day4)
    assert engine.get_day_count() == 4

    prices_day5 = {"stock_a": 98.0, "stock_b": 49.0, "stock_c": 24.0}
    signals_day5 = [make_signal("stock_a", 0.5)]
    engine.run_daily("2024-01-06", signals_day5, prices_day5)
    assert engine.get_day_count() == 5

    snapshots = engine.get_daily_snapshots()
    assert len(snapshots) == 5


def test_get_portfolio(engine: SimulationEngine, broker: InMemoryBroker, config: SimulationConfig):
    engine.initialize()
    prices = {"stock_a": 100.0}
    signals = [make_signal("stock_a", 0.8)]
    engine.run_daily("2024-01-02", signals, prices)

    portfolio = engine.get_portfolio()
    assert isinstance(portfolio, Portfolio)
    assert "stock_a" in portfolio.positions


def test_get_day_count(engine: SimulationEngine, broker: InMemoryBroker, config: SimulationConfig):
    engine.initialize()
    assert engine.get_day_count() == 0

    prices = {"stock_a": 100.0}
    signals = [make_signal("stock_a", 0.8)]
    engine.run_daily("2024-01-02", signals, prices)
    assert engine.get_day_count() == 1
