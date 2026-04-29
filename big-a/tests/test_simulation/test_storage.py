from __future__ import annotations

import json
from datetime import datetime

import pytest

from big_a.simulation.storage import SimulationStorage
from big_a.simulation.types import (
    Order,
    OrderSide,
    OrderType,
    Portfolio,
    Position,
    SignalSource,
    SignalStrength,
    StockSignal,
    TradeRecord,
    TradingDecision,
)


@pytest.fixture
def storage(tmp_path):
    return SimulationStorage(
        base_dir=str(tmp_path / "simulation"),
        trades_dir=str(tmp_path / "simulation" / "trades"),
        decisions_dir=str(tmp_path / "simulation" / "decisions"),
        snapshots_dir=str(tmp_path / "simulation" / "snapshots"),
    )


class TestSaveAndLoadTrade:
    def test_save_and_load_trade(self, storage):
        trade = TradeRecord(
            order_id="abc123",
            stock_code="000001",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=10.5,
            commission=0.5,
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        )
        storage.save_trade(trade, "2024-01-15")
        trades = storage.load_trades()
        assert len(trades) == 1
        assert trades[0].order_id == "abc123"
        assert trades[0].stock_code == "000001"
        assert trades[0].side == OrderSide.BUY
        assert trades[0].quantity == 100
        assert trades[0].fill_price == 10.5
        assert trades[0].commission == 0.5
        assert trades[0].timestamp == datetime(2024, 1, 15, 10, 30, 0)


class TestSaveMultipleTrades:
    def test_save_multiple_trades(self, storage):
        trades = [
            TradeRecord(
                order_id=f"ord{i}",
                stock_code="000001",
                side=OrderSide.BUY,
                quantity=100,
                fill_price=10.0,
                commission=0.5,
                timestamp=datetime(2024, 1, 15, 10, i),
            )
            for i in range(3)
        ]
        storage.save_trades(trades, "2024-01-15")
        loaded = storage.load_trades()
        assert len(loaded) == 3


class TestJsonlAppend:
    def test_jsonl_append(self, storage):
        trade1 = TradeRecord(
            order_id="a",
            stock_code="000001",
            side=OrderSide.BUY,
            quantity=1,
            fill_price=1.0,
            commission=0.0,
            timestamp=datetime(2024, 1, 1),
        )
        trade2 = TradeRecord(
            order_id="b",
            stock_code="000001",
            side=OrderSide.BUY,
            quantity=2,
            fill_price=2.0,
            commission=0.0,
            timestamp=datetime(2024, 1, 2),
        )
        storage.save_trade(trade1, "2024-01-01")
        storage.save_trade(trade2, "2024-01-02")
        trade3 = TradeRecord(
            order_id="c",
            stock_code="000001",
            side=OrderSide.BUY,
            quantity=3,
            fill_price=3.0,
            commission=0.0,
            timestamp=datetime(2024, 1, 3),
        )
        trade4 = TradeRecord(
            order_id="d",
            stock_code="000001",
            side=OrderSide.BUY,
            quantity=4,
            fill_price=4.0,
            commission=0.0,
            timestamp=datetime(2024, 1, 4),
        )
        trade5 = TradeRecord(
            order_id="e",
            stock_code="000001",
            side=OrderSide.BUY,
            quantity=5,
            fill_price=5.0,
            commission=0.0,
            timestamp=datetime(2024, 1, 5),
        )
        storage.save_trades([trade3, trade4, trade5], "2024-01-05")
        loaded = storage.load_trades()
        assert len(loaded) == 5


class TestSaveAndLoadSnapshot:
    def test_save_and_load_snapshot(self, storage):
        portfolio = Portfolio(
            cash=100000.0,
            positions={
                "000001": Position(
                    stock_code="000001",
                    quantity=100,
                    avg_price=50.0,
                    current_price=55.0,
                    unrealized_pnl=500.0,
                    realized_pnl=0.0,
                    entry_date="2024-01-10",
                ),
                "000002": Position(
                    stock_code="000002",
                    quantity=200,
                    avg_price=30.0,
                    current_price=28.0,
                    unrealized_pnl=-400.0,
                    realized_pnl=0.0,
                    entry_date="2024-01-11",
                ),
            },
            daily_pnl=100.0,
            updated_at=datetime(2024, 1, 15, 12, 0, 0),
        )
        storage.save_snapshot(portfolio, "2024-01-15")
        loaded = storage.load_latest_snapshot()
        assert loaded is not None
        assert loaded.cash == 100000.0
        assert len(loaded.positions) == 2
        assert "000001" in loaded.positions
        assert "000002" in loaded.positions


class TestLoadLatestSnapshot:
    def test_load_latest_snapshot(self, storage):
        p1 = Portfolio(cash=100.0, positions={}, updated_at=datetime(2024, 1, 1))
        p2 = Portfolio(cash=200.0, positions={}, updated_at=datetime(2024, 1, 2))
        p3 = Portfolio(cash=300.0, positions={}, updated_at=datetime(2024, 1, 3))
        storage.save_snapshot(p1, "2024-01-01")
        storage.save_snapshot(p2, "2024-01-02")
        storage.save_snapshot(p3, "2024-01-03")
        latest = storage.load_latest_snapshot()
        assert latest is not None
        assert latest.cash == 300.0

    def test_load_latest_snapshot_empty(self, storage):
        assert storage.load_latest_snapshot() is None


class TestSaveAndLoadDecision:
    def test_save_and_load_decision(self, storage):
        decision = TradingDecision(
            date="2024-01-15",
            signals=[
                StockSignal(
                    stock_code="000001",
                    score=0.75,
                    signal=SignalStrength.BUY,
                    source=SignalSource.kronos,
                    reasoning="Strong momentum",
                )
            ],
            orders=[
                Order(
                    stock_code="000001",
                    side=OrderSide.BUY,
                    quantity=100,
                    price=10.0,
                )
            ],
            reasoning="Based on signals",
        )
        storage.save_decision(decision, "2024-01-15")
        decisions_dir = storage.decisions_dir / "2024-01-15.json"
        assert decisions_dir.exists()
        with open(decisions_dir, encoding="utf-8") as f:
            data = json.load(f)
        assert data["date"] == "2024-01-15"
        assert len(data["signals"]) == 1
        assert len(data["orders"]) == 1


class TestEnsureDirsCreatesStructure:
    def test_ensure_dirs_creates_structure(self, tmp_path):
        s = SimulationStorage(
            base_dir=str(tmp_path / "sim"),
            trades_dir=str(tmp_path / "sim" / "trades"),
            decisions_dir=str(tmp_path / "sim" / "decisions"),
            snapshots_dir=str(tmp_path / "sim" / "snapshots"),
        )
        assert (tmp_path / "sim").is_dir()
        assert (tmp_path / "sim" / "trades").is_dir()
        assert (tmp_path / "sim" / "decisions").is_dir()
        assert (tmp_path / "sim" / "snapshots").is_dir()


class TestLoadTradesDateFilter:
    def test_load_trades_date_filter(self, storage):
        for i, date in enumerate(["2024-01-01", "2024-01-05", "2024-01-10"]):
            trade = TradeRecord(
                order_id=f"ord{i}",
                stock_code="000001",
                side=OrderSide.BUY,
                quantity=10,
                fill_price=10.0,
                commission=0.0,
                timestamp=datetime(2024, 1, 1),
            )
            storage.save_trade(trade, date)
        filtered = storage.load_trades(start_date="2024-01-03", end_date="2024-01-07")
        assert len(filtered) == 1
        assert filtered[0].order_id == "ord1"


class TestPortfolioSerializationRoundtrip:
    def test_portfolio_serialization_roundtrip(self, storage):
        portfolio = Portfolio(
            cash=500000.0,
            positions={
                "000001": Position(
                    stock_code="000001",
                    quantity=100,
                    avg_price=10.0,
                    current_price=12.0,
                    unrealized_pnl=200.0,
                    realized_pnl=50.0,
                    entry_date="2024-01-01",
                )
            },
            daily_pnl=250.0,
            updated_at=datetime(2024, 6, 1, 9, 30, 0),
        )
        storage.save_snapshot(portfolio, "2024-06-01")
        loaded = storage.load_latest_snapshot()
        assert loaded is not None
        assert loaded.cash == 500000.0
        assert "000001" in loaded.positions
        assert loaded.positions["000001"].quantity == 100
        assert loaded.positions["000001"].avg_price == 10.0