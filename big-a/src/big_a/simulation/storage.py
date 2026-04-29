from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from big_a.simulation.types import (
    Order,
    Portfolio,
    Position,
    StockSignal,
    TradeRecord,
    TradingDecision,
)


class SimulationStorage:
    """Persist trades, decisions, snapshots, and run logs as JSON/JSONL files."""

    def __init__(
        self,
        base_dir: str = "data/simulation",
        trades_dir: str = "data/simulation/trades",
        decisions_dir: str = "data/simulation/decisions",
        snapshots_dir: str = "data/simulation/snapshots",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.trades_dir = Path(trades_dir)
        self.decisions_dir = Path(decisions_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create all storage directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    # ── Trades ───────────────────────────────────────────────────────────────

    def save_trade(self, trade: TradeRecord, date: str) -> None:
        """Append a single trade as a JSON line to trades_dir/{date}.jsonl."""
        line = self._serialize_trade(trade)
        filepath = self.trades_dir / f"{date}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def save_trades(self, trades: list[TradeRecord], date: str) -> None:
        """Append multiple trades to the same JSONL file."""
        filepath = self.trades_dir / f"{date}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            for trade in trades:
                f.write(json.dumps(self._serialize_trade(trade)) + "\n")

    def load_trades(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> list[TradeRecord]:
        """Load all trades from JSONL files, optionally filtered by date range."""
        trades: list[TradeRecord] = []
        for filepath in sorted(self.trades_dir.glob("*.jsonl")):
            date_str = filepath.stem
            if start_date is not None and date_str < start_date:
                continue
            if end_date is not None and date_str > end_date:
                continue
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    trades.append(self._deserialize_trade(json.loads(line)))
        return trades

    # ── Decisions ────────────────────────────────────────────────────────────

    def save_decision(self, decision: TradingDecision, date: str) -> None:
        filepath = self.decisions_dir / f"{date}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._serialize_decision(decision), f, default=str)

    # ── Snapshots ────────────────────────────────────────────────────────────

    def save_snapshot(self, portfolio: Portfolio, date: str) -> None:
        filepath = self.snapshots_dir / f"{date}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._serialize_portfolio(portfolio), f, default=str)

    def load_latest_snapshot(self) -> Portfolio | None:
        """Load the most recent portfolio snapshot by filename ordering, or None if empty."""
        files = sorted(self.snapshots_dir.glob("*.json"))
        if not files:
            return None
        with open(files[-1], encoding="utf-8") as f:
            return self._deserialize_portfolio(json.load(f))

    # ── Run logs ─────────────────────────────────────────────────────────────

    def save_run_log(
        self,
        run_id: str,
        start_time: datetime,
        end_time: datetime,
        status: str,
        summary: dict,
    ) -> None:
        """Save a run log to base_dir/runs/{run_id}.json."""
        run_dir = self.base_dir / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        filepath = run_dir / f"{run_id}.json"
        data = {
            "run_id": run_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "status": status,
            "summary": summary,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    # ── Serialization helpers ────────────────────────────────────────────────

    def _serialize_portfolio(self, p: Portfolio) -> dict:
        return {
            "cash": p.cash,
            "positions": {
                code: {
                    "stock_code": pos.stock_code,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "entry_date": pos.entry_date,
                }
                for code, pos in p.positions.items()
            },
            "total_value": p.total_value,
            "daily_pnl": p.daily_pnl,
            "updated_at": p.updated_at.isoformat(),
        }

    def _deserialize_portfolio(self, data: dict) -> Portfolio:
        positions = {
            code: Position(**pos_data) for code, pos_data in data["positions"].items()
        }
        return Portfolio(
            cash=data["cash"],
            positions=positions,
            total_value=data["total_value"],
            daily_pnl=data["daily_pnl"],
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def _serialize_trade(self, t: TradeRecord) -> dict:
        return {
            "order_id": t.order_id,
            "stock_code": t.stock_code,
            "side": t.side.value,
            "quantity": t.quantity,
            "fill_price": t.fill_price,
            "commission": t.commission,
            "timestamp": t.timestamp.isoformat(),
        }

    def _deserialize_trade(self, data: dict) -> TradeRecord:
        from big_a.simulation.types import OrderSide

        return TradeRecord(
            order_id=data["order_id"],
            stock_code=data["stock_code"],
            side=OrderSide(data["side"]),
            quantity=data["quantity"],
            fill_price=data["fill_price"],
            commission=data["commission"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    def _serialize_decision(self, d: TradingDecision) -> dict:
        return {
            "date": d.date,
            "signals": [s.model_dump() for s in d.signals],
            "orders": [o.model_dump() for o in d.orders],
            "reasoning": d.reasoning,
        }

    def _deserialize_decision(self, data: dict) -> TradingDecision:
        return TradingDecision(
            date=data["date"],
            signals=[StockSignal(**s) for s in data["signals"]],
            orders=[Order(**o) for o in data["orders"]],
            reasoning=data["reasoning"],
        )