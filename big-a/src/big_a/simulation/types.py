from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class SignalStrength(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalSource(str, Enum):
    kronos = "kronos"
    lightgbm = "lightgbm"
    llm = "llm"
    fused = "fused"


class Order(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    stock_code: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    filled_at: datetime | None = None
    commission: float = 0.0


class Position(BaseModel):
    stock_code: str
    quantity: int = Field(ge=0)
    avg_price: float = Field(gt=0)
    current_price: float = Field(gt=0)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_date: str

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        return (self.current_price - self.avg_price) / self.avg_price


class Portfolio(BaseModel):
    cash: float = Field(ge=0)
    positions: dict[str, Position] = {}
    total_value: float = 0.0
    daily_pnl: float = 0.0
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def total_position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "total_value", self.cash + self.total_position_value)


class TradeRecord(BaseModel):
    order_id: str
    stock_code: str
    side: OrderSide
    quantity: int
    fill_price: float
    commission: float
    timestamp: datetime


class StockSignal(BaseModel):
    stock_code: str
    score: float = Field(ge=-1.0, le=1.0)
    signal: SignalStrength
    source: SignalSource
    reasoning: str = ""

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))


class TradingDecision(BaseModel):
    date: str
    signals: list[StockSignal] = []
    orders: list[Order] = []
    reasoning: str = ""


class SimulationConfig(BaseModel):
    initial_capital: float = Field(default=500000.0, gt=0)
    account: str = "sim_001"
    max_weight: float = Field(default=0.25, gt=0, le=1.0)
    stop_loss: float = -0.08
    rebalance_freq: int = Field(default=5, gt=0)
    topk: int = Field(default=5, gt=0)
    n_drop: int = Field(default=1, ge=0)
    risk_degree: float = Field(default=0.95, gt=0, le=1.0)
    max_total_loss: float = -0.20
    min_cash: float = Field(default=10000.0, ge=0)
    open_cost: float = Field(default=0.0005, ge=0)
    close_cost: float = Field(default=0.0015, ge=0)
    min_commission: float = Field(default=5.0, ge=0)
    limit_threshold: float = Field(default=0.095, gt=0)
    deal_price: str = "close"
    universe_base_pool: str = "csi300"
    universe_watchlist: str = "configs/watchlist.yaml"
    llm_enabled: bool = True
    llm_api_base: str = "https://api.minimaxi.com/anthropic"
    llm_model: str = "MiniMax-M2.7"
    llm_timeout: int = Field(default=30, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_temperature: float = Field(default=0.3, ge=0, le=2.0)
    llm_max_tokens: int = Field(default=4096, gt=0)
    fusion_llm_weight: float = Field(default=0.5, ge=0, le=1.0)
    fusion_quant_weight: float = Field(default=0.5, ge=0, le=1.0)
    storage_base_dir: str = "data/simulation"
    storage_trades_dir: str = "data/simulation/trades"
    storage_decisions_dir: str = "data/simulation/decisions"
    storage_snapshots_dir: str = "data/simulation/snapshots"
