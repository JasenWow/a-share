from __future__ import annotations

from typing import Protocol

from big_a.simulation.types import Order, Position, Portfolio


class BrokerInterface(Protocol):
    """Abstract broker interface. InMemory implementation for simulation, QMT for real trading."""

    def submit_order(self, order: Order) -> Order:
        """Submit an order and return the updated order with final status."""

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending order by ID and return the updated order."""

    def get_position(self, stock_code: str) -> Position | None:
        """Get current position for a stock code, or None if not held."""

    def get_all_positions(self) -> dict[str, Position]:
        """Get all current positions as a dict keyed by stock_code."""

    def get_balance(self) -> float:
        """Get current cash balance."""

    def get_portfolio(self) -> Portfolio:
        """Get current portfolio state with all positions and total value."""

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all held positions, recalculating unrealized PnL."""