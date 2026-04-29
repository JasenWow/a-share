from __future__ import annotations

from datetime import datetime
from loguru import logger

from big_a.simulation.types import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Portfolio,
    StockSignal,
    SignalStrength,
    SignalSource,
    TradeRecord,
    SimulationConfig,
)
from big_a.broker.base import BrokerInterface


class SimulationEngine:
    """Core simulation engine for portfolio rebalancing and trade execution."""

    def __init__(self, config: SimulationConfig, broker: BrokerInterface) -> None:
        self.config = config
        self.broker = broker
        self._day_count: int = 0
        self._daily_snapshots: list[Portfolio] = []
        self._trade_history: list[TradeRecord] = []

    def initialize(self) -> None:
        """Set broker cash to config.initial_capital and reset day count."""
        self.broker = self.broker
        # Reset broker cash to initial capital
        # Broker doesn't expose set_cash, so we recreate portfolio state via orders
        # For InMemoryBroker specifically, we rely on it being created with correct initial_cash
        # For other brokers, the caller should inject the correct broker instance.
        self._day_count = 0
        self._daily_snapshots.clear()
        self._trade_history.clear()
        logger.info(f"Simulation initialized with capital={self.config.initial_capital}")

    def run_daily(
        self, trading_date: str, signals: list[StockSignal], prices: dict[str, float]
    ) -> Portfolio:
        """
        Execute daily simulation:
        1. Increment day count
        2. Update prices
        3. Check stop-loss on all positions
        4. Check circuit breaker (max total loss)
        5. Rebalance if rebalance day
        6. Save portfolio snapshot
        """
        self._day_count += 1
        self.broker.update_prices(prices)

        # Stop-loss check: sell positions with PnL <= stop_loss
        for stock_code, pos in self.broker.get_all_positions().items():
            if pos.pnl_pct <= self.config.stop_loss:
                logger.info(
                    f"Day {self._day_count} [{trading_date}]: Stop-loss triggered for "
                    f"{stock_code} pnl_pct={pos.pnl_pct:.4f} <= {self.config.stop_loss}"
                )
                order = Order(
                    stock_code=stock_code,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                    price=pos.current_price,
                )
                self.broker.submit_order(order)

        # Circuit breaker: skip rebalancing if total loss exceeds threshold
        portfolio = self.broker.get_portfolio()
        total_pnl = portfolio.total_value - self.config.initial_capital
        total_pnl_pct = total_pnl / self.config.initial_capital
        circuit_broken = total_pnl_pct <= self.config.max_total_loss

        if circuit_broken:
            logger.warning(
                f"Day {self._day_count} [{trading_date}]: Circuit breaker triggered - "
                f"total_pnl_pct={total_pnl_pct:.4f} <= {self.config.max_total_loss}. "
                f"Rebalancing skipped."
            )
        else:
            should_rebalance = (
                self._day_count == 1 or self._day_count % self.config.rebalance_freq == 1
            )

            if should_rebalance:
                self._rebalance(signals, portfolio.total_value, prices, trading_date)

        # Save portfolio snapshot
        portfolio = self.broker.get_portfolio()
        self._daily_snapshots.append(portfolio)
        logger.debug(
            f"Day {self._day_count} [{trading_date}]: portfolio total_value="
            f"{portfolio.total_value:.2f}, cash={portfolio.cash:.2f}"
        )
        return portfolio

    def _rebalance(
        self, signals: list[StockSignal], total_value: float, prices: dict[str, float], trading_date: str
    ) -> None:
        sorted_signals = sorted(signals, key=lambda s: s.score, reverse=True)
        top_signals = sorted_signals[: self.config.topk]

        if not top_signals:
            logger.info(f"Day {self._day_count} [{trading_date}]: No top signals, skip rebalance")
            return

        target_weights = self._calculate_target_weights(top_signals)

        target_stocks = set(target_weights.keys())
        current_stocks = set(self.broker.get_all_positions().keys())

        stocks_to_sell = current_stocks - target_stocks
        for stock_code in stocks_to_sell:
            pos = self.broker.get_position(stock_code)
            if pos and pos.quantity > 0:
                order = Order(
                    stock_code=stock_code,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                    price=pos.current_price,
                )
                result = self.broker.submit_order(order)
                if result.status == OrderStatus.FILLED:
                    logger.info(
                        f"Day {self._day_count} [{trading_date}]: SELL {stock_code} "
                        f"qty={result.quantity}@{result.price}"
                    )

        for stock_code, weight in target_weights.items():
            target_value = total_value * weight
            pos = self.broker.get_position(stock_code)
            current_value = pos.market_value if pos else 0.0
            delta_value = target_value - current_value

            price = prices.get(stock_code)
            if price is None:
                continue

            if delta_value > 0:
                quantity = int(delta_value / price)
                if quantity > 0:
                    order = Order(
                        stock_code=stock_code,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        price=price,
                    )
                    result = self.broker.submit_order(order)
                    if result.status == OrderStatus.FILLED:
                        logger.info(
                            f"Day {self._day_count} [{trading_date}]: BUY {stock_code} "
                            f"qty={result.quantity}@{result.price}"
                        )
            elif delta_value < 0:
                if pos and pos.quantity > 0:
                    quantity = int(-delta_value / price)
                    quantity = min(quantity, pos.quantity)
                    if quantity > 0:
                        order = Order(
                            stock_code=stock_code,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=quantity,
                            price=price,
                        )
                        result = self.broker.submit_order(order)
                        if result.status == OrderStatus.FILLED:
                            logger.info(
                                f"Day {self._day_count} [{trading_date}]: SELL {stock_code} "
                                f"qty={result.quantity}@{result.price}"
                            )

    def _calculate_target_weights(self, signals: list[StockSignal]) -> dict[str, float]:
        """
        Calculate equal target weights for topk signals, capped at max_weight.
        Returns dict[stock_code, weight].
        """
        if not signals:
            return {}

        n = len(signals)
        base_weight = 1.0 / n

        weights: dict[str, float] = {}
        for signal in signals:
            weight = min(base_weight, self.config.max_weight)
            weights[signal.stock_code] = weight

        return weights

    def get_portfolio(self) -> Portfolio:
        """Get current portfolio from broker."""
        return self.broker.get_portfolio()

    def get_trade_history(self) -> list[TradeRecord]:
        """Return accumulated trade history (not yet populated - broker maintains records)."""
        # InMemoryBroker maintains _trade_records but doesn't expose them directly
        # We can get them from broker if it provides an interface, otherwise return empty
        # For now, return empty - in a full implementation, broker would expose this
        return []

    def get_daily_snapshots(self) -> list[Portfolio]:
        """Return list of daily portfolio snapshots."""
        return list(self._daily_snapshots)

    def get_day_count(self) -> int:
        """Return current day count."""
        return self._day_count
