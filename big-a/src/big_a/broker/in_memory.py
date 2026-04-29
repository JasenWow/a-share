from __future__ import annotations

from datetime import datetime

from loguru import logger

from big_a.simulation.types import (
    Order,
    OrderSide,
    OrderStatus,
    Portfolio,
    Position,
    TradeRecord,
)


class InMemoryBroker:
    def __init__(
        self,
        initial_cash: float,
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        min_commission: float = 5.0,
        limit_threshold: float = 0.095,
    ) -> None:
        self._cash = initial_cash
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_commission = min_commission
        self._limit_threshold = limit_threshold
        self._positions: dict[str, Position] = {}
        self._orders: list[Order] = []
        self._trade_records: list[TradeRecord] = []
        self._prices: dict[str, float] = {}

    def submit_order(self, order: Order) -> Order:
        stock = order.stock_code
        side = order.side
        quantity = order.quantity
        price = self._prices.get(stock, order.price)
        order.price = price

        if side == OrderSide.BUY:
            total_cost = quantity * price
            commission = max(total_cost * self._open_cost, self._min_commission)
            total_required = total_cost + commission

            if self._cash < total_required:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Buy order rejected: insufficient cash {self._cash:.2f} < {total_required:.2f}")
                return order

            self._cash -= total_required

            if stock in self._positions:
                pos = self._positions[stock]
                new_qty = pos.quantity + quantity
                pos.avg_price = (pos.avg_price * pos.quantity + price * quantity) / new_qty
                pos.quantity = new_qty
                pos.current_price = price
            else:
                from datetime import date
                self._positions[stock] = Position(
                    stock_code=stock,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_date=date.today().isoformat(),
                )

            order.status = OrderStatus.FILLED
            order.commission = commission
            order.filled_at = datetime.now()

        elif side == OrderSide.SELL:
            if stock not in self._positions:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Sell order rejected: no position for {stock}")
                return order

            pos = self._positions[stock]
            if pos.quantity < quantity:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Sell order rejected: insufficient holdings {pos.quantity} < {quantity}")
                return order

            proceeds = quantity * price
            commission = max(proceeds * self._close_cost, self._min_commission)
            realized_pnl = (price - pos.avg_price) * quantity - commission

            self._cash += proceeds - commission

            pos.quantity -= quantity
            pos.realized_pnl += realized_pnl
            pos.current_price = price

            if pos.quantity == 0:
                del self._positions[stock]

            order.status = OrderStatus.FILLED
            order.commission = commission
            order.filled_at = datetime.now()

        self._orders.append(order)

        trade_record = TradeRecord(
            order_id=order.id,
            stock_code=stock,
            side=side,
            quantity=quantity,
            fill_price=price,
            commission=commission,
            timestamp=datetime.now(),
        )
        self._trade_records.append(trade_record)

        logger.debug(f"Order {order.id} {side.value} {quantity}@{price} -> {order.status.value}")
        return order

    def cancel_order(self, order_id: str) -> Order:
        for order in self._orders:
            if order.id == order_id:
                if order.status != OrderStatus.PENDING:
                    raise ValueError(f"Cannot cancel order {order_id} with status {order.status.value}")
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order {order_id} cancelled")
                return order
        raise ValueError(f"Order {order_id} not found")

    def get_position(self, stock_code: str) -> Position | None:
        return self._positions.get(stock_code)

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_balance(self) -> float:
        return self._cash

    def get_portfolio(self) -> Portfolio:
        total_position_value = sum(p.market_value for p in self._positions.values())
        total_value = self._cash + total_position_value
        return Portfolio(
            cash=self._cash,
            positions=dict(self._positions),
            total_value=total_value,
            daily_pnl=0.0,
            updated_at=datetime.now(),
        )

    def update_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)
        for stock, pos in self._positions.items():
            if stock in self._prices:
                pos.current_price = self._prices[stock]
                pos.unrealized_pnl = (pos.current_price - pos.avg_price) * pos.quantity