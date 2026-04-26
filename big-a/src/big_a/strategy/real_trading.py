"""Real trading strategy with stop-loss, weekly rebalancing, and position weight caps."""
from __future__ import annotations

import copy
from typing import Any

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.contrib.strategy import TopkDropoutStrategy


class RealTradingStrategy(TopkDropoutStrategy):
    """Custom strategy for small capital real trading with stop-loss and position limits."""

    def __init__(
        self,
        *,
        stop_loss: float = -0.08,
        max_weight: float = 0.25,
        rebalance_freq: int = 5,
        **kwargs: Any,
    ):
        """Initialize RealTradingStrategy.

        Parameters
        ----------
        stop_loss : float, default -0.08
            Stop-loss threshold. Sell if loss exceeds this percentage.
        max_weight : float, default 0.25
            Maximum position weight for any single stock.
        rebalance_freq : int, default 5
            Rebalance frequency in trading days (5 = weekly).
        **kwargs : Any
            Additional arguments passed to TopkDropoutStrategy
            (topk, n_drop, signal, risk_degree, etc.).
        """
        super().__init__(**kwargs)
        self.stop_loss = stop_loss
        self.max_weight = max_weight
        self.rebalance_freq = rebalance_freq
        self._step_count = 0
        self._entry_price: dict[str, float] = {}

    def generate_trade_decision(self, execute_result=None):
        """Generate trade decisions with stop-loss and scheduled rebalancing.

        Phase 1: Stop-loss check (every day)
        Phase 2: Scheduled rebalancing (every N days)
        """
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_temp = copy.deepcopy(self.trade_position)

        stop_loss_orders = []
        for code in list(current_temp.get_stock_list()):
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            ):
                continue

            current_price = self.trade_exchange.get_deal_price(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            )
            entry = self._entry_price.get(code, current_temp.get_stock_price(code))

            if entry is not None and entry > 0 and current_price is not None:
                pnl = (current_price - entry) / entry
                if pnl <= self.stop_loss:
                    sell_amount = current_temp.get_stock_amount(code)
                    sell_order = Order(
                        stock_id=code,
                        amount=sell_amount,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=Order.SELL,
                    )
                    if self.trade_exchange.check_order(sell_order):
                        stop_loss_orders.append(sell_order)
                        trade_val, trade_cost, _ = self.trade_exchange.deal_order(
                            sell_order, position=current_temp
                        )
                        self._entry_price.pop(code, None)

        self._step_count += 1
        if self._step_count % self.rebalance_freq == 0:
            rebalance_decision = super().generate_trade_decision(execute_result)

            buy_orders = [
                o for o in rebalance_decision.get_decision() if o.direction == Order.BUY
            ]
            sell_orders = [
                o for o in rebalance_decision.get_decision() if o.direction == Order.SELL
            ]

            filtered_buy_orders = []
            total_value = current_temp.calculate_value()

            for order in buy_orders:
                buy_price = self.trade_exchange.get_deal_price(
                    stock_id=order.stock_id,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.BUY,
                )
                if buy_price is None:
                    continue

                buy_value = order.amount * buy_price
                existing_amount = current_temp.get_stock_amount(order.stock_id)
                existing_value = existing_amount * current_temp.get_stock_price(order.stock_id)
                existing_weight = existing_value / total_value if total_value > 0 else 0
                new_weight = buy_value / total_value if total_value > 0 else 0

                if existing_weight + new_weight > self.max_weight:
                    max_value = self.max_weight * total_value
                    allowed_buy_value = max_value - existing_value
                    if allowed_buy_value > 0:
                        allowed_amount = allowed_buy_value / buy_price
                        factor = self.trade_exchange.get_factor(
                            stock_id=order.stock_id,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        )
                        order.amount = self.trade_exchange.round_amount_by_trade_unit(
                            allowed_amount, factor
                        )

                filtered_buy_orders.append(order)

            all_orders = stop_loss_orders + sell_orders + filtered_buy_orders
            return TradeDecisionWO(all_orders, self)

        return TradeDecisionWO(stop_loss_orders, self)

    def post_exe_step(self, execute_result=None):
        """Track entry prices for newly bought stocks.

        Parameters
        ----------
        execute_result : list, optional
            List of execution result tuples (order, trade_val, trade_cost, trade_price).
        """
        if execute_result is None:
            return

        for result in execute_result:
            if hasattr(result, "__iter__") and len(result) >= 4:
                order, trade_val, trade_cost, trade_price = result[:4]
                if (
                    hasattr(order, "direction")
                    and order.direction == Order.BUY
                    and trade_price is not None
                ):
                    self._entry_price[order.stock_id] = trade_price
