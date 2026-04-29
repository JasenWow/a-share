from __future__ import annotations

import pytest

from big_a.broker.in_memory import InMemoryBroker
from big_a.simulation.types import Order, OrderSide, OrderStatus


class TestInMemoryBroker:
    def test_initial_state(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        assert broker.get_balance() == 500000.0
        assert broker.get_all_positions() == {}
        assert broker.get_position("000001") is None

    def test_buy_order_creates_position(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        broker.update_prices({"000001": 1800.0})
        order = Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=1800.0)
        filled = broker.submit_order(order)
        assert filled.status == OrderStatus.FILLED
        pos = broker.get_position("000001")
        assert pos is not None
        assert pos.quantity == 100
        assert pos.avg_price == 1800.0

    def test_buy_order_commission(self):
        broker = InMemoryBroker(initial_cash=500000.0, open_cost=0.0005, min_commission=5.0)
        broker.update_prices({"000001": 1800.0})
        initial_cash = broker.get_balance()
        order = Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=1800.0)
        filled = broker.submit_order(order)
        assert filled.status == OrderStatus.FILLED
        expected_cost = 100 * 1800.0
        expected_commission = max(expected_cost * 0.0005, 5.0)
        assert filled.commission == expected_commission
        assert broker.get_balance() == pytest.approx(initial_cash - expected_cost - expected_commission)

    def test_sell_order_reduces_position(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        broker.update_prices({"000001": 1800.0})
        buy_order = Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=1800.0)
        broker.submit_order(buy_order)
        initial_cash = broker.get_balance()
        sell_order = Order(stock_code="000001", side=OrderSide.SELL, quantity=50, price=1850.0)
        filled = broker.submit_order(sell_order)
        assert filled.status == OrderStatus.FILLED
        pos = broker.get_position("000001")
        assert pos is not None
        assert pos.quantity == 50
        assert broker.get_balance() > initial_cash

    def test_sell_order_commission_with_tax(self):
        broker = InMemoryBroker(initial_cash=500000.0, close_cost=0.0015, min_commission=5.0)
        broker.update_prices({"000001": 1800.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=1800.0))
        sell_order = Order(stock_code="000001", side=OrderSide.SELL, quantity=100, price=1850.0)
        filled = broker.submit_order(sell_order)
        assert filled.status == OrderStatus.FILLED
        proceeds = 100 * 1800.0
        commission = max(proceeds * 0.0015, 5.0)
        assert filled.commission == commission

    def test_sell_full_position_removes(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        broker.update_prices({"000001": 1800.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=1800.0))
        broker.submit_order(Order(stock_code="000001", side=OrderSide.SELL, quantity=100, price=1850.0))
        assert broker.get_position("000001") is None

    def test_sell_insufficient_holdings(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        broker.update_prices({"000001": 1800.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=50, price=1800.0))
        sell_order = Order(stock_code="000001", side=OrderSide.SELL, quantity=100, price=1850.0)
        filled = broker.submit_order(sell_order)
        assert filled.status == OrderStatus.REJECTED

    def test_buy_insufficient_funds(self):
        broker = InMemoryBroker(initial_cash=10000.0)
        broker.update_prices({"000001": 1800.0})
        order = Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=1800.0)
        filled = broker.submit_order(order)
        assert filled.status == OrderStatus.REJECTED

    def test_buy_increases_avg_price(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        broker.update_prices({"000001": 100.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=100.0))
        broker.update_prices({"000001": 110.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=110.0))
        pos = broker.get_position("000001")
        assert pos is not None
        assert pos.avg_price == 105.0
        assert pos.quantity == 200

    def test_update_prices_unrealized_pnl(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        broker.update_prices({"000001": 100.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=100.0))
        broker.update_prices({"000001": 110.0})
        pos = broker.get_position("000001")
        assert pos is not None
        assert pos.unrealized_pnl == 1000.0

    def test_realized_pnl_on_sell(self):
        broker = InMemoryBroker(initial_cash=500000.0, close_cost=0.0015, min_commission=5.0)
        broker.update_prices({"000001": 100.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=100.0))
        broker.update_prices({"000001": 110.0})
        sell_order = Order(stock_code="000001", side=OrderSide.SELL, quantity=50, price=110.0)
        filled = broker.submit_order(sell_order)
        assert filled.status == OrderStatus.FILLED
        proceeds = 50 * 110.0
        commission = max(proceeds * 0.0015, 5.0)
        expected_pnl = (110.0 - 100.0) * 50 - commission
        pos = broker.get_position("000001")
        assert pos.realized_pnl == pytest.approx(expected_pnl)

    def test_get_portfolio_total_value(self):
        broker = InMemoryBroker(initial_cash=500000.0, open_cost=0.0005, min_commission=5.0)
        broker.update_prices({"000001": 100.0, "000002": 50.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=100.0))
        broker.submit_order(Order(stock_code="000002", side=OrderSide.BUY, quantity=1000, price=50.0))
        portfolio = broker.get_portfolio()
        pos1_val = 100 * 100.0
        pos2_val = 1000 * 50.0
        commission1 = max(pos1_val * 0.0005, 5.0)
        commission2 = max(pos2_val * 0.0005, 5.0)
        expected_cash = 500000.0 - pos1_val - commission1 - pos2_val - commission2
        expected_value = expected_cash + pos1_val + pos2_val
        assert portfolio.total_value == expected_value
        assert portfolio.cash == pytest.approx(expected_cash)

    def test_cancel_pending_order(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        order = Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=1800.0)
        broker.submit_order(order)
        pos = broker.get_position("000001")
        assert pos is not None
        assert pos.quantity == 100

    def test_min_commission_applied(self):
        broker = InMemoryBroker(initial_cash=500000.0, open_cost=0.0003, min_commission=5.0)
        broker.update_prices({"000001": 10.0})
        order = Order(stock_code="000001", side=OrderSide.BUY, quantity=10, price=10.0)
        filled = broker.submit_order(order)
        assert filled.status == OrderStatus.FILLED
        assert filled.commission == 5.0

    def test_cancel_non_pending_order_raises(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        broker.update_prices({"000001": 1800.0})
        order = Order(stock_code="000001", side=OrderSide.BUY, quantity=100, price=1800.0)
        broker.submit_order(order)
        with pytest.raises(ValueError, match="Cannot cancel"):
            broker.cancel_order(order.id)

    def test_cancel_nonexistent_order_raises(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        with pytest.raises(ValueError, match="not found"):
            broker.cancel_order("nonexistent_id")

    def test_multiple_buy_orders_same_stock(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        broker.update_prices({"000001": 100.0})
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=50, price=100.0))
        broker.submit_order(Order(stock_code="000001", side=OrderSide.BUY, quantity=50, price=100.0))
        pos = broker.get_position("000001")
        assert pos is not None
        assert pos.quantity == 100
        assert pos.avg_price == 100.0

    def test_portfolio_updated_at_timestamp(self):
        broker = InMemoryBroker(initial_cash=500000.0)
        portfolio = broker.get_portfolio()
        assert portfolio.updated_at is not None