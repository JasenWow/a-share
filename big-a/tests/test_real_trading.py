"""Unit tests for RealTradingStrategy using mocks (no real Qlib data required)."""
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from big_a.strategy.real_trading import RealTradingStrategy
from qlib.backtest.decision import Order, OrderDir


@pytest.fixture
def mock_strategy():
    """Create a RealTradingStrategy with mocked Qlib infrastructure."""
    with patch("qlib.contrib.strategy.TopkDropoutStrategy.__init__", return_value=None):
        strategy = RealTradingStrategy(
            stop_loss=-0.08,
            max_weight=0.25,
            rebalance_freq=5,
        )
    # Mock all required attributes (properties need PropertyMock)
    type(strategy).trade_calendar = PropertyMock(return_value=MagicMock())
    type(strategy).trade_position = PropertyMock(return_value=MagicMock())
    type(strategy).trade_exchange = PropertyMock(return_value=MagicMock())
    strategy.signal = MagicMock()
    strategy.risk_degree = 0.95
    return strategy


class TestStopLoss:
    """Test stop-loss functionality."""

    def test_stop_loss_trigger(self, mock_strategy):
        """Stock with loss > 8% should generate sell order."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 1
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = ["SH600000"]
        s.trade_position.get_stock_amount.return_value = 1000.0
        s.trade_position.get_stock_price.return_value = 92.0
        s.trade_exchange.is_stock_tradable.return_value = True
        s.trade_exchange.get_deal_price.return_value = 91.0  # 9% loss from 100
        s.trade_exchange.check_order.return_value = True
        s.trade_exchange.deal_order.return_value = (91000.0, 150.0, 91.0)
        s._entry_price = {"SH600000": 100.0}

        decision = s.generate_trade_decision()

        orders = decision.get_decision()
        assert len(orders) == 1
        assert orders[0].direction == Order.SELL
        assert orders[0].stock_id == "SH600000"
        assert orders[0].amount == 1000.0
        assert "SH600000" not in s._entry_price

    def test_stop_loss_not_triggered(self, mock_strategy):
        """Stock with small loss should NOT generate sell order."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 1
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = ["SH600000"]
        s.trade_position.get_stock_amount.return_value = 1000.0
        s.trade_position.get_stock_price.return_value = 97.0
        s.trade_exchange.is_stock_tradable.return_value = True
        s.trade_exchange.get_deal_price.return_value = 97.0  # only -3% from 100
        s._entry_price = {"SH600000": 100.0}

        decision = s.generate_trade_decision()

        orders = decision.get_decision()
        assert len(orders) == 0
        assert "SH600000" in s._entry_price

    def test_limit_stock_not_traded(self, mock_strategy):
        """Stock at limit should NOT be sold for stop loss."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 1
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = ["SH600000"]
        s.trade_position.get_stock_amount.return_value = 1000.0
        s.trade_position.get_stock_price.return_value = 90.0
        s.trade_exchange.is_stock_tradable.return_value = False  # at limit!
        s._entry_price = {"SH600000": 100.0}

        decision = s.generate_trade_decision()

        orders = decision.get_decision()
        assert len(orders) == 0

    def test_stop_loss_uses_position_price_as_fallback(self, mock_strategy):
        """Use position price when entry price not tracked."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 1
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = ["SH600000"]
        s.trade_position.get_stock_amount.return_value = 1000.0
        s.trade_position.get_stock_price.return_value = 100.0  # fallback entry price
        s.trade_exchange.is_stock_tradable.return_value = True
        s.trade_exchange.get_deal_price.return_value = 91.0  # 9% loss
        s.trade_exchange.check_order.return_value = True
        s.trade_exchange.deal_order.return_value = (91000.0, 150.0, 91.0)
        s._entry_price = {}  # no tracked entry price

        decision = s.generate_trade_decision()

        orders = decision.get_decision()
        assert len(orders) == 1
        assert orders[0].direction == Order.SELL

    def test_stop_loss_skips_invalid_order(self, mock_strategy):
        """Skip order if exchange check fails."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 1
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = ["SH600000"]
        s.trade_position.get_stock_amount.return_value = 1000.0
        s.trade_position.get_stock_price.return_value = 100.0
        s.trade_exchange.is_stock_tradable.return_value = True
        s.trade_exchange.get_deal_price.return_value = 91.0
        s.trade_exchange.check_order.return_value = False  # order invalid!
        s._entry_price = {"SH600000": 100.0}

        decision = s.generate_trade_decision()

        orders = decision.get_decision()
        assert len(orders) == 0
        assert "SH600000" in s._entry_price  # entry price NOT removed


class TestRebalance:
    """Test rebalancing functionality."""

    def test_rebalance_on_correct_day(self, mock_strategy):
        """On rebalance day (every 5th step), parent strategy is called."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 5
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = []
        s.trade_position.calculate_value.return_value = 1000000.0
        s._step_count = 4

        # Mock parent's generate_trade_decision
        mock_parent_decision = MagicMock()
        mock_parent_decision.get_decision.return_value = []
        with patch.object(
            type(s).__bases__[0], "generate_trade_decision", return_value=mock_parent_decision
        ):
            decision = s.generate_trade_decision()

        assert s._step_count == 5
        assert mock_parent_decision.get_decision.call_count == 2

    def test_no_rebalance_on_non_rebalance_day(self, mock_strategy):
        """Non-rebalance day should not call parent strategy."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 2
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = []
        s._step_count = 1

        decision = s.generate_trade_decision()

        assert s._step_count == 2
        orders = decision.get_decision()
        assert len(orders) == 0

    def test_rebalance_on_multiple_of_freq(self, mock_strategy):
        """Rebalance on step 10 (2nd rebalance day)."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 10
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = []
        s.trade_position.calculate_value.return_value = 1000000.0
        s._step_count = 9

        mock_parent_decision = MagicMock()
        mock_parent_decision.get_decision.return_value = []
        with patch.object(
            type(s).__bases__[0], "generate_trade_decision", return_value=mock_parent_decision
        ):
            decision = s.generate_trade_decision()

        assert s._step_count == 10
        assert mock_parent_decision.get_decision.call_count == 2


class TestWeightCap:
    """Test position weight capping."""

    def test_weight_cap(self, mock_strategy):
        """Buy orders are capped at max_weight."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 5
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = ["SH600000"]
        s.trade_position.get_stock_amount.return_value = 10000.0
        s.trade_position.get_stock_price.return_value = 20.0  # existing: 200k
        s.trade_position.calculate_value.return_value = 1000000.0  # total: 1M
        s._step_count = 4

        # Mock parent returns a large buy order that would exceed max_weight
        buy_order = Order(
            stock_id="SH600000",
            amount=20000.0,  # would buy 400k value at 20/share
            start_time="2024-01-15",
            end_time="2024-01-15",
            direction=Order.BUY,
        )
        mock_parent_decision = MagicMock()
        mock_parent_decision.get_decision.return_value = [buy_order]

        s.trade_exchange.get_deal_price.return_value = 20.0
        s.trade_exchange.get_factor.return_value = 100.0
        s.trade_exchange.round_amount_by_trade_unit.side_effect = lambda x, f: int(x / f) * f

        with patch.object(
            type(s).__bases__[0], "generate_trade_decision", return_value=mock_parent_decision
        ):
            decision = s.generate_trade_decision()

        orders = decision.get_decision()
        buy_orders = [o for o in orders if o.direction == Order.BUY]
        assert len(buy_orders) == 1

        # Existing: 200k / 1M = 20%
        # Max allowed: 25% * 1M = 250k
        # Allowed buy: 250k - 200k = 50k
        # Allowed amount: 50k / 20 = 2500 shares
        # Rounded to 100 shares: 2500
        assert buy_orders[0].amount == 2500.0

    def test_weight_cap_with_no_existing_position(self, mock_strategy):
        """Weight cap works for new position."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 5
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = []
        s.trade_position.get_stock_amount.return_value = 0.0
        s.trade_position.get_stock_price.return_value = 0.0
        s.trade_position.calculate_value.return_value = 1000000.0
        s._step_count = 4

        # Large buy order
        buy_order = Order(
            stock_id="SH600001",
            amount=50000.0,  # 1M value at 20/share
            start_time="2024-01-15",
            end_time="2024-01-15",
            direction=Order.BUY,
        )
        mock_parent_decision = MagicMock()
        mock_parent_decision.get_decision.return_value = [buy_order]

        s.trade_exchange.get_deal_price.return_value = 20.0
        s.trade_exchange.get_factor.return_value = 100.0
        s.trade_exchange.round_amount_by_trade_unit.side_effect = lambda x, f: int(x / f) * f

        with patch.object(
            type(s).__bases__[0], "generate_trade_decision", return_value=mock_parent_decision
        ):
            decision = s.generate_trade_decision()

        orders = decision.get_decision()
        buy_orders = [o for o in orders if o.direction == Order.BUY]
        assert len(buy_orders) == 1

        # Max allowed: 25% * 1M = 250k
        # Allowed amount: 250k / 20 = 12500 shares
        # Rounded: 12500
        assert buy_orders[0].amount == 12500.0

    def test_weight_cap_skips_if_no_buy_price(self, mock_strategy):
        """Skip buy order if cannot get buy price."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 5
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = []
        s.trade_position.calculate_value.return_value = 1000000.0
        s._step_count = 4

        buy_order = Order(
            stock_id="SH600001",
            amount=1000.0,
            start_time="2024-01-15",
            end_time="2024-01-15",
            direction=Order.BUY,
        )
        mock_parent_decision = MagicMock()
        mock_parent_decision.get_decision.return_value = [buy_order]

        s.trade_exchange.get_deal_price.return_value = None  # no price!

        with patch.object(
            type(s).__bases__[0], "generate_trade_decision", return_value=mock_parent_decision
        ):
            decision = s.generate_trade_decision()

        orders = decision.get_decision()
        buy_orders = [o for o in orders if o.direction == Order.BUY]
        assert len(buy_orders) == 0


class TestEntryPriceTracking:
    """Test entry price tracking."""

    def test_entry_price_tracking(self, mock_strategy):
        """post_exe_step should track buy entry prices."""
        s = mock_strategy
        mock_order = MagicMock()
        mock_order.direction = Order.BUY
        mock_order.stock_id = "SH600000"

        execute_result = [(mock_order, 100000.0, 50.0, 10.0)]
        s.post_exe_step(execute_result)

        assert s._entry_price.get("SH600000") == 10.0

    def test_post_exe_step_ignores_sells(self, mock_strategy):
        """Only BUY orders update entry prices."""
        s = mock_strategy
        mock_order = MagicMock()
        mock_order.direction = Order.SELL
        mock_order.stock_id = "SH600000"

        execute_result = [(mock_order, 100000.0, 50.0, 10.0)]
        s.post_exe_step(execute_result)

        assert "SH600000" not in s._entry_price

    def test_post_exe_step_with_none_result(self, mock_strategy):
        """post_exe_step(None) should not crash."""
        s = mock_strategy
        s.post_exe_step(None)  # Should not raise exception

        assert len(s._entry_price) == 0

    def test_post_exe_step_with_empty_list(self, mock_strategy):
        """post_exe_step([]) should not crash."""
        s = mock_strategy
        s.post_exe_step([])

        assert len(s._entry_price) == 0

    def test_post_exe_step_tracks_multiple_buys(self, mock_strategy):
        """Track entry prices for multiple buy orders."""
        s = mock_strategy
        mock_order1 = MagicMock()
        mock_order1.direction = Order.BUY
        mock_order1.stock_id = "SH600000"

        mock_order2 = MagicMock()
        mock_order2.direction = Order.BUY
        mock_order2.stock_id = "SH600001"

        execute_result = [
            (mock_order1, 100000.0, 50.0, 10.0),
            (mock_order2, 200000.0, 100.0, 20.0),
        ]
        s.post_exe_step(execute_result)

        assert s._entry_price.get("SH600000") == 10.0
        assert s._entry_price.get("SH600001") == 20.0


class TestCombinedScenarios:
    """Test complex scenarios combining multiple features."""

    def test_stop_loss_on_rebalance_day(self, mock_strategy):
        """Stop-loss should work on rebalance day too."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 5
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = ["SH600000"]
        s.trade_position.get_stock_amount.return_value = 1000.0
        s.trade_position.get_stock_price.return_value = 100.0
        s.trade_position.calculate_value.return_value = 1000000.0
        s.trade_exchange.is_stock_tradable.return_value = True
        s.trade_exchange.get_deal_price.return_value = 91.0
        s.trade_exchange.check_order.return_value = True
        s.trade_exchange.deal_order.return_value = (91000.0, 150.0, 91.0)
        s._step_count = 4
        s._entry_price = {"SH600000": 100.0}

        # Parent returns no orders
        mock_parent_decision = MagicMock()
        mock_parent_decision.get_decision.return_value = []

        with patch.object(
            type(s).__bases__[0], "generate_trade_decision", return_value=mock_parent_decision
        ):
            decision = s.generate_trade_decision()

        orders = decision.get_decision()
        # Should have stop-loss sell order
        assert len(orders) == 1
        assert orders[0].direction == Order.SELL
        assert "SH600000" not in s._entry_price

    def test_stop_loss_and_rebalance_orders_combined(self, mock_strategy):
        """Stop-loss and rebalance orders should be combined."""
        s = mock_strategy
        s.trade_calendar.get_trade_step.return_value = 5
        s.trade_calendar.get_step_time.return_value = ("2024-01-15", "2024-01-15")
        s.trade_position.get_stock_list.return_value = ["SH600000"]
        s.trade_position.get_stock_amount.return_value = 1000.0
        s.trade_position.get_stock_price.return_value = 100.0
        s.trade_position.calculate_value.return_value = 1000000.0

        # Stop-loss setup
        s.trade_exchange.is_stock_tradable.side_effect = lambda **kwargs: True
        s.trade_exchange.get_deal_price.side_effect = lambda **kwargs: 91.0
        s.trade_exchange.check_order.return_value = True
        s.trade_exchange.deal_order.return_value = (91000.0, 150.0, 91.0)
        s._step_count = 4
        s._entry_price = {"SH600000": 100.0}

        # Parent returns a buy order
        buy_order = Order(
            stock_id="SH600001",
            amount=1000.0,
            start_time="2024-01-15",
            end_time="2024-01-15",
            direction=Order.BUY,
        )
        mock_parent_decision = MagicMock()
        mock_parent_decision.get_decision.return_value = [buy_order]

        # Reset get_deal_price for the buy order
        call_count = [0]

        def get_deal_price_side_effect(**kwargs):
            call_count[0] += 1
            # First call is for stop-loss check (SH600000), return 91.0
            # Second call is for buy order (SH600001), return 20.0
            if call_count[0] == 1:
                return 91.0
            else:
                return 20.0

        s.trade_exchange.get_deal_price.side_effect = get_deal_price_side_effect

        with patch.object(
            type(s).__bases__[0], "generate_trade_decision", return_value=mock_parent_decision
        ):
            decision = s.generate_trade_decision()

        orders = decision.get_decision()
        # Should have stop-loss sell order + buy order
        assert len(orders) == 2
        sell_orders = [o for o in orders if o.direction == Order.SELL]
        buy_orders = [o for o in orders if o.direction == Order.BUY]
        assert len(sell_orders) == 1
        assert len(buy_orders) == 1
        assert sell_orders[0].stock_id == "SH600000"
        assert buy_orders[0].stock_id == "SH600001"
