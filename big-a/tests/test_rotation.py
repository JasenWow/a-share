"""Tests for rotation module - all mocked, no network or Qlib dependency."""
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from big_a.data.rotation import (
    _calc_sector_momentum,
    get_top_sectors,
    rank_sectors,
)


def create_price_data(stocks, days=25, start_price=10.0, trend=0.01):
    """Helper to create mock price data with a trend."""
    dates = pd.date_range(start=date.today() - timedelta(days=days), periods=days)
    index = pd.MultiIndex.from_product([stocks, dates], names=["instrument", "datetime"])

    prices = []
    for stock in stocks:
        stock_prices = [start_price]
        for i in range(1, days):
            price = stock_prices[-1] * (1 + trend + np.random.uniform(-0.01, 0.01))
            stock_prices.append(price)
        prices.extend(stock_prices)

    df = pd.DataFrame({"$close": prices}, index=index)
    return df


class TestSectorRanking:
    """Test sector ranking functionality."""

    def test_sector_ranking(self):
        """Mock 3 sectors with different momentum, verify ranking order."""
        classification = {
            "SH600000": "银行",
            "SH600001": "银行",
            "SH601166": "银行",
            "SH600519": "食品饮料",
            "SH600887": "食品饮料",
            "SH600036": "食品饮料",
            "SH000001": "计算机",
            "SH000002": "计算机",
            "SH000003": "计算机",
        }

        with patch('big_a.qlib_config.init_qlib'), \
             patch('big_a.data.sector.fetch_sw_classification') as mock_fetch, \
             patch('big_a.data.rotation._calc_sector_momentum') as mock_momentum:

            mock_fetch.return_value = classification
            mock_momentum.side_effect = lambda sector, days: {
                "银行": 5.2,
                "食品饮料": 3.8,
                "计算机": 2.1,
            }.get(sector, -float('inf'))

            result = rank_sectors(lookback_days=20)

            assert len(result) == 3
            assert result[0] == ("银行", 5.2)
            assert result[1] == ("食品饮料", 3.8)
            assert result[2] == ("计算机", 2.1)
            assert all(result[i][1] >= result[i + 1][1] for i in range(len(result) - 1))

    def test_top_k_sectors(self):
        """Verify returns exactly K sectors."""
        classification = {
            "SH600000": "银行",
            "SH600001": "银行",
            "SH600519": "食品饮料",
            "SH600887": "食品饮料",
            "SH000001": "计算机",
            "SH000002": "计算机",
        }

        with patch('big_a.qlib_config.init_qlib'), \
             patch('big_a.data.sector.fetch_sw_classification') as mock_fetch, \
             patch('big_a.data.rotation._calc_sector_momentum') as mock_momentum:

            mock_fetch.return_value = classification
            mock_momentum.side_effect = lambda sector, days: {
                "银行": 5.2,
                "食品饮料": 3.8,
                "计算机": 2.1,
            }.get(sector, -float('inf'))

            result = get_top_sectors(top_k=2, lookback_days=20)

            assert len(result) == 2
            assert result == ["银行", "食品饮料"]

    def test_no_data_returns_empty(self):
        """Mock D.features returns empty, returns empty list."""
        with patch('big_a.qlib_config.init_qlib'), \
             patch('big_a.data.sector.fetch_sw_classification') as mock_fetch:

            mock_fetch.return_value = {}

            result = rank_sectors(lookback_days=20)

            assert result == []

    def test_get_top_sectors_handles_fewer_than_k(self):
        """Only 3 sectors available, top_k=5, returns 3."""
        classification = {
            "SH600000": "银行",
            "SH600001": "银行",
            "SH600519": "食品饮料",
            "SH600887": "食品饮料",
            "SH000001": "计算机",
            "SH000002": "计算机",
        }

        with patch('big_a.qlib_config.init_qlib'), \
             patch('big_a.data.sector.fetch_sw_classification') as mock_fetch, \
             patch('big_a.data.rotation._calc_sector_momentum') as mock_momentum:

            mock_fetch.return_value = classification
            mock_momentum.side_effect = lambda sector, days: {
                "银行": 5.2,
                "食品饮料": 3.8,
                "计算机": 2.1,
            }.get(sector, -float('inf'))

            result = get_top_sectors(top_k=5, lookback_days=20)

            assert len(result) == 3
            assert result == ["银行", "食品饮料", "计算机"]


class TestCalcSectorMomentum:
    """Test _calc_sector_momentum function."""

    def test_small_sector_excluded(self):
        """Sector with <5 stocks excluded."""
        with patch('big_a.data.sector.get_sector_stocks') as mock_get_sector:
            mock_get_sector.return_value = ["SH600000", "SH600001", "SH600004"]

            result = _calc_sector_momentum("银行", lookback_days=20)

            assert result == float('-inf')

    def test_momentum_calculation(self):
        """Verify momentum = (current/historical - 1) * 100."""
        stocks = ["SH600000", "SH600001", "SH600519", "SH600887", "SH600036"]

        df = create_price_data(stocks, days=25, start_price=10.0, trend=0.02)

        mock_d = MagicMock()
        mock_d.features.return_value = df

        with patch('big_a.data.sector.get_sector_stocks') as mock_get_sector, \
             patch('qlib.data.D', mock_d), \
             patch('datetime.date') as mock_date:

            mock_get_sector.return_value = stocks
            mock_date.today.return_value = date(2024, 1, 25)

            result = _calc_sector_momentum("银行", lookback_days=20)

            # Momentum should be positive and reasonable
            assert result > 0
            assert isinstance(result, float)

    def test_equal_weight_sector_index(self):
        """Verify mean across stocks."""
        stocks = ["SH600000", "SH600001", "SH600519", "SH600887", "SH600036"]

        # Create data where each stock has consistent trend
        dates = pd.date_range(start=date(2024, 1, 1), periods=25)
        index = pd.MultiIndex.from_product([stocks, dates], names=["instrument", "datetime"])

        prices = []
        for stock in stocks:
            stock_prices = [10.0]
            for i in range(1, 25):
                stock_prices.append(10.0 * (1.02 ** i))
            prices.extend(stock_prices)

        df = pd.DataFrame({"$close": prices}, index=index)

        mock_d = MagicMock()
        mock_d.features.return_value = df

        with patch('big_a.data.sector.get_sector_stocks') as mock_get_sector, \
             patch('qlib.data.D', mock_d), \
             patch('datetime.date') as mock_date:

            mock_get_sector.return_value = stocks
            mock_date.today.return_value = date(2024, 1, 25)

            result = _calc_sector_momentum("银行", lookback_days=20)

            # All stocks have same 2% daily return, so sector index should have same momentum
            expected_momentum = (10.0 * (1.02 ** 24) / (10.0 * (1.02 ** 4)) - 1) * 100
            assert abs(result - expected_momentum) < 0.01

    def test_empty_features_returns_inf(self):
        """D.features returns empty DataFrame, returns -inf."""
        stocks = ["SH600000", "SH600001", "SH600519", "SH600887", "SH600036"]

        mock_d = MagicMock()
        mock_d.features.return_value = pd.DataFrame()

        with patch('big_a.data.sector.get_sector_stocks') as mock_get_sector, \
             patch('qlib.data.D', mock_d):

            mock_get_sector.return_value = stocks

            result = _calc_sector_momentum("银行", lookback_days=20)

            assert result == float('-inf')
