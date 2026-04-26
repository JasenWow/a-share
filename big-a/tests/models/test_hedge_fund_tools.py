"""Tests for hedge fund tools (Qlib data tools)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestGetPrices:
    """Tests for get_prices function."""

    @patch("qlib.data.D")
    def test_get_prices_returns_dataframe(self, mock_d: MagicMock) -> None:
        """Test that get_prices returns a DataFrame."""
        from big_a.models.hedge_fund.tools.qlib_tools import get_prices

        mock_df = pd.DataFrame(
            {"$open": [1.0], "$high": [2.0], "$low": [0.5], "$close": [1.5], "$volume": [1000]},
            index=pd.MultiIndex.from_tuples([("AAPL", pd.Timestamp("2024-01-01"))], names=["instrument", "datetime"]),
        )
        mock_d.features.return_value = mock_df

        result = get_prices(["AAPL"], "2024-01-01", "2024-01-02")

        assert isinstance(result, pd.DataFrame)
        mock_d.features.assert_called_once_with(
            ["AAPL"], fields=["$open", "$high", "$low", "$close", "$volume"], start_time="2024-01-01", end_time="2024-01-02"
        )

    @patch("qlib.data.D")
    def test_get_prices_strips_dollar_prefix(self, mock_d: MagicMock) -> None:
        """Test that get_prices strips $ prefix from column names."""
        from big_a.models.hedge_fund.tools.qlib_tools import get_prices

        mock_df = pd.DataFrame(
            {"$open": [1.0], "$high": [2.0], "$low": [0.5], "$close": [1.5], "$volume": [1000]},
            index=pd.MultiIndex.from_tuples([("AAPL", pd.Timestamp("2024-01-01"))], names=["instrument", "datetime"]),
        )
        mock_d.features.return_value = mock_df

        result = get_prices(["AAPL"], "2024-01-01", "2024-01-02")

        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert "$open" not in result.columns

    @patch("qlib.data.D")
    def test_get_prices_with_custom_fields(self, mock_d: MagicMock) -> None:
        """Test that get_prices uses custom fields when provided."""
        from big_a.models.hedge_fund.tools.qlib_tools import get_prices

        mock_df = pd.DataFrame(
            {"$close": [1.5]},
            index=pd.MultiIndex.from_tuples([("AAPL", pd.Timestamp("2024-01-01"))], names=["instrument", "datetime"]),
        )
        mock_d.features.return_value = mock_df

        result = get_prices(["AAPL"], "2024-01-01", "2024-01-02", fields=["$close"])

        assert isinstance(result, pd.DataFrame)
        mock_d.features.assert_called_once_with(["AAPL"], fields=["$close"], start_time="2024-01-01", end_time="2024-01-02")


class TestGetTechnicalIndicators:
    """Tests for get_technical_indicators function."""

    @patch("qlib.data.D")
    def test_get_technical_indicators_returns_dataframe(self, mock_d: MagicMock) -> None:
        """Test that get_technical_indicators returns a DataFrame."""
        from big_a.models.hedge_fund.tools.qlib_tools import get_technical_indicators

        mock_df = pd.DataFrame(
            {
                "Mean($close, 5)": [1.5],
                "Mean($close, 10)": [1.4],
                "Mean($close, 20)": [1.3],
                "Std($change, 20)": [0.1],
                "Rank($volume)": [50],
                "Mean($volume, 5)": [1000],
                "Mean($volume, 20)": [900],
                "$volume / Mean($volume, 20)": [1.11],
            },
            index=pd.MultiIndex.from_tuples([("AAPL", pd.Timestamp("2024-01-01"))], names=["instrument", "datetime"]),
        )
        mock_d.features.return_value = mock_df

        result = get_technical_indicators(["AAPL"], "2024-01-01", "2024-01-02")

        assert isinstance(result, pd.DataFrame)
        expected_fields = [
            "Mean($close, 5)",
            "Mean($close, 10)",
            "Mean($close, 20)",
            "Std($change, 20)",
            "Rank($volume)",
            "Mean($volume, 5)",
            "Mean($volume, 20)",
            "$volume / Mean($volume, 20)",
        ]
        mock_d.features.assert_called_once_with(["AAPL"], fields=expected_fields, start_time="2024-01-01", end_time="2024-01-02")


class TestGetMarketData:
    """Tests for get_market_data function."""

    @patch("qlib.data.D")
    def test_get_market_data_returns_dict(self, mock_d: MagicMock) -> None:
        """Test that get_market_data returns a dict with instrument data."""
        from big_a.models.hedge_fund.tools.qlib_tools import get_market_data

        mock_df = pd.DataFrame(
            {"$close": [1.5], "$volume": [1000], "$change": [0.1]},
            index=pd.MultiIndex.from_tuples([("AAPL", pd.Timestamp("2024-01-01"))], names=["instrument", "datetime"]),
        )
        mock_d.features.return_value = mock_df

        result = get_market_data(["AAPL"], "2024-01-01")

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert result["AAPL"] == {"close": 1.5, "volume": 1000, "change": 0.1}
        mock_d.features.assert_called_once_with(["AAPL"], fields=["$close", "$volume", "$change"], start_time="2024-01-01", end_time="2024-01-01")

    @patch("qlib.data.D")
    def test_get_market_data_handles_missing_instrument(self, mock_d: MagicMock) -> None:
        """Test that get_market_data returns None values for missing instruments."""
        from big_a.models.hedge_fund.tools.qlib_tools import get_market_data

        mock_df = pd.DataFrame(
            {"$close": [1.5], "$volume": [1000], "$change": [0.1]},
            index=pd.MultiIndex.from_tuples([("AAPL", pd.Timestamp("2024-01-01"))], names=["instrument", "datetime"]),
        )
        mock_d.features.return_value = mock_df

        result = get_market_data(["AAPL", "MSFT"], "2024-01-01")

        assert isinstance(result, dict)
        assert result["AAPL"] == {"close": 1.5, "volume": 1000, "change": 0.1}
        assert result["MSFT"] == {"close": None, "volume": None, "change": None}

    @patch("qlib.data.D")
    def test_get_market_data_handles_multiindex_row(self, mock_d: MagicMock) -> None:
        """Test that get_market_data handles MultiIndex rows correctly."""
        from big_a.models.hedge_fund.tools.qlib_tools import get_market_data

        mock_df = pd.DataFrame(
            {
                "$close": [1.4, 1.5],
                "$volume": [900, 1000],
                "$change": [0.08, 0.1],
            },
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2024-01-01")), ("AAPL", pd.Timestamp("2024-01-02"))],
                names=["instrument", "datetime"],
            ),
        )
        mock_d.features.return_value = mock_df

        result = get_market_data(["AAPL"], "2024-01-01")

        assert isinstance(result, dict)
        assert result["AAPL"] == {"close": 1.5, "volume": 1000, "change": 0.1}
