"""Tests for sector classification module.

All tests use mocking to avoid network dependencies.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from big_a.data.sector import (
    _to_qlib_code, _to_raw_code,
    fetch_sw_classification, get_stock_sector, get_sector_stocks,
    _save_to_cache, _load_from_cache, refresh_sector_data,
)


class TestToQlibCode:
    """Tests for _to_qlib_code function."""

    def test_to_qlib_code_shanghai(self):
        """Test converting Shanghai stock code to Qlib format."""
        assert _to_qlib_code("600000") == "SH600000"
        assert _to_qlib_code("600519") == "SH600519"

    def test_to_qlib_code_shenzhen(self):
        """Test converting Shenzhen stock codes to Qlib format."""
        assert _to_qlib_code("000001") == "SZ000001"
        assert _to_qlib_code("300001") == "SZ300001"

    def test_to_qlib_code_beijing(self):
        """Test converting Beijing stock code to Qlib format."""
        assert _to_qlib_code("430047") == "BJ430047"
        assert _to_qlib_code("832566") == "BJ832566"

    def test_to_qlib_code_unknown_prefix(self):
        """Test handling of unknown stock code prefix."""
        with patch('big_a.data.sector.logger') as mock_logger:
            result = _to_qlib_code("999999")
            assert result == "999999"
            mock_logger.warning.assert_called_once()


class TestToRawCode:
    """Tests for _to_raw_code function."""

    def test_to_raw_code(self):
        """Test converting Qlib format back to raw code."""
        assert _to_raw_code("SH600000") == "600000"
        assert _to_raw_code("SZ000001") == "000001"
        assert _to_raw_code("SZ300001") == "300001"
        assert _to_raw_code("BJ430047") == "430047"

    def test_to_raw_code_no_prefix(self):
        """Test handling of code without exchange prefix."""
        assert _to_raw_code("600000") == "600000"
        assert _to_raw_code("000001") == "000001"


class TestFetchSwClassification:
    """Tests for fetch_sw_classification function."""

    def test_fetch_sw_classification(self, tmp_path):
        """Test fetching SW classification with mocked AKShare."""
        mock_df = pd.DataFrame({
            'stock_code': ['600000', '000001', '300001', '600519'],
            'industry_name': ['银行', '银行', '计算机', '食品饮料'],
        })

        with patch('akshare.stock_industry_clf_hist_sw', return_value=mock_df):
            with patch('big_a.data.sector.CACHE_FILE', tmp_path / 'test_cache.parquet'):
                result = fetch_sw_classification(force_refresh=True)

                assert isinstance(result, dict)
                assert len(result) == 4
                assert result['SH600000'] == '银行'
                assert result['SZ000001'] == '银行'
                assert result['SZ300001'] == '计算机'
                assert result['SH600519'] == '食品饮料'


class TestCacheRoundtrip:
    """Tests for cache save/load functionality."""

    def test_cache_roundtrip(self, tmp_path):
        """Test saving to parquet and reading back, verify identical."""
        test_data = {
            'SH600000': '银行',
            'SZ000001': '银行',
            'SZ300001': '计算机',
        }

        cache_file = tmp_path / 'test_cache.parquet'

        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            _save_to_cache(test_data)

            assert cache_file.exists()

            loaded_data = _load_from_cache()

            assert loaded_data == test_data


class TestGetStockSector:
    """Tests for get_stock_sector function."""

    def test_get_stock_sector(self, tmp_path):
        """Test querying known stock, verify correct sector."""
        test_data = {
            'SH600000': '银行',
            'SZ000001': '银行',
            'SZ300001': '计算机',
        }

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            _save_to_cache(test_data)

            sector = get_stock_sector('SH600000')
            assert sector == '银行'

            sector = get_stock_sector('000001')
            assert sector == '银行'

    def test_get_stock_sector_missing(self, tmp_path):
        """Test querying unknown stock, verify returns None."""
        test_data = {
            'SH600000': '银行',
            'SZ000001': '银行',
        }

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            _save_to_cache(test_data)

            sector = get_stock_sector('SH999999')
            assert sector is None


class TestGetSectorStocks:
    """Tests for get_sector_stocks function."""

    def test_get_sector_stocks(self, tmp_path):
        """Test querying known sector, verify returns list of stock codes."""
        test_data = {
            'SH600000': '银行',
            'SZ000001': '银行',
            'SH601398': '银行',
            'SZ300001': '计算机',
        }

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            _save_to_cache(test_data)

            stocks = get_sector_stocks('银行')
            assert isinstance(stocks, list)
            assert len(stocks) == 3
            assert 'SH600000' in stocks
            assert 'SZ000001' in stocks
            assert 'SH601398' in stocks

            for stock in stocks:
                assert stock.startswith(('SH', 'SZ', 'BJ'))


class TestNetworkFailureFallback:
    """Tests for network failure fallback behavior."""

    def test_network_failure_fallback(self, tmp_path):
        """Test mock akshare to raise exception, set up cache first, verify uses cache."""
        test_data = {
            'SH600000': '银行',
            'SZ000001': '银行',
        }

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            _save_to_cache(test_data)

            with patch('akshare.stock_industry_clf_hist_sw', side_effect=Exception("Network error")):
                result = fetch_sw_classification(force_refresh=False)

                assert result == test_data
                assert len(result) == 2
                assert result['SH600000'] == '银行'


class TestRefreshForcesUpdate:
    """Tests for force_refresh behavior."""

    def test_refresh_forces_update(self, tmp_path):
        """Test force_refresh=True bypasses cache."""
        cache_data = {
            'SH600000': '银行',
            'SZ000001': '银行',
        }

        fresh_df = pd.DataFrame({
            'stock_code': ['600000', '000001', '300001'],
            'industry_name': ['银行', '银行', '计算机'],
        })

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            _save_to_cache(cache_data)

            with patch('big_a.data.sector._load_from_cache', wraps=_load_from_cache) as mock_load:
                with patch('akshare.stock_industry_clf_hist_sw', return_value=fresh_df):
                    result = fetch_sw_classification(force_refresh=True)

                    mock_load.assert_not_called()

                    assert len(result) == 3
                    assert 'SZ300001' in result


class TestRefreshSectorData:
    """Tests for refresh_sector_data function."""

    def test_refresh_sector_data_success(self, tmp_path):
        """Test successful refresh updates cache."""
        fresh_df = pd.DataFrame({
            'stock_code': ['600000', '000001'],
            'industry_name': ['银行', '银行'],
        })

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            with patch('akshare.stock_industry_clf_hist_sw', return_value=fresh_df):
                refresh_sector_data()

                loaded_data = _load_from_cache()
                assert loaded_data is not None
                assert len(loaded_data) == 2

    def test_refresh_sector_data_failure(self, tmp_path):
        """Test refresh raises RuntimeError when both AKShare and cache fail."""
        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            with patch('akshare.stock_industry_clf_hist_sw', side_effect=Exception("Network error")):
                with pytest.raises(RuntimeError, match="Failed to refresh sector data"):
                    refresh_sector_data()


class TestEmptyDataHandling:
    """Additional tests for edge cases."""

    def test_empty_dataframe_from_akshare(self, tmp_path):
        """Test handling of empty DataFrame from AKShare."""
        empty_df = pd.DataFrame()

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            with patch('akshare.stock_industry_clf_hist_sw', return_value=empty_df):
                result = fetch_sw_classification(force_refresh=True)
                assert result == {}

    def test_missing_columns_from_akshare(self, tmp_path):
        """Test handling of DataFrame with missing required columns."""
        bad_df = pd.DataFrame({
            'wrong_col1': ['600000', '000001'],
            'wrong_col2': ['银行', '银行'],
        })

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            with patch('akshare.stock_industry_clf_hist_sw', return_value=bad_df):
                result = fetch_sw_classification(force_refresh=True)
                assert result == {}

    def test_get_sector_stocks_unknown(self, tmp_path):
        """Test querying unknown sector returns empty list."""
        test_data = {
            'SH600000': '银行',
            'SZ000001': '银行',
        }

        cache_file = tmp_path / 'test_cache.parquet'
        with patch('big_a.data.sector.CACHE_FILE', cache_file):
            _save_to_cache(test_data)

            stocks = get_sector_stocks('Unknown Sector')
            assert stocks == []
