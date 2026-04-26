"""Tests for screener module - all mocked, no network or Qlib dependency."""
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from big_a.data.screener import (
    build_universe,
    filter_by_sectors,
    load_base_pool,
    load_watchlist,
    validate_instruments,
)


class TestLoadWatchlist:
    """Test load_watchlist function."""

    def test_load_watchlist(self, tmp_path):
        """Mock YAML, verify extracts stock codes."""
        with patch('big_a.config.load_config') as mock_load_config:
            mock_load_config.return_value = {
                "watchlist": {
                    "SH600000": "浦发银行",
                    "SH600519": "贵州茅台",
                    "SH600001": "邯郸钢铁",
                }
            }

            result = load_watchlist("watchlist.yaml")

            assert set(result) == {"SH600000", "SH600519", "SH600001"}
            assert len(result) == 3

    def test_load_watchlist_empty(self, tmp_path):
        """Empty watchlist returns empty list."""
        with patch('big_a.config.load_config') as mock_load_config:
            mock_load_config.return_value = {"watchlist": {}}

            result = load_watchlist("empty.yaml")

            assert result == []


class TestLoadBasePool:
    """Test load_base_pool function."""

    def test_load_base_pool_csi300(self):
        """Mock D.instruments/D.list_instruments, verify returns list."""
        mock_d = MagicMock()
        mock_d.instruments.return_value = "csi300"
        mock_d.list_instruments.return_value = ["SH600000", "SH600519", "SH600001", "SH600004"]

        with patch('big_a.qlib_config.init_qlib'), \
             patch('qlib.data.D', mock_d):

            result = load_base_pool("csi300")

            assert isinstance(result, list)
            assert len(result) == 4
            assert "SH600000" in result
            assert "SH600519" in result


class TestValidateInstruments:
    """Test validate_instruments function."""

    def test_validate_instruments(self, tmp_path):
        """Mock FEATURES_DIR, some valid some invalid, verify filtering."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()

        # Create feature directories for valid codes
        (features_dir / "sh600000").mkdir()
        (features_dir / "sh600519").mkdir()
        # Don't create for SH600999 (invalid)

        with patch('big_a.data.screener.FEATURES_DIR', features_dir):
            result = validate_instruments(["SH600000", "SH600519", "SH600999"])

            assert "SH600000" in result
            assert "SH600519" in result
            assert "SH600999" not in result
            assert len(result) == 2

    def test_validate_instruments_no_features_dir(self, tmp_path):
        """No features directory returns empty list."""
        non_existent_dir = tmp_path / "non_existent"

        with patch('big_a.data.screener.FEATURES_DIR', non_existent_dir):
            result = validate_instruments(["SH600000", "SH600519"])

            assert result == []


class TestFilterBySectors:
    """Test filter_by_sectors function."""

    def test_filter_by_sectors(self):
        """Mock get_sector_stocks, verify intersection."""
        codes = ["SH600000", "SH600519", "SH600001", "SH600004"]
        active_sectors = ["银行"]

        with patch('big_a.data.sector.get_sector_stocks') as mock_get_sector_stocks:
            # Bank sector contains SH600000 and SH600004
            mock_get_sector_stocks.return_value = ["SH600000", "SH600004", "SH601166"]

            result = filter_by_sectors(codes, active_sectors)

            assert "SH600000" in result
            assert "SH600004" in result
            assert "SH600519" not in result  # Not in bank sector
            assert "SH600001" not in result  # Not in bank sector
            assert len(result) == 2

    def test_filter_by_sectors_empty(self):
        """Empty active_sectors returns all codes."""
        codes = ["SH600000", "SH600519", "SH600001"]

        result = filter_by_sectors(codes, [])

        assert set(result) == set(codes)
        assert len(result) == 3


class TestBuildUniverse:
    """Test build_universe function."""

    def test_csi300_plus_watchlist(self, tmp_path):
        """Mock everything, verify merge + dedup."""
        mock_d = MagicMock()
        mock_d.instruments.return_value = "csi300"
        mock_d.list_instruments.return_value = ["SH600000", "SH600519", "SH600001"]

        with patch('big_a.config.load_config') as mock_load_config, \
             patch('big_a.qlib_config.init_qlib'), \
             patch('qlib.data.D', mock_d), \
             patch('big_a.data.screener.FEATURES_DIR') as mock_features_dir, \
             patch('big_a.data.screener.validate_instruments') as mock_validate:

            # Setup mocks
            def load_config_side_effect(path):
                if 'universe' in str(path):
                    return {
                        "universe": {
                            "base_pool": "csi300",
                            "watchlist": "watchlist.yaml",
                            "sector_rotation": {"enabled": False},
                        }
                    }
                else:
                    return {
                        "watchlist": {
                            "SH600519": "茅台",
                            "SH600999": "测试",
                        }
                    }

            mock_load_config.side_effect = load_config_side_effect
            mock_features_dir.exists.return_value = True

            # All codes are valid
            mock_validate.return_value = ["SH600000", "SH600519", "SH600001", "SH600999"]

            result = build_universe("universe.yaml")

            # Should have base pool + watchlist merged and deduplicated
            assert "SH600000" in result  # from base pool
            assert "SH600519" in result  # from both (deduped)
            assert "SH600001" in result  # from base pool
            assert "SH600999" in result  # from watchlist
            assert len(result) == 4

    def test_empty_watchlist_fallback(self, tmp_path):
        """Empty watchlist, returns just base pool."""
        mock_d = MagicMock()
        mock_d.instruments.return_value = "csi300"
        mock_d.list_instruments.return_value = ["SH600000", "SH600519"]

        with patch('big_a.config.load_config') as mock_load_config, \
             patch('big_a.qlib_config.init_qlib'), \
             patch('qlib.data.D', mock_d), \
             patch('big_a.data.screener.FEATURES_DIR') as mock_features_dir, \
             patch('big_a.data.screener.validate_instruments') as mock_validate:

            def load_config_side_effect(path):
                if 'universe' in str(path):
                    return {
                        "universe": {
                            "base_pool": "csi300",
                            "watchlist": "empty_watchlist.yaml",
                            "sector_rotation": {"enabled": False},
                        }
                    }
                else:
                    return {"watchlist": {}}

            mock_load_config.side_effect = load_config_side_effect
            mock_features_dir.exists.return_value = True
            mock_validate.return_value = ["SH600000", "SH600519"]

            result = build_universe("universe.yaml")

            assert set(result) == {"SH600000", "SH600519"}
            assert len(result) == 2

    def test_sector_rotation_disabled(self, tmp_path):
        """sector_rotation.enabled=false, no rotation filter applied."""
        mock_d = MagicMock()
        mock_d.instruments.return_value = "csi300"
        mock_d.list_instruments.return_value = ["SH600000", "SH600519"]

        with patch('big_a.config.load_config') as mock_load_config, \
             patch('big_a.qlib_config.init_qlib'), \
             patch('qlib.data.D', mock_d), \
             patch('big_a.data.screener.FEATURES_DIR') as mock_features_dir, \
             patch('big_a.data.screener.validate_instruments') as mock_validate, \
             patch('big_a.data.screener.filter_by_sectors') as mock_filter:

            def load_config_side_effect(path):
                if 'universe' in str(path):
                    return {
                        "universe": {
                            "base_pool": "csi300",
                            "watchlist": "watchlist.yaml",
                            "sector_rotation": {"enabled": False},
                        }
                    }
                else:
                    return {"watchlist": {"SH600519": "茅台"}}

            mock_load_config.side_effect = load_config_side_effect
            mock_features_dir.exists.return_value = True
            mock_validate.return_value = ["SH600000", "SH600519"]

            result = build_universe("universe.yaml")

            # filter_by_sectors should NOT be called when sector_rotation is disabled
            mock_filter.assert_not_called()
            assert set(result) == {"SH600000", "SH600519"}

    def test_build_universe_integration(self, tmp_path):
        """Full flow with all mocks including sector rotation."""
        mock_d = MagicMock()
        mock_d.instruments.return_value = "csi300"
        mock_d.list_instruments.return_value = ["SH600000", "SH600519", "SH600001"]

        with patch('big_a.config.load_config') as mock_load_config, \
             patch('big_a.qlib_config.init_qlib'), \
             patch('qlib.data.D', mock_d), \
             patch('big_a.data.screener.FEATURES_DIR') as mock_features_dir, \
             patch('big_a.data.screener.validate_instruments') as mock_validate, \
             patch('big_a.data.rotation.get_top_sectors') as mock_top_sectors, \
             patch('big_a.data.screener.filter_by_sectors') as mock_filter:

            def load_config_side_effect(path):
                if 'universe' in str(path):
                    return {
                        "universe": {
                            "base_pool": "csi300",
                            "watchlist": "watchlist.yaml",
                            "sector_rotation": {
                                "enabled": True,
                                "lookback_days": 20,
                                "top_k_sectors": 2,
                            },
                        }
                    }
                else:
                    return {"watchlist": {"SH600999": "测试"}}

            mock_load_config.side_effect = load_config_side_effect
            mock_features_dir.exists.return_value = True
            mock_validate.return_value = ["SH600000", "SH600519", "SH600001", "SH600999"]
            mock_top_sectors.return_value = ["银行", "计算机"]
            mock_filter.return_value = ["SH600000", "SH600001"]

            result = build_universe("universe.yaml")

            # Should have called get_top_sectors and filter_by_sectors
            mock_top_sectors.assert_called_once_with(lookback_days=20, top_k=2)
            mock_filter.assert_called_once()
            assert result == ["SH600000", "SH600001"]

    def test_invalid_watchlist_excluded(self, tmp_path):
        """Watchlist stock not in features, excluded with warning."""
        # Create a features directory with SH600519 and SH600000
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        (features_dir / "sh600519").mkdir()
        (features_dir / "sh600000").mkdir()
        # SH600999 doesn't have a feature directory

        mock_d = MagicMock()
        mock_d.instruments.return_value = "csi300"
        mock_d.list_instruments.return_value = ["SH600000"]

        with patch('big_a.config.load_config') as mock_load_config, \
             patch('big_a.qlib_config.init_qlib'), \
             patch('qlib.data.D', mock_d), \
             patch('big_a.data.screener.FEATURES_DIR', features_dir):

            def load_config_side_effect(path):
                if 'universe' in str(path):
                    return {
                        "universe": {
                            "base_pool": "csi300",
                            "watchlist": "watchlist.yaml",
                            "sector_rotation": {"enabled": False},
                        }
                    }
                else:
                    return {
                        "watchlist": {
                            "SH600519": "茅台",
                            "SH600999": "测试",
                        }
                    }

            mock_load_config.side_effect = load_config_side_effect

            result = build_universe("universe.yaml")

            # SH600999 should be excluded because it has no feature data
            assert "SH600519" in result
            assert "SH600000" in result
            assert "SH600999" not in result
