"""Tests for scheduler flows using Prefect."""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from prefect import Flow

# Ensure src/ package is on sys.path so that local packages under src/big_a can be imported
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_daily_flow_fn():
    """Test that daily_pipeline.fn() calls both update_market_data and update_sector_data."""
    from big_a.scheduler.flows import daily_pipeline, update_market_data, update_sector_data

    with patch("big_a.data.updater.update_incremental") as mock_update_incremental, \
         patch("big_a.data.sector.refresh_sector_data") as mock_refresh_sector_data:

        daily_pipeline.fn()

        mock_update_incremental.assert_called_once()
        mock_refresh_sector_data.assert_called_once()


def test_partial_failure_resilient():
    """Test that if update_market_data fails, update_sector_data still runs."""
    from big_a.scheduler.flows import daily_pipeline

    with patch("big_a.data.updater.update_incremental") as mock_update_incremental, \
         patch("big_a.data.sector.refresh_sector_data") as mock_refresh_sector_data:

        mock_update_incremental.side_effect = Exception("Market data update failed")

        daily_pipeline.fn()

        mock_update_incremental.assert_called_once()
        mock_refresh_sector_data.assert_called_once()


def test_import():
    """Test that flows can be imported and daily_pipeline is a Prefect Flow."""
    from big_a.scheduler import flows

    # Verify the module has the expected attributes
    assert hasattr(flows, "daily_pipeline")
    assert hasattr(flows, "update_market_data")
    assert hasattr(flows, "update_sector_data")

    # Verify daily_pipeline is a Prefect Flow
    assert isinstance(flows.daily_pipeline, Flow)

    # Verify tasks are Prefect Tasks
    from prefect import Task
    assert isinstance(flows.update_market_data, Task)
    assert isinstance(flows.update_sector_data, Task)
