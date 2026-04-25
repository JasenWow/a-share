"""Tests for custom A-share operators and factor expressions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ---------------------------------------------------------------------------
# Unit tests – no Qlib data required
# ---------------------------------------------------------------------------
class TestLimitStatus:
    def test_up_limit(self):
        from big_a.factors.custom_ops import LimitStatus
        from qlib.data.base import Feature

        close = pd.Series([10.0, 10.0, 10.0])
        high = pd.Series([10.0, 10.0, 10.0])
        low = pd.Series([9.0, 9.5, 10.0])

        op = LimitStatus.__new__(LimitStatus)
        op._features = [Feature("close"), Feature("high"), Feature("low")]
        op.tolerance = 0.001

        result = op._compute([close, high, low])
        np.testing.assert_array_equal(result.values, [1, 1, 1])

    def test_down_limit(self):
        from big_a.factors.custom_ops import LimitStatus

        close = pd.Series([5.0, 5.0])
        high = pd.Series([6.0, 6.0])
        low = pd.Series([5.0, 5.0])

        op = LimitStatus.__new__(LimitStatus)
        op._features = []
        op.tolerance = 0.001

        result = op._compute([close, high, low])
        np.testing.assert_array_equal(result.values, [-1, -1])

    def test_no_limit(self):
        from big_a.factors.custom_ops import LimitStatus

        close = pd.Series([9.5, 9.8])
        high = pd.Series([10.0, 10.0])
        low = pd.Series([9.0, 9.0])

        op = LimitStatus.__new__(LimitStatus)
        op._features = []
        op.tolerance = 0.001

        result = op._compute([close, high, low])
        np.testing.assert_array_equal(result.values, [0, 0])

    def test_zero_close(self):
        from big_a.factors.custom_ops import LimitStatus

        close = pd.Series([0.0])
        high = pd.Series([0.0])
        low = pd.Series([0.0])

        op = LimitStatus.__new__(LimitStatus)
        op._features = []
        op.tolerance = 0.001

        result = op._compute([close, high, low])
        np.testing.assert_array_equal(result.values, [0])


class TestVWAP:
    def test_basic_computation(self):
        from big_a.factors.custom_ops import VWAP

        open_p = pd.Series([10.0])
        high = pd.Series([11.0])
        low = pd.Series([9.0])
        close = pd.Series([10.5])
        volume = pd.Series([1000.0])

        op = VWAP.__new__(VWAP)
        op._features = []

        result = op._compute([open_p, high, low, close, volume])
        expected = (10.0 + 11.0 + 9.0 + 10.5 * 2) / 5
        np.testing.assert_almost_equal(result.iloc[0], expected)

    def test_multiple_rows(self):
        from big_a.factors.custom_ops import VWAP

        open_p = pd.Series([10.0, 20.0])
        high = pd.Series([12.0, 22.0])
        low = pd.Series([8.0, 18.0])
        close = pd.Series([11.0, 21.0])
        volume = pd.Series([500.0, 600.0])

        op = VWAP.__new__(VWAP)
        op._features = []

        result = op._compute([open_p, high, low, close, volume])
        assert len(result) == 2
        expected_0 = (10.0 + 12.0 + 8.0 + 11.0 * 2) / 5
        expected_1 = (20.0 + 22.0 + 18.0 + 21.0 * 2) / 5
        np.testing.assert_almost_equal(result.iloc[0], expected_0)
        np.testing.assert_almost_equal(result.iloc[1], expected_1)


class TestVolumeRatio:
    def test_basic_computation(self):
        from big_a.factors.custom_ops import VolumeRatio

        volume = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])

        op = VolumeRatio.__new__(VolumeRatio)
        op.volume = None
        op.window = 3

        class _FakeFeature:
            def load(self, *args, **kwargs):
                return volume

            def get_longest_back_rolling(self):
                return 0

            def get_extended_window_size(self):
                return 0, 0

        op.volume = _FakeFeature()
        result = op._load_internal("dummy", 0, 4)

        assert len(result) == 5
        assert result.notna().all()
        last_ratio = 500.0 / ((300.0 + 400.0 + 500.0) / 3)
        np.testing.assert_almost_equal(result.iloc[-1], last_ratio)


class TestAlphaFactors:
    def test_factors_list_not_empty(self):
        from big_a.factors.alpha_factors import CUSTOM_FACTORS

        assert len(CUSTOM_FACTORS) == 9

    def test_factors_are_strings(self):
        from big_a.factors.alpha_factors import CUSTOM_FACTORS

        for expr in CUSTOM_FACTORS:
            assert isinstance(expr, str)
            assert len(expr) > 0

    def test_aliases_match_factors(self):
        from big_a.factors.alpha_factors import CUSTOM_FACTORS, FACTOR_ALIASES

        assert len(FACTOR_ALIASES) == len(CUSTOM_FACTORS)
        for alias, expr in FACTOR_ALIASES.items():
            assert expr in CUSTOM_FACTORS

    def test_factors_under_limit(self):
        from big_a.factors.alpha_factors import CUSTOM_FACTORS

        assert len(CUSTOM_FACTORS) <= 10


# ---------------------------------------------------------------------------
# Integration tests – require Qlib data
# ---------------------------------------------------------------------------
@pytest.mark.skip_if_no_data
class TestCustomOpsIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, qlib_initialized):
        pass

    def test_vwap_expression(self):
        from big_a.qlib_config import init_qlib
        init_qlib()
        from qlib.data import D

        df = D.features(
            ["SH600000"],
            ["VWAP($open,$high,$low,$close,$volume)"],
            "2024-01-01",
            "2024-06-30",
        )
        df = df.dropna()
        assert not df.empty
        assert (df.iloc[:, 0] > 0).all()

    def test_limit_status_expression(self):
        from big_a.qlib_config import init_qlib
        init_qlib()
        from qlib.data import D

        df = D.features(
            ["SH600000"],
            ["LimitStatus($close,$high,$low)"],
            "2024-01-01",
            "2024-06-30",
        )
        df = df.dropna()
        assert not df.empty
        assert set(df.iloc[:, 0].unique()).issubset({-1.0, 0.0, 1.0})

    def test_volume_ratio_expression(self):
        from big_a.qlib_config import init_qlib
        init_qlib()
        from qlib.data import D

        df = D.features(
            ["SH600000"],
            ["VolumeRatio($volume,20)"],
            "2024-01-01",
            "2024-06-30",
        )
        df = df.dropna()
        assert not df.empty
        assert (df.iloc[:, 0] > 0).all()

    def test_alpha_factor_expressions(self):
        from big_a.factors.alpha_factors import CUSTOM_FACTORS
        from big_a.qlib_config import init_qlib
        init_qlib()
        from qlib.data import D

        df = D.features(
            ["SH600000"],
            CUSTOM_FACTORS[:3],
            "2024-01-01",
            "2024-06-30",
        )
        df = df.dropna()
        assert not df.empty
