"""Tests for big_a.data.validation — all mocked, no live qlib data required."""

import math

import pytest

import big_a.data.validation as val


# ---------------------------------------------------------------------------
# Mock D objects
# ---------------------------------------------------------------------------

class _MockCalendarD:
    """Returns a calendar with a deliberate gap (missing 2026-01-03)."""

    def calendar(self, *args, **kwargs):
        return ["2026-01-02", "2026-01-04"]


class _MockStockD:
    """300 instruments with a stable feature matrix (no NaN, no big gaps)."""

    def list_instruments(self, market=None):
        return [f"INS{i}" for i in range(300)]

    def features(self, instruments, fields, start_time=None, end_time=None):
        return {
            "INS0": {
                "open": [1.0, 1.01],
                "high": [1.1, 1.2],
                "low": [0.9, 0.8],
                "close": [1.0, 1.02],
                "volume": [100, 100],
            },
            "INS1": {
                "open": [2.0, 2.01],
                "high": [2.0, 2.1],
                "low": [1.5, 1.6],
                "close": [2.0, 2.05],
                "volume": [150, 200],
            },
        }


class _MockD(_MockCalendarD, _MockStockD):
    """Combined mock covering calendar + stock queries."""
    pass


# Helpers -------------------------------------------------------------------

def _install_mock_d(monkeypatch, mock):
    monkeypatch.setattr(val, "D", mock)


# ---------------------------------------------------------------------------
# check_calendar_integrity
# ---------------------------------------------------------------------------

class TestCheckCalendarIntegrity:

    def test_detects_missing_day(self, monkeypatch):
        _install_mock_d(monkeypatch, _MockCalendarD())
        res = val.check_calendar_integrity()
        assert res["valid"] is False
        assert "2026-01-03" in res["missing_days"]

    def test_continuous_calendar(self, monkeypatch):
        class ContinuousD:
            def calendar(self, *a, **kw):
                return ["2026-01-02", "2026-01-03", "2026-01-04"]

        _install_mock_d(monkeypatch, ContinuousD())
        res = val.check_calendar_integrity("2026-01-02", "2026-01-04")
        assert res["valid"] is True
        assert res["total_days"] == 3
        assert res["missing_days"] == []

    def test_d_none_returns_invalid(self, monkeypatch):
        _install_mock_d(monkeypatch, None)
        res = val.check_calendar_integrity()
        assert res["valid"] is False
        assert res["total_days"] == 0


# ---------------------------------------------------------------------------
# check_price_continuity
# ---------------------------------------------------------------------------

class TestCheckPriceContinuity:

    def test_no_anomalies(self, monkeypatch):
        _install_mock_d(monkeypatch, _MockStockD())
        res = val.check_price_continuity()
        assert res["valid"] is True
        assert res["anomalies"] == []

    def test_detects_large_gap(self, monkeypatch):
        class GapD:
            def list_instruments(self, market=None):
                return ["INS0"]

            def features(self, instruments, fields, start_time=None, end_time=None):
                # 20% jump between close[0] and close[1]
                return {"INS0": {"close": [100.0, 120.0]}}

        _install_mock_d(monkeypatch, GapD())
        res = val.check_price_continuity()
        assert res["valid"] is False
        assert len(res["anomalies"]) == 1
        a = res["anomalies"][0]
        assert a["instrument"] == "INS0"
        assert math.isclose(a["percent_change"], 0.2, rel_tol=1e-6)

    def test_d_none_returns_invalid(self, monkeypatch):
        _install_mock_d(monkeypatch, None)
        res = val.check_price_continuity()
        assert res["valid"] is False
        assert res["anomalies"] == []


# ---------------------------------------------------------------------------
# check_nan_ratio
# ---------------------------------------------------------------------------

class TestCheckNanRatio:

    def test_nan_ratios(self, monkeypatch):
        class NanD:
            def list_instruments(self, market=None):
                return ["INS0"]

            def features(self, instruments, fields, start_time=None, end_time=None):
                return {
                    "INS0": {
                        "open": [float("nan"), 2.0],
                        "high": [1.1, 1.2],
                        "low": [0.9, 0.8],
                        "close": [1.0, float("nan")],
                        "volume": [100, 100],
                    },
                }

        _install_mock_d(monkeypatch, NanD())
        res = val.check_nan_ratio()
        # total = 10 (5 fields × 2 values), open has 1 NaN => 0.1
        assert math.isclose(res["open"], 0.1, rel_tol=1e-6)
        assert math.isclose(res["high"], 0.0, rel_tol=1e-6)
        assert math.isclose(res["close"], 0.1, rel_tol=1e-6)

    def test_no_nans(self, monkeypatch):
        _install_mock_d(monkeypatch, _MockStockD())
        res = val.check_nan_ratio()
        for field_name, ratio in res.items():
            assert math.isclose(ratio, 0.0, rel_tol=1e-6), f"{field_name} should be 0.0"


# ---------------------------------------------------------------------------
# check_stock_coverage
# ---------------------------------------------------------------------------

class TestCheckStockCoverage:

    def test_valid_csi300(self, monkeypatch):
        _install_mock_d(monkeypatch, _MockStockD())
        res = val.check_stock_coverage()
        assert res["stock_count"] == 300
        assert res["valid"] is True

    def test_too_few_stocks(self, monkeypatch):
        class FewD:
            def list_instruments(self, market=None):
                return [f"INS{i}" for i in range(250)]

        _install_mock_d(monkeypatch, FewD())
        res = val.check_stock_coverage()
        assert res["stock_count"] == 250
        assert res["valid"] is False

    def test_too_many_stocks(self, monkeypatch):
        class ManyD:
            def list_instruments(self, market=None):
                return [f"INS{i}" for i in range(315)]

        _install_mock_d(monkeypatch, ManyD())
        res = val.check_stock_coverage()
        assert res["stock_count"] == 315
        assert res["valid"] is False

    def test_d_none_returns_zero(self, monkeypatch):
        _install_mock_d(monkeypatch, None)
        res = val.check_stock_coverage()
        assert res["stock_count"] == 0
        assert res["valid"] is False


# ---------------------------------------------------------------------------
# generate_data_report
# ---------------------------------------------------------------------------

class TestGenerateDataReport:

    def test_report_keys(self, monkeypatch):
        _install_mock_d(monkeypatch, _MockD())
        r = val.generate_data_report("csi300", "2026-01-02", "2026-01-04")
        assert r["market"] == "csi300"
        assert r["start_date"] == "2026-01-02"
        assert r["end_date"] == "2026-01-04"
        assert "calendar" in r
        assert "price_continuity" in r
        assert "nan_ratio" in r
        assert "stock_coverage" in r

    def test_report_calendar_from_mock(self, monkeypatch):
        _install_mock_d(monkeypatch, _MockD())
        r = val.generate_data_report()
        assert r["calendar"]["valid"] is False
        assert "2026-01-03" in r["calendar"]["missing_days"]

    def test_report_stock_coverage_from_mock(self, monkeypatch):
        _install_mock_d(monkeypatch, _MockD())
        r = val.generate_data_report()
        assert r["stock_coverage"]["stock_count"] == 300
        assert r["stock_coverage"]["valid"] is True
