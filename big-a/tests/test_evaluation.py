"""Tests for evaluation module with known data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from big_a.backtest.evaluation import (
    calc_ic,
    calc_icir,
    calc_max_drawdown,
    calc_rank_ic,
    calc_sharpe,
    calc_turnover,
    compare_models,
    plot_ic_series,
    plot_model_comparison,
)
from big_a.backtest.metrics import MAX_DRAWDOWN_THRESHOLD, SUCCESS_IC, SUCCESS_SHARPE


# ---------------------------------------------------------------------------
# Fixtures: known synthetic data
# ---------------------------------------------------------------------------

def _make_cross_sectional_data(n_dates: int = 20, n_stocks: int = 50, seed: int = 42):
    """Build predicted scores and actual returns with known correlation."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    instruments = [f"STOCK{i:03d}" for i in range(n_stocks)]

    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    actual_vals = rng.standard_normal(len(idx))
    predicted_vals = actual_vals * 0.95 + rng.standard_normal(len(idx)) * 0.05

    predicted = pd.DataFrame({"score": predicted_vals}, index=idx)
    actual = pd.Series(actual_vals, index=idx, name="score")

    return predicted, actual, dates, instruments


def _make_perfect_data(n_dates: int = 10, n_stocks: int = 30):
    """Build data where predicted ≈ actual (very high correlation)."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2024-03-01", periods=n_dates)
    instruments = [f"STOCK{i:03d}" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    actual_vals = rng.standard_normal(len(idx))
    predicted_vals = actual_vals + rng.standard_normal(len(idx)) * 0.001

    predicted = pd.DataFrame({"score": predicted_vals}, index=idx)
    actual = pd.Series(actual_vals, index=idx, name="score")

    return predicted, actual


@pytest.fixture
def cs_data():
    return _make_cross_sectional_data()


@pytest.fixture
def perfect_data():
    return _make_perfect_data()


# ---------------------------------------------------------------------------
# Tests: IC and Rank IC
# ---------------------------------------------------------------------------

class TestCalcIC:
    def test_perfect_correlation(self, perfect_data):
        predicted, actual = perfect_data
        ic = calc_ic(predicted, actual)
        assert len(ic) > 0
        assert ic.mean() > 0.99

    def test_moderate_correlation(self, cs_data):
        predicted, actual, _, _ = cs_data
        ic = calc_ic(predicted, actual)
        assert len(ic) == 20
        assert 0.5 < ic.mean() < 1.0

    def test_accepts_series(self, cs_data):
        predicted, actual, _, _ = cs_data
        ic = calc_ic(predicted["score"], actual)
        assert len(ic) == 20

    def test_missing_column_raises(self, cs_data):
        predicted, _, _, _ = cs_data
        bad_df = predicted.rename(columns={"score": "wrong"})
        with pytest.raises(ValueError, match="score"):
            calc_ic(bad_df, pd.Series(dtype=float))


class TestCalcRankIC:
    def test_perfect_correlation(self, perfect_data):
        predicted, actual = perfect_data
        ric = calc_rank_ic(predicted, actual)
        assert ric.mean() > 0.99

    def test_length_matches_dates(self, cs_data):
        predicted, actual, _, _ = cs_data
        ric = calc_rank_ic(predicted, actual)
        assert len(ric) == 20


class TestCalcICIR:
    def test_high_icir_for_perfect_data(self, perfect_data):
        predicted, actual = perfect_data
        ic = calc_ic(predicted, actual)
        icir = calc_icir(ic)
        assert icir > 5.0

    def test_nan_for_single_value(self):
        ic = pd.Series([0.5])
        assert np.isnan(calc_icir(ic))


# ---------------------------------------------------------------------------
# Tests: Sharpe, MaxDrawdown, Turnover
# ---------------------------------------------------------------------------

class TestCalcSharpe:
    def test_positive_sharpe(self):
        daily = pd.Series(np.random.default_rng(42).standard_normal(252) * 0.01 + 0.001)
        sharpe = calc_sharpe(daily)
        assert sharpe > 0

    def test_zero_std_returns_nan(self):
        daily = pd.Series([1.0, 1.0, 1.0])
        assert np.isnan(calc_sharpe(daily))

    def test_single_value_returns_nan(self):
        assert np.isnan(calc_sharpe(pd.Series([0.01])))


class TestCalcMaxDrawdown:
    def test_no_drawdown(self):
        cum = pd.Series([1.0, 1.01, 1.02, 1.03])
        assert calc_max_drawdown(cum) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        cum = pd.Series([1.0, 1.1, 0.9, 1.05])
        dd = calc_max_drawdown(cum)
        expected = (1.1 - 0.9) / 1.1
        assert dd == pytest.approx(expected, abs=1e-6)

    def test_single_value(self):
        assert calc_max_drawdown(pd.Series([1.0])) == 0.0


class TestCalcTurnover:
    def test_no_change(self):
        idx = pd.MultiIndex.from_tuples(
            [("2024-01-01", "A"), ("2024-01-01", "B"),
             ("2024-01-02", "A"), ("2024-01-02", "B")],
            names=["datetime", "instrument"],
        )
        positions = pd.DataFrame({"weight": [0.5, 0.5, 0.5, 0.5]}, index=idx)
        assert calc_turnover(positions) == pytest.approx(0.0)

    def test_full_rotation(self):
        idx = pd.MultiIndex.from_tuples(
            [("2024-01-01", "A"), ("2024-01-01", "B"),
             ("2024-01-02", "A"), ("2024-01-02", "B")],
            names=["datetime", "instrument"],
        )
        positions = pd.DataFrame({"weight": [1.0, 0.0, 0.0, 1.0]}, index=idx)
        assert calc_turnover(positions) == pytest.approx(1.0)

    def test_missing_weight_column(self):
        idx = pd.MultiIndex.from_tuples(
            [("2024-01-01", "A")], names=["datetime", "instrument"],
        )
        df = pd.DataFrame({"val": [1.0]}, index=idx)
        with pytest.raises(ValueError, match="weight"):
            calc_turnover(df)


# ---------------------------------------------------------------------------
# Tests: compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_comparison_table(self, cs_data):
        predicted, actual, _, _ = cs_data
        noisy = predicted.copy()
        noisy["score"] += np.random.default_rng(7).standard_normal(len(noisy)) * 2.0

        result = compare_models(
            {"clean": predicted, "noisy": noisy},
            actual,
        )
        assert len(result) == 2
        assert "mean_ic" in result.columns
        assert "mean_rank_ic" in result.columns
        assert "icir" in result.columns
        assert result.loc["clean", "mean_ic"] > result.loc["noisy", "mean_ic"]


# ---------------------------------------------------------------------------
# Tests: plotting (smoke tests — just ensure no exception)
# ---------------------------------------------------------------------------

class TestPlotting:
    def test_plot_ic_series(self, cs_data, tmp_path):
        predicted, actual, _, _ = cs_data
        ic = calc_ic(predicted, actual)
        plot_ic_series(ic, save_path=str(tmp_path / "ic.png"))
        assert (tmp_path / "ic.png").exists()

    def test_plot_model_comparison(self, cs_data, tmp_path):
        predicted, actual, _, _ = cs_data
        comp = compare_models({"model_a": predicted, "model_b": predicted}, actual)
        plot_model_comparison(comp, metric="mean_ic", save_path=str(tmp_path / "compare.png"))
        assert (tmp_path / "compare.png").exists()


# ---------------------------------------------------------------------------
# Tests: metric thresholds exist and are reasonable
# ---------------------------------------------------------------------------

class TestMetricThresholds:
    def test_success_ic_positive(self):
        assert SUCCESS_IC > 0

    def test_success_sharpe_positive(self):
        assert SUCCESS_SHARPE > 0

    def test_max_drawdown_fraction(self):
        assert 0 < MAX_DRAWDOWN_THRESHOLD < 1
