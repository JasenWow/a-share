"""Tests for backtest analysis and visualization modules."""
import numpy as np
import pandas as pd
import pytest

from big_a.backtest.analysis import (
    analyze_backtest,
    generate_report,
    _max_drawdown_duration,
    _format_summary,
)
from big_a.backtest.plots import (
    plot_drawdown,
    plot_ic_series,
    plot_monthly_returns,
    plot_nav,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def report_df():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-04", periods=120)
    return pd.DataFrame(
        {
            "return": rng.normal(0.001, 0.015, 120),
            "bench": rng.normal(0.0005, 0.01, 120),
            "cost": rng.uniform(0.0001, 0.001, 120),
            "turnover": rng.uniform(0.01, 0.2, 120),
        },
        index=dates,
    )


@pytest.fixture
def simple_cum():
    return pd.Series([1.0, 1.05, 0.95, 1.02, 1.08, 0.99, 1.10])


# ---------------------------------------------------------------------------
# analyze_backtest
# ---------------------------------------------------------------------------

class TestAnalyzeBacktest:
    def test_returns_all_keys(self, report_df):
        result = analyze_backtest(report_df)
        expected_keys = {
            "annualized_return", "annualized_benchmark", "excess_return",
            "sharpe_ratio", "information_ratio",
            "max_drawdown", "drawdown_duration_days",
            "total_cost", "mean_turnover", "max_turnover",
            "monthly_return_distribution", "n_trading_days",
            "start_date", "end_date",
        }
        assert expected_keys == set(result.keys()) - {"_report_df"}

    def test_n_trading_days(self, report_df):
        result = analyze_backtest(report_df)
        assert result["n_trading_days"] == 120

    def test_dates_are_strings(self, report_df):
        result = analyze_backtest(report_df)
        assert isinstance(result["start_date"], str)
        assert isinstance(result["end_date"], str)

    def test_sharpe_is_finite(self, report_df):
        result = analyze_backtest(report_df)
        assert np.isfinite(result["sharpe_ratio"])

    def test_information_ratio_is_finite(self, report_df):
        result = analyze_backtest(report_df)
        assert np.isfinite(result["information_ratio"])

    def test_max_drawdown_non_negative(self, report_df):
        result = analyze_backtest(report_df)
        assert result["max_drawdown"] >= 0

    def test_turnover_stats(self, report_df):
        result = analyze_backtest(report_df)
        assert result["max_turnover"] >= result["mean_turnover"]
        assert result["mean_turnover"] > 0

    def test_monthly_distribution_is_series(self, report_df):
        result = analyze_backtest(report_df)
        assert isinstance(result["monthly_return_distribution"], pd.Series)
        assert len(result["monthly_return_distribution"]) > 0

    def test_excess_return_is_difference(self, report_df):
        result = analyze_backtest(report_df)
        expected = result["annualized_return"] - result["annualized_benchmark"]
        assert result["excess_return"] == pytest.approx(expected)

    def test_handles_string_index(self, report_df):
        df = report_df.copy()
        df.index = df.index.strftime("%Y-%m-%d")
        result = analyze_backtest(df)
        assert result["n_trading_days"] == 120


# ---------------------------------------------------------------------------
# _max_drawdown_duration
# ---------------------------------------------------------------------------

class TestMaxDrawdownDuration:
    def test_no_drawdown(self):
        cum = pd.Series([1.0, 1.01, 1.02, 1.03])
        assert _max_drawdown_duration(cum) == 0

    def test_single_dip(self):
        cum = pd.Series([1.0, 1.1, 0.9, 0.85, 1.0])
        assert _max_drawdown_duration(cum) == 3

    def test_two_dips_longer_second(self):
        cum = pd.Series([1.0, 0.95, 1.0, 0.90, 0.88, 0.87, 1.0])
        assert _max_drawdown_duration(cum) == 3


# ---------------------------------------------------------------------------
# _format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_contains_key_metrics(self, report_df):
        result = analyze_backtest(report_df)
        text = _format_summary(result)
        assert "Sharpe Ratio" in text
        assert "Max Drawdown" in text
        assert "Turnover" in text


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_creates_files(self, report_df, tmp_path):
        analysis = analyze_backtest(report_df)
        analysis["_report_df"] = report_df
        out = generate_report(analysis, tmp_path / "report_out")

        assert (out / "summary.txt").exists()
        assert (out / "nav.png").exists()
        assert (out / "drawdown.png").exists()
        assert (out / "monthly_returns.png").exists()

    def test_summary_content(self, report_df, tmp_path):
        analysis = analyze_backtest(report_df)
        analysis["_report_df"] = report_df
        out = generate_report(analysis, tmp_path / "report_out")

        text = (out / "summary.txt").read_text()
        assert "BACKTEST PERFORMANCE SUMMARY" in text

    def test_without_report_df(self, report_df, tmp_path):
        analysis = analyze_backtest(report_df)
        out = generate_report(analysis, tmp_path / "minimal")
        assert (out / "summary.txt").exists()
        assert not (out / "nav.png").exists()


# ---------------------------------------------------------------------------
# Plots (smoke tests)
# ---------------------------------------------------------------------------

class TestPlotNav:
    def test_saves_png(self, tmp_path):
        dates = pd.bdate_range("2022-01-04", periods=30)
        strategy = pd.Series(np.linspace(1.0, 1.2, 30), index=dates)
        benchmark = pd.Series(np.linspace(1.0, 1.1, 30), index=dates)
        path = str(tmp_path / "nav.png")
        plot_nav(strategy, benchmark, save_path=path)
        assert (tmp_path / "nav.png").exists()


class TestPlotDrawdown:
    def test_saves_png(self, tmp_path):
        dates = pd.bdate_range("2022-01-04", periods=30)
        cum = pd.Series(np.linspace(1.0, 1.1, 30), index=dates)
        path = str(tmp_path / "dd.png")
        plot_drawdown(cum, save_path=path)
        assert (tmp_path / "dd.png").exists()


class TestPlotMonthlyReturns:
    def test_saves_png(self, tmp_path):
        dates = pd.bdate_range("2022-01-04", periods=60)
        returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 60), index=dates)
        path = str(tmp_path / "monthly.png")
        plot_monthly_returns(returns, save_path=path)
        assert (tmp_path / "monthly.png").exists()


class TestPlotICSeries:
    def test_saves_png(self, tmp_path):
        dates = pd.bdate_range("2022-01-04", periods=20)
        ic = pd.Series(np.random.default_rng(2).normal(0.05, 0.1, 20), index=dates)
        path = str(tmp_path / "ic.png")
        plot_ic_series(ic, save_path=path)
        assert (tmp_path / "ic.png").exists()
