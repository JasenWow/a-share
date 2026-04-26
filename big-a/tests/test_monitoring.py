"""Smoke tests for monitoring and experiment tracking modules."""
from __future__ import annotations


def test_experiment_imports():
    """Verify all experiment module functions can be imported."""
    from big_a.experiment import (
        start_experiment,
        log_params,
        log_metrics,
        log_artifact,
        log_model_config,
        log_backtest_config,
        end_experiment,
        get_experiment_summary,
    )
    assert callable(start_experiment)
    assert callable(log_params)
    assert callable(log_metrics)


def test_experiment_no_raw_mlflow():
    """Verify experiment module does not directly import mlflow."""
    import big_a.experiment as exp_module
    import inspect
    source = inspect.getsource(exp_module)
    assert "import mlflow" not in source


def test_reporting_imports():
    """Verify all reporting module functions can be imported."""
    from big_a.backtest.reporting import (
        plot_rolling_metrics,
        plot_factor_distribution,
        plot_factor_correlation,
        plot_factor_ic_decay,
        plot_prediction_vs_actual,
        plot_residual_analysis,
        plot_quantile_returns,
        plot_holding_concentration,
        plot_turnover_analysis,
    )
    assert callable(plot_rolling_metrics)
    assert callable(plot_factor_distribution)


def test_reporting_figure_creation():
    """Verify plot_rolling_metrics returns valid Figure with synthetic data."""
    import numpy as np
    from big_a.backtest.rolling import WindowResult
    from big_a.backtest.reporting import plot_rolling_metrics

    results = [
        WindowResult(
            window_idx=i,
            train_start="2010-01-01",
            train_end="2014-12-31",
            valid_start="2015-01-01",
            valid_end="2015-12-31",
            test_start="2016-01-01",
            test_end="2016-12-31",
            ic=0.03 + i * 0.01,
            sharpe=0.5 + i * 0.1,
            max_drawdown=0.1,
        )
        for i in range(5)
    ]
    fig = plot_rolling_metrics(results)
    assert fig is not None
    assert len(fig.data) == 3  # 3 default metrics


def test_reporting_factor_charts():
    """Verify factor distribution and correlation charts work with synthetic data."""
    import numpy as np
    import pandas as pd
    from big_a.backtest.reporting import plot_factor_distribution, plot_factor_correlation

    np.random.seed(42)
    features = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])

    fig1 = plot_factor_distribution(features)
    assert fig1 is not None

    fig2 = plot_factor_correlation(features)
    assert fig2 is not None


def test_plotly_import():
    """Verify plotly is installed and importable."""
    import plotly
    assert hasattr(plotly, "__version__")


def test_pyarrow_import():
    """Verify pyarrow is installed and importable."""
    import pyarrow
    assert hasattr(pyarrow, "__version__")
