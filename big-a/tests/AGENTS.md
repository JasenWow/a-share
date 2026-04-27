# Tests Knowledge Base

## OVERVIEW

32 pytest test files validating models, backtest, strategy, data, and report modules with heavy mocking to avoid live Qlib data dependencies.

## STRUCTURE

```
big-a/tests/
├── conftest.py              # Global pytest config, Qlib init, skip_if_no_data marker
├── test_smoke.py            # Package import validation, version checks
├── test_kronos.py           # Kronos model tests (class-based)
├── test_lightgbm.py         # LightGBM model tests
├── test_backtest.py         # Backtest engine tests
├── test_rolling.py          # Rolling backtest tests
├── test_evaluation.py       # IC/sharpe metrics tests
├── test_real_trading.py     # RealTradingStrategy tests
├── test_rotation.py         # Rotation strategy tests
├── test_screener.py          # Screener tests
├── test_sector.py           # Sector strategy tests
├── test_scheduler.py         # Scheduler tests
├── test_monitoring.py        # Monitoring tests
├── test_analysis.py          # Analysis tests
├── test_report.py            # Report scorer + formatter tests (902 lines)
├── test_data_validation.py   # Data validation with mock D objects
├── test_alpha158.py          # Alpha158 factor tests
├── test_custom_factors.py    # Custom factor tests
└── models/
    ├── __init__.py
    ├── test_hedge_fund_llm.py
    ├── test_hedge_fund_tools.py
    ├── test_hedge_fund_types.py
    ├── test_deps.py
    ├── test_hedge_fund_signal_generator.py
    ├── test_hedge_fund_agents_portfolio.py
    ├── test_hedge_fund_agents_risk.py
    ├── test_hedge_fund_agents_valuation.py
    ├── test_hedge_fund_agents_sentiment.py
    ├── test_hedge_fund_agents_technicals.py
    ├── test_hedge_fund_agents_investors_batch1.py
    └── test_hedge_fund_agents_investors_batch2.py
```

## WHERE TO LOOK

| Area | Key Files |
|------|-----------|
| Global config | `conftest.py` - skip_if_no_data marker, Qlib session init |
| Package sanity | `test_smoke.py` - import checks, version validation |
| Model tests | `test_kronos.py`, `test_lightgbm.py`, `models/test_hedge_fund_*.py` |
| Backtest logic | `test_backtest.py`, `test_rolling.py`, `test_evaluation.py` |
| Strategy tests | `test_real_trading.py` (453 lines, comprehensive mocking) |
| Report generation | `test_report.py` (902 lines, scorer + formatter) |
| Data/factors | `test_data_validation.py`, `test_alpha158.py`, `test_custom_factors.py` |

## CONVENTIONS

**Custom marker:**
- `@pytest.mark.skip_if_no_data` - skips tests when Qlib CN data unavailable
- conftest.py checks `data/qlib_data/cn_data/calendars/day.txt` existence

**Qlib initialization:**
- conftest.py `init_qlib()` called once per session via `qlib_initialized` fixture
- Tests that require live data use `@pytest.mark.skip_if_no_data` or request `qlib_initialized` fixture

**Mocking patterns:**
- `@patch("big_a.models.kronos.KronosPredictor")` for model loading
- `@patch("qlib.contrib.strategy.TopkDropoutStrategy")` for backtest strategy
- `@patch("qlib.contrib.evaluate.backtest_daily")` for backtest execution
- `@patch("qlib.data.D")` for Qlib data queries
- `@patch.object(WatchlistScorer, "fetch_market_data")` for instance methods
- `PropertyMock` for property mocking (e.g., `type(strategy).trade_calendar = PropertyMock(...)`)

**Class-based test grouping:**
- `class TestPrepareStockSequence` groups related tests
- `class TestPredict`, `class TestGenerateSignals` for Kronos
- `class TestRunBacktest`, `class TestComputeAnalysis` for backtest

**Helper factories:**
- `_make_ohlcv(n_rows, seed)` - creates synthetic OHLCV DataFrames with numpy random
- `_make_cross_sectional_data()` - creates predicted/actual with known correlation
- Mock classes like `_MockD`, `_MockCalendarD` for data layer mocking

**sys.path manipulation:**
- Each test file adds `src/` to sys.path: `SRC_ROOT = Path(__file__).resolve().parents[1] / "src"`
