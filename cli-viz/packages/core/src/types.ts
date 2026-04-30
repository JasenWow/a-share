/** Single row from backtest_report.parquet (DatetimeIndex + columns: return, bench, cost, turnover) */
export interface BacktestReportRow {
  date: string;
  return: number;
  bench: number;
  cost: number;
  turnover: number;
}

/** Full backtest report */
export interface BacktestReport {
  rows: BacktestReportRow[];
}

/** Backtest analysis metrics from analyze_backtest() Python output */
export interface BacktestAnalysis {
  annualized_return: number;
  annualized_benchmark: number;
  excess_return: number;
  sharpe_ratio: number;
  information_ratio: number;
  max_drawdown: number;
  drawdown_duration_days: number;
  total_cost: number;
  mean_turnover: number;
  max_turnover: number;
  monthly_return_distribution: Record<string, number>;
  n_trading_days: number;
  start_date: string;
  end_date: string;
}

/** Portfolio position from compute_portfolio() */
export interface PortfolioPosition {
  instrument: string;
  name: string;
  weight: number;
  allocation: number;
  signal: string;
  entry_price: number;
}

/** Watchlist scoring summary */
export interface WatchlistSummary {
  total_stocks: number;
  bullish_count: number;
  bearish_count: number;
  avg_score: number;
  best_stock: string | null;
  worst_stock: string | null;
}

/** Kronos model prediction (from CSV: signal_date, instrument, score, score_pct) */
export interface KronosPrediction {
  signal_date: string;
  instrument: string;
  score: number;
  score_pct: number;
}

/** LightGBM model prediction (from parquet MultiIndex: date, instrument, score) */
export interface LightGBMPrediction {
  date: string;
  instrument: string;
  score: number;
}

/** Market data OHLCV row (from fetch_market_data) */
export interface MarketDataRow {
  date: string;
  instrument: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount: number;
  factor: number;
  change_pct?: number;
}

/** Factor value */
export interface FactorValue {
  date: string;
  instrument: string;
  factor_name: string;
  value: number;
}

/** Merged score from Kronos + LightGBM */
export interface MergedScore {
  instrument: string;
  kronos_score: number;
  lightgbm_score: number;
  combined_score: number;
  signal: 'Strong Buy' | 'Buy' | 'Sell' | 'Strong Sell';
}

/** Correlation matrix for factors */
export interface CorrelationMatrix {
  factor_names: string[];
  values: number[][]; // square matrix, values[i][j] = correlation between factor_names[i] and factor_names[j]
}

/** Factor return contribution */
export interface FactorReturn {
  factor_name: string;
  return_contribution: number;
}
