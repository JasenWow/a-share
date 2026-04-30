import { describe, expect, test } from 'bun:test';
import type {
  BacktestReportRow,
  BacktestReport,
  BacktestAnalysis,
  PortfolioPosition,
  WatchlistSummary,
  KronosPrediction,
  LightGBMPrediction,
  MarketDataRow,
  FactorValue,
  MergedScore,
  CorrelationMatrix,
  FactorReturn,
} from '../src/types.js';

describe('types', () => {
  test('BacktestReportRow has correct shape', () => {
    const row: BacktestReportRow = {
      date: '2024-01-01',
      return: 0.05,
      bench: 0.03,
      cost: 0.001,
      turnover: 0.1,
    };
    expect(row.date).toBe('2024-01-01');
    expect(row.return).toBe(0.05);
    expect(row.bench).toBe(0.03);
    expect(row.cost).toBe(0.001);
    expect(row.turnover).toBe(0.1);
  });

  test('BacktestReport contains rows array', () => {
    const report: BacktestReport = {
      rows: [
        { date: '2024-01-01', return: 0.05, bench: 0.03, cost: 0.001, turnover: 0.1 },
        { date: '2024-01-02', return: 0.02, bench: 0.01, cost: 0.001, turnover: 0.05 },
      ],
    };
    expect(report.rows).toHaveLength(2);
    expect(report.rows[0].date).toBe('2024-01-01');
  });

  test('WatchlistSummary best_stock and worst_stock can be null', () => {
    const summaryNull: WatchlistSummary = {
      total_stocks: 10,
      bullish_count: 6,
      bearish_count: 4,
      avg_score: 0.55,
      best_stock: null,
      worst_stock: null,
    };
    expect(summaryNull.best_stock).toBeNull();
    expect(summaryNull.worst_stock).toBeNull();

    const summaryWithValues: WatchlistSummary = {
      total_stocks: 10,
      bullish_count: 6,
      bearish_count: 4,
      avg_score: 0.55,
      best_stock: 'AAPL',
      worst_stock: 'TSLA',
    };
    expect(summaryWithValues.best_stock).toBe('AAPL');
    expect(summaryWithValues.worst_stock).toBe('TSLA');
  });

  test('MergedScore signal is one of literal union values', () => {
    const signals: MergedScore['signal'][] = ['Strong Buy', 'Buy', 'Sell', 'Strong Sell'];
    for (const signal of signals) {
      const score: MergedScore = {
        instrument: 'AAPL',
        kronos_score: 0.8,
        lightgbm_score: 0.7,
        combined_score: 0.75,
        signal,
      };
      expect(signals.includes(score.signal)).toBe(true);
    }
  });

  test('CorrelationMatrix is square', () => {
    const factor_names = ['factor_a', 'factor_b', 'factor_c'];
    const values = [
      [1.0, 0.5, 0.2],
      [0.5, 1.0, 0.3],
      [0.2, 0.3, 1.0],
    ];
    const matrix: CorrelationMatrix = { factor_names, values };

    expect(matrix.factor_names).toHaveLength(3);
    expect(matrix.values).toHaveLength(3);
    for (const row of matrix.values) {
      expect(row).toHaveLength(matrix.factor_names.length);
    }
  });

  test('BacktestAnalysis has all required fields', () => {
    const analysis: BacktestAnalysis = {
      annualized_return: 0.15,
      annualized_benchmark: 0.10,
      excess_return: 0.05,
      sharpe_ratio: 1.5,
      information_ratio: 0.8,
      max_drawdown: -0.12,
      drawdown_duration_days: 45,
      total_cost: 0.02,
      mean_turnover: 0.08,
      max_turnover: 0.15,
      monthly_return_distribution: { Jan: 0.05, Feb: 0.03 },
      n_trading_days: 252,
      start_date: '2023-01-01',
      end_date: '2023-12-31',
    };
    expect(analysis.annualized_return).toBe(0.15);
    expect(analysis.monthly_return_distribution).toEqual({ Jan: 0.05, Feb: 0.03 });
  });

  test('PortfolioPosition has correct shape', () => {
    const position: PortfolioPosition = {
      instrument: 'AAPL',
      name: 'Apple Inc.',
      weight: 0.15,
      allocation: 15000,
      signal: 'Buy',
      entry_price: 180.5,
    };
    expect(position.instrument).toBe('AAPL');
    expect(position.weight).toBe(0.15);
  });

  test('MarketDataRow change_pct is optional', () => {
    const withoutChangePct: MarketDataRow = {
      date: '2024-01-01',
      instrument: 'AAPL',
      open: 180,
      high: 185,
      low: 179,
      close: 183,
      volume: 1000000,
      amount: 183000000,
      factor: 1.0,
    };
    expect(withoutChangePct.change_pct).toBeUndefined();

    const withChangePct: MarketDataRow = {
      ...withoutChangePct,
      change_pct: 1.67,
    };
    expect(withChangePct.change_pct).toBe(1.67);
  });

  test('FactorReturn has correct shape', () => {
    const factorReturn: FactorReturn = {
      factor_name: 'momentum_20d',
      return_contribution: 0.025,
    };
    expect(factorReturn.factor_name).toBe('momentum_20d');
    expect(factorReturn.return_contribution).toBe(0.025);
  });
});
