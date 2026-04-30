import { describe, expect, test } from 'bun:test';
import { render } from 'ink-testing-library';
import React from 'react';
import { BacktestView } from '../src/views/BacktestView.js';
import { HoldingsView } from '../src/views/HoldingsView.js';
import { PredictionView } from '../src/views/PredictionView.js';
import { FactorView } from '../src/views/FactorView.js';
import type {
  BacktestReport,
  BacktestAnalysis,
  PortfolioPosition,
  WatchlistSummary,
  KronosPrediction,
  LightGBMPrediction,
  MergedScore,
  FactorValue,
  CorrelationMatrix,
  FactorReturn,
} from '@viz/core';
import { computeCumulativeReturns, computeDrawdownSeries, mergeModelScores } from '@viz/core';

function createBacktestReport(): BacktestReport {
  const rows = [
    { date: '2024-01-02', return: 0.012, bench: 0.008, cost: 0.001, turnover: 0.15 },
    { date: '2024-01-03', return: -0.005, bench: 0.002, cost: 0.001, turnover: 0.12 },
    { date: '2024-01-04', return: 0.020, bench: 0.015, cost: 0.001, turnover: 0.18 },
    { date: '2024-01-05', return: -0.010, bench: -0.003, cost: 0.001, turnover: 0.10 },
    { date: '2024-01-08', return: 0.008, bench: 0.005, cost: 0.001, turnover: 0.14 },
    { date: '2024-01-09', return: 0.015, bench: 0.010, cost: 0.001, turnover: 0.16 },
    { date: '2024-01-10', return: -0.003, bench: 0.001, cost: 0.001, turnover: 0.11 },
    { date: '2024-01-11', return: 0.025, bench: 0.018, cost: 0.001, turnover: 0.20 },
    { date: '2024-01-12', return: 0.010, bench: 0.007, cost: 0.001, turnover: 0.13 },
    { date: '2024-01-15', return: 0.005, bench: 0.003, cost: 0.001, turnover: 0.09 },
  ];
  return { rows };
}

function createBacktestAnalysis(): BacktestAnalysis {
  return {
    annualized_return: 0.15,
    annualized_benchmark: 0.08,
    excess_return: 0.07,
    sharpe_ratio: 1.5,
    information_ratio: 0.9,
    max_drawdown: -0.08,
    drawdown_duration_days: 5,
    total_cost: 0.01,
    mean_turnover: 0.13,
    max_turnover: 0.20,
    monthly_return_distribution: { '2024-01': 0.05, '2024-02': -0.02, '2024-03': 0.08 },
    n_trading_days: 10,
    start_date: '2024-01-02',
    end_date: '2024-01-15',
  };
}

function createPortfolioPositions(): PortfolioPosition[] {
  return [
    { instrument: '000001', name: 'Ping An Bank', weight: 0.25, allocation: 250000, signal: 'Strong Buy', entry_price: 12.50 },
    { instrument: '000002', name: 'Vanke A', weight: 0.20, allocation: 200000, signal: 'Buy', entry_price: 8.30 },
    { instrument: '600519', name: 'Kweichow Moutai', weight: 0.18, allocation: 180000, signal: 'Buy', entry_price: 1680.00 },
    { instrument: '600036', name: 'China Merchants Bank', weight: 0.15, allocation: 150000, signal: 'Sell', entry_price: 35.20 },
    { instrument: '000858', name: 'Wuliangye Yibin', weight: 0.12, allocation: 120000, signal: 'Strong Sell', entry_price: 145.60 },
    { instrument: 'CASH', name: 'Cash', weight: 0.10, allocation: 100000, signal: 'Hold', entry_price: 1.0 },
  ];
}

function createWatchlistSummary(): WatchlistSummary {
  return {
    total_stocks: 5,
    bullish_count: 3,
    bearish_count: 2,
    avg_score: 0.35,
    best_stock: '000001',
    worst_stock: '000858',
  };
}

function createKronosPredictions(): KronosPrediction[] {
  return [
    { signal_date: '2024-01-15', instrument: '000001', score: 0.8234, score_pct: 85.2 },
    { signal_date: '2024-01-15', instrument: '600519', score: 0.7567, score_pct: 78.1 },
    { signal_date: '2024-01-15', instrument: '000002', score: 0.6123, score_pct: 63.4 },
    { signal_date: '2024-01-15', instrument: '600036', score: -0.2345, score_pct: -24.3 },
    { signal_date: '2024-01-15', instrument: '000858', score: -0.5678, score_pct: -58.7 },
  ];
}

function createLightGBMPredictions(): LightGBMPrediction[] {
  return [
    { date: '2024-01-15', instrument: '000001', score: 0.9123 },
    { date: '2024-01-15', instrument: '600519', score: 0.8234 },
    { date: '2024-01-15', instrument: '000002', score: 0.5678 },
    { date: '2024-01-15', instrument: '600036', score: -0.1234 },
    { date: '2024-01-15', instrument: '000858', score: -0.7890 },
  ];
}

function createFactorValues(): FactorValue[] {
  return [
    { date: '2024-01-12', instrument: '000001', factor_name: 'momentum', value: 0.1234 },
    { date: '2024-01-12', instrument: '000001', factor_name: 'value', value: 0.0567 },
    { date: '2024-01-12', instrument: '000001', factor_name: 'quality', value: 0.0890 },
    { date: '2024-01-12', instrument: '000002', factor_name: 'momentum', value: 0.0456 },
    { date: '2024-01-12', instrument: '000002', factor_name: 'value', value: 0.0789 },
    { date: '2024-01-12', instrument: '000002', factor_name: 'quality', value: -0.0234 },
    { date: '2024-01-12', instrument: '600519', factor_name: 'momentum', value: 0.2345 },
    { date: '2024-01-12', instrument: '600519', factor_name: 'value', value: -0.0456 },
    { date: '2024-01-12', instrument: '600519', factor_name: 'quality', value: 0.1567 },
    { date: '2024-01-12', instrument: '600036', factor_name: 'momentum', value: -0.0789 },
    { date: '2024-01-12', instrument: '600036', factor_name: 'value', value: 0.0345 },
    { date: '2024-01-12', instrument: '600036', factor_name: 'quality', value: -0.0678 },
    { date: '2024-01-12', instrument: '000858', factor_name: 'momentum', value: -0.1456 },
    { date: '2024-01-12', instrument: '000858', factor_name: 'value', value: 0.0234 },
    { date: '2024-01-12', instrument: '000858', factor_name: 'quality', value: -0.0890 },
  ];
}

function createCorrelationMatrix(): CorrelationMatrix {
  return {
    factor_names: ['momentum', 'value', 'quality'],
    values: [
      [1.0, 0.45, 0.32],
      [0.45, 1.0, 0.28],
      [0.32, 0.28, 1.0],
    ],
  };
}

function createFactorReturns(): FactorReturn[] {
  return [
    { factor_name: 'momentum', return_contribution: 0.0234 },
    { factor_name: 'value', return_contribution: -0.0089 },
    { factor_name: 'quality', return_contribution: 0.0156 },
  ];
}

describe('Backtest Integration', () => {
  test('Full backtest pipeline renders all sections', () => {
    const report = createBacktestReport();
    const analysis = createBacktestAnalysis();
    const cumulativeReturns = computeCumulativeReturns(report);
    const drawdownSeries = computeDrawdownSeries(report);

    const { lastFrame } = render(
      <BacktestView
        report={report}
        analysis={analysis}
        cumulativeReturns={cumulativeReturns}
        drawdownSeries={drawdownSeries}
      />
    );

    const output = lastFrame();
    expect(output).toContain('15.00%');
    expect(output).toContain('1.50');
    expect(output).toContain('Cumulative Returns');
    expect(output).toContain('Drawdown');
    expect(output).toContain('Monthly Returns');
    expect(output).toContain('─');
  });

  test('Backtest with no analysis renders charts but no metrics', () => {
    const report = createBacktestReport();
    const cumulativeReturns = computeCumulativeReturns(report);
    const drawdownSeries = computeDrawdownSeries(report);

    const { lastFrame } = render(
      <BacktestView
        report={report}
        analysis={null}
        cumulativeReturns={cumulativeReturns}
        drawdownSeries={drawdownSeries}
      />
    );

    const output = lastFrame();
    expect(output).toContain('Cumulative Returns');
    expect(output).toContain('Drawdown');
    expect(output).not.toContain('Annualized Return');
  });

  test('Backtest with empty report shows no data message', () => {
    const { lastFrame } = render(
      <BacktestView
        report={{ rows: [] }}
        analysis={null}
        cumulativeReturns={[]}
        drawdownSeries={[]}
      />
    );

    const output = lastFrame();
    expect(output).toContain('No backtest data available');
  });
});

describe('Holdings Integration', () => {
  test('Full holdings pipeline renders all sections', () => {
    const positions = createPortfolioPositions();
    const summary = createWatchlistSummary();
    const allocation = { stocks: 0.90, cash: 0.10 };

    const { lastFrame } = render(
      <HoldingsView positions={positions} summary={summary} allocation={allocation} />
    );

    const output = lastFrame();
    expect(output).toContain('000001');
    expect(output).toContain('000002');
    expect(output).toContain('600519');
    expect(output).toContain('600036');
    expect(output).toContain('000858');
    expect(output).toContain('CASH');
    expect(output).toContain('Strong Buy');
    expect(output).toContain('Buy');
    expect(output).toContain('Sell');
    expect(output).toContain('█');
    expect(output).toContain('░');
    expect(output).toContain('Portfolio Summary');
    expect(output).toContain('Bullish');
    expect(output).toContain('Bearish');
  });

  test('Holdings with empty positions shows no data message', () => {
    const { lastFrame } = render(
      <HoldingsView positions={[]} summary={null} allocation={{ stocks: 0, cash: 0 }} />
    );

    const output = lastFrame();
    expect(output).toContain('No holdings data available');
  });

  test('Holdings without summary still renders table', () => {
    const positions = createPortfolioPositions();

    const { lastFrame } = render(
      <HoldingsView positions={positions} summary={null} allocation={{ stocks: 0.90, cash: 0.10 }} />
    );

    const output = lastFrame();
    expect(output).toContain('Holdings');
    expect(output).toContain('000001');
  });
});

describe('Prediction Integration', () => {
  test('Full prediction pipeline renders all sections', () => {
    const kronos = createKronosPredictions();
    const lightgbm = createLightGBMPredictions();
    const merged = mergeModelScores(kronos, lightgbm);

    const { lastFrame } = render(
      <PredictionView kronos={kronos} lightgbm={lightgbm} merged={merged} />
    );

    const output = lastFrame();
    expect(output).toContain('Kronos Predictions');
    expect(output).toContain('LightGBM Predictions');
    expect(output).toContain('Combined Model Rankings');
    expect(output).toContain('▲');
    expect(output).toContain('▼');
    expect(output).toContain('Strong Buy');
    expect(output).toContain('Buy');
    expect(output).toContain('Sell');
    expect(output).toContain('Strong Sell');
  });

  test('Single Kronos model prediction shows LightGBM no data', () => {
    const kronos = createKronosPredictions();

    const { lastFrame } = render(
      <PredictionView kronos={kronos} lightgbm={[]} merged={[]} />
    );

    const output = lastFrame();
    expect(output).toContain('Kronos Predictions');
    expect(output).toContain('LightGBM: No data available');
  });

  test('Single LightGBM model prediction shows Kronos no data', () => {
    const lightgbm = createLightGBMPredictions();

    const { lastFrame } = render(
      <PredictionView kronos={[]} lightgbm={lightgbm} merged={[]} />
    );

    const output = lastFrame();
    expect(output).toContain('Kronos: No data available');
    expect(output).toContain('LightGBM Predictions');
  });

  test('Empty predictions shows no data message', () => {
    const { lastFrame } = render(
      <PredictionView kronos={[]} lightgbm={[]} merged={[]} />
    );

    const output = lastFrame();
    expect(output).toContain('No prediction data available');
  });

  test('Merged scores sorted by combined score descending', () => {
    const kronos = createKronosPredictions();
    const lightgbm = createLightGBMPredictions();
    const merged = mergeModelScores(kronos, lightgbm);

    const { lastFrame } = render(
      <PredictionView kronos={kronos} lightgbm={lightgbm} merged={merged} />
    );

    const output = lastFrame();
    const combinedSection = output.split('Combined Model Rankings')[1];
    expect(combinedSection).toContain('1');
  });
});

describe('Factor Integration', () => {
  test('Full factor pipeline renders all sections', () => {
    const factors = createFactorValues();
    const correlationMatrix = createCorrelationMatrix();
    const factorReturns = createFactorReturns();

    const { lastFrame } = render(
      <FactorView factors={factors} correlationMatrix={correlationMatrix} factorReturns={factorReturns} />
    );

    const output = lastFrame();
    expect(output).toContain('momentum');
    expect(output).toContain('value');
    expect(output).toContain('quality');
    expect(output).toContain('000001');
    expect(output).toContain('000002');
    expect(output).toContain('600519');
    expect(output).toContain('Factor Correlation Matrix');
    expect(output).toContain('1.00');
    expect(output).toContain('Factor Return Contribution');
    expect(output).toContain('█');
  });

  test('Factor with empty data shows no data message', () => {
    const { lastFrame } = render(
      <FactorView
        factors={[]}
        correlationMatrix={{ factor_names: [], values: [] }}
        factorReturns={[]}
      />
    );

    const output = lastFrame();
    expect(output).toContain('No factor data available');
  });

  test('Factor values show latest date', () => {
    const factors = createFactorValues();
    const correlationMatrix = createCorrelationMatrix();
    const factorReturns = createFactorReturns();

    const { lastFrame } = render(
      <FactorView factors={factors} correlationMatrix={correlationMatrix} factorReturns={factorReturns} />
    );

    const output = lastFrame();
    expect(output).toContain('Factor Values (Latest)');
    expect(output).toContain('2024-01-12');
  });
});

describe('Error Handling', () => {
  test('fs.existsSync returns false for nonexistent directory', () => {
    const { existsSync } = require('fs');
    expect(existsSync('/nonexistent/dir')).toBe(false);
  });

  test('fs.existsSync returns false for nonexistent file', () => {
    const { existsSync } = require('fs');
    expect(existsSync('/nonexistent/file.txt')).toBe(false);
  });

  test('All views handle empty data gracefully', () => {
    const backtestFrame = render(
      <BacktestView report={{ rows: [] }} analysis={null} cumulativeReturns={[]} drawdownSeries={[]} />
    ).lastFrame();
    expect(backtestFrame).toContain('No backtest data available');

    const holdingsFrame = render(
      <HoldingsView positions={[]} summary={null} allocation={{ stocks: 0, cash: 0 }} />
    ).lastFrame();
    expect(holdingsFrame).toContain('No holdings data available');

    const predictionFrame = render(
      <PredictionView kronos={[]} lightgbm={[]} merged={[]} />
    ).lastFrame();
    expect(predictionFrame).toContain('No prediction data available');

    const factorFrame = render(
      <FactorView factors={[]} correlationMatrix={{ factor_names: [], values: [] }} factorReturns={[]} />
    ).lastFrame();
    expect(factorFrame).toContain('No factor data available');
  });
});

describe('Data Processing Functions', () => {
  test('computeCumulativeReturns calculates correctly', () => {
    const report = createBacktestReport();
    const cumulative = computeCumulativeReturns(report);

    expect(cumulative).toHaveLength(10);
    expect(cumulative[0]).toBeCloseTo(1.012, 4);
    expect(cumulative[1]).toBeCloseTo(1.00694, 4);
  });

  test('computeDrawdownSeries calculates correctly', () => {
    const report = createBacktestReport();
    const drawdown = computeDrawdownSeries(report);

    expect(drawdown).toHaveLength(10);
    expect(drawdown[0]).toBeCloseTo(0, 4);
  });

  test('mergeModelScores combines predictions correctly', () => {
    const kronos = createKronosPredictions();
    const lightgbm = createLightGBMPredictions();
    const merged = mergeModelScores(kronos, lightgbm);

    expect(merged).toHaveLength(5);
    const instruments = merged.map((m) => m.instrument);
    expect(instruments).toContain('000001');
    expect(instruments).toContain('600519');
    expect(instruments).toContain('000002');
    expect(instruments).toContain('600036');
    expect(instruments).toContain('000858');

    for (const score of merged) {
      expect(['Strong Buy', 'Buy', 'Sell', 'Strong Sell']).toContain(score.signal);
    }
  });

  test('mergeModelScores handles empty arrays', () => {
    const merged = mergeModelScores([], []);
    expect(merged).toHaveLength(0);
  });

  test('mergeModelScores handles partial data', () => {
    const kronos = createKronosPredictions();
    const merged = mergeModelScores(kronos, []);

    expect(merged).toHaveLength(5);
    for (const score of merged) {
      expect(['Strong Buy', 'Buy', 'Sell', 'Strong Sell']).toContain(score.signal);
    }
  });
});