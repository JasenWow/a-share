import { describe, expect, test } from 'bun:test';
import React from 'react';
import { render } from 'ink-testing-library';
import { BacktestView } from '../../src/views/BacktestView.js';
import type { BacktestReport, BacktestAnalysis } from '@viz/core';

function makeAnalysis(overrides: Partial<BacktestAnalysis> = {}): BacktestAnalysis {
  return {
    annualized_return: 0.15,
    annualized_benchmark: 0.1,
    excess_return: 0.05,
    sharpe_ratio: 1.5,
    information_ratio: 1.2,
    max_drawdown: -0.08,
    drawdown_duration_days: 30,
    total_cost: 0.005,
    mean_turnover: 0.12,
    max_turnover: 0.5,
    monthly_return_distribution: {
      '2025-01': 0.03,
      '2025-02': -0.02,
      '2025-03': 0.05,
    },
    n_trading_days: 120,
    start_date: '2025-01-02',
    end_date: '2025-06-30',
    ...overrides,
  };
}

function makeReport(rowCount: number): BacktestReport {
  const rows = [];
  for (let i = 0; i < rowCount; i++) {
    rows.push({
      date: `2025-${String(Math.floor(i / 20) + 1).padStart(2, '0')}-${String((i % 20) + 1).padStart(2, '0')}`,
      return: 0.001 * (Math.sin(i) + 1),
      bench: 0.0005 * (Math.cos(i) + 1),
      cost: 0.0001,
      turnover: 0.1,
    });
  }
  return { rows };
}

function computeCumulativeReturns(report: BacktestReport): number[] {
  const cumulative: number[] = [];
  let running = 1.0;
  for (const row of report.rows) {
    running *= 1 + row.return;
    cumulative.push(running);
  }
  return cumulative;
}

function computeDrawdownSeries(cumulativeReturns: number[]): number[] {
  if (cumulativeReturns.length === 0) return [];
  const drawdowns: number[] = [];
  let peak = cumulativeReturns[0];
  for (const value of cumulativeReturns) {
    if (value > peak) peak = value;
    drawdowns.push((value - peak) / peak);
  }
  return drawdowns;
}

describe('BacktestView', () => {
  test('renders metrics table with formatted values', () => {
    const report = makeReport(20);
    const analysis = makeAnalysis();
    const cumulativeReturns = computeCumulativeReturns(report);
    const drawdownSeries = computeDrawdownSeries(cumulativeReturns);

    const { lastFrame, unmount } = render(
      <BacktestView
        report={report}
        analysis={analysis}
        cumulativeReturns={cumulativeReturns}
        drawdownSeries={drawdownSeries}
      />,
    );

    const output = lastFrame();
    expect(output).toContain('15.00%');
    expect(output).toContain('1.50');
    expect(output).toContain('-8.00%');
    expect(output).toContain('2025-01-02');
    expect(output).toContain('2025-06-30');
    unmount();
  });

  test('renders cumulative returns chart with ASCII characters', () => {
    const report = makeReport(20);
    const analysis = makeAnalysis();
    const cumulativeReturns = computeCumulativeReturns(report);
    const drawdownSeries = computeDrawdownSeries(cumulativeReturns);

    const { lastFrame, unmount } = render(
      <BacktestView
        report={report}
        analysis={analysis}
        cumulativeReturns={cumulativeReturns}
        drawdownSeries={drawdownSeries}
      />,
    );

    const output = lastFrame();
    expect(output).toContain('Cumulative Returns');
    expect(output).toContain('┤');
    unmount();
  });

  test('renders empty state when no rows', () => {
    const report: BacktestReport = { rows: [] };
    const { lastFrame, unmount } = render(
      <BacktestView report={report} analysis={null} cumulativeReturns={[]} drawdownSeries={[]} />,
    );

    const output = lastFrame();
    expect(output).toContain('No backtest data available');
    unmount();
  });

  test('renders drawdown chart section', () => {
    const report = makeReport(20);
    const analysis = makeAnalysis();
    const cumulativeReturns = computeCumulativeReturns(report);
    const drawdownSeries = computeDrawdownSeries(cumulativeReturns);

    const { lastFrame, unmount } = render(
      <BacktestView
        report={report}
        analysis={analysis}
        cumulativeReturns={cumulativeReturns}
        drawdownSeries={drawdownSeries}
      />,
    );

    const output = lastFrame();
    expect(output).toContain('Drawdown');
    unmount();
  });
});
