import { describe, expect, test } from 'bun:test';
import React from 'react';
import { render } from 'ink-testing-library';
import { HoldingsView } from '../../src/views/HoldingsView.js';
import type { PortfolioPosition, WatchlistSummary } from '@viz/core';

const stockPositions: PortfolioPosition[] = [
  { instrument: '600519', name: '贵州茅台', weight: 0.30, allocation: 300000, signal: 'Strong Buy', entry_price: 1800.5 },
  { instrument: '000858', name: '五粮液', weight: 0.20, allocation: 200000, signal: 'Buy', entry_price: 150.2 },
  { instrument: '601318', name: '中国平安', weight: 0.10, allocation: 100000, signal: 'Sell', entry_price: 45.8 },
  { instrument: 'CASH', name: 'Cash', weight: 0.40, allocation: 400000, signal: 'Hold', entry_price: 0 },
];

const sampleSummary: WatchlistSummary = {
  total_stocks: 5,
  bullish_count: 3,
  bearish_count: 2,
  avg_score: 0.35,
  best_stock: '600519',
  worst_stock: '601318',
};

const sampleAllocation = { stocks: 0.6, cash: 0.4 };

describe('HoldingsView', () => {
  test('renders holdings table with instrument codes', () => {
    const { lastFrame } = render(
      <HoldingsView positions={stockPositions} summary={null} allocation={sampleAllocation} />,
    );
    const output = lastFrame()!;
    expect(output).toContain('600519');
    expect(output).toContain('000858');
    expect(output).toContain('601318');
    expect(output).toContain('CASH');
  });

  test('renders weight percentages', () => {
    const { lastFrame } = render(
      <HoldingsView positions={stockPositions} summary={null} allocation={sampleAllocation} />,
    );
    const output = lastFrame()!;
    expect(output).toContain('30.0%');
    expect(output).toContain('20.0%');
    expect(output).toContain('40.0%');
  });

  test('renders signal labels', () => {
    const { lastFrame } = render(
      <HoldingsView positions={stockPositions} summary={null} allocation={sampleAllocation} />,
    );
    const output = lastFrame()!;
    expect(output).toContain('Strong Buy');
    expect(output).toContain('Buy');
    expect(output).toContain('Sell');
    expect(output).toContain('Hold');
  });

  test('renders weight distribution bars with block characters', () => {
    const { lastFrame } = render(
      <HoldingsView positions={stockPositions} summary={null} allocation={sampleAllocation} />,
    );
    const output = lastFrame()!;
    expect(output).toContain('█');
    expect(output).toContain('░');
  });

  test('renders empty portfolio message', () => {
    const { lastFrame } = render(
      <HoldingsView positions={[]} summary={null} allocation={{ stocks: 0, cash: 1 }} />,
    );
    const output = lastFrame()!;
    expect(output).toContain('No holdings data available');
  });

  test('renders summary header when summary provided', () => {
    const { lastFrame } = render(
      <HoldingsView positions={stockPositions} summary={sampleSummary} allocation={sampleAllocation} />,
    );
    const output = lastFrame()!;
    expect(output).toContain('Total Stocks: 5');
    expect(output).toContain('Bullish: 3');
    expect(output).toContain('Bearish: 2');
    expect(output).toContain('Avg Score: 0.35');
    expect(output).toContain('600519');
    expect(output).toContain('601318');
  });

  test('renders allocation summary', () => {
    const { lastFrame } = render(
      <HoldingsView positions={stockPositions} summary={null} allocation={sampleAllocation} />,
    );
    const output = lastFrame()!;
    expect(output).toContain('60.0%');
    expect(output).toContain('40.0%');
  });

  test('signal color coding — colored signal labels render without error', () => {
    const { lastFrame, unmount } = render(
      <HoldingsView positions={stockPositions} summary={null} allocation={sampleAllocation} />,
    );
    const output = lastFrame()!;
    expect(output).toContain('Strong Buy');
    expect(output).toContain('Buy');
    expect(output).toContain('Sell');
    expect(output).toContain('Hold');
    unmount();
  });
});
