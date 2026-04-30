import { describe, expect, test } from 'bun:test';
import { parseWatchlistSummary, computePortfolioAllocation } from '../../src/loaders/watchlist.js';
import type { PortfolioPosition } from '../../src/types.js';

describe('parseWatchlistSummary', () => {
  test('parse watchlist summary with valid data', () => {
    const rawData = {
      total_stocks: 10,
      bullish_count: 6,
      bearish_count: 4,
      avg_score: 0.55,
      best_stock: 'AAPL',
      worst_stock: 'TSLA',
    };

    const summary = parseWatchlistSummary(rawData);
    expect(summary.total_stocks).toBe(10);
    expect(summary.bullish_count).toBe(6);
    expect(summary.bearish_count).toBe(4);
    expect(summary.avg_score).toBe(0.55);
    expect(summary.best_stock).toBe('AAPL');
    expect(summary.worst_stock).toBe('TSLA');
  });

  test('parse watchlist summary with missing fields defaults', () => {
    const rawData = {};

    const summary = parseWatchlistSummary(rawData);
    expect(summary.total_stocks).toBe(0);
    expect(summary.bullish_count).toBe(0);
    expect(summary.bearish_count).toBe(0);
    expect(summary.avg_score).toBe(0);
    expect(summary.best_stock).toBeNull();
    expect(summary.worst_stock).toBeNull();
  });

  test('parse watchlist summary with null best_stock and worst_stock', () => {
    const rawData = {
      total_stocks: 5,
      bullish_count: 3,
      bearish_count: 2,
      avg_score: 0.45,
      best_stock: null,
      worst_stock: null,
    };

    const summary = parseWatchlistSummary(rawData);
    expect(summary.best_stock).toBeNull();
    expect(summary.worst_stock).toBeNull();
  });
});

describe('computePortfolioAllocation', () => {
  test('compute allocation with mixed stocks+cash', () => {
    const positions: PortfolioPosition[] = [
      { instrument: 'AAPL', name: 'Apple', weight: 0.4, allocation: 40000, signal: 'Buy', entry_price: 180 },
      { instrument: 'GOOGL', name: 'Google', weight: 0.3, allocation: 30000, signal: 'Strong Buy', entry_price: 140 },
      { instrument: 'CASH', name: '现金', weight: 0.3, allocation: 30000, signal: 'Buy', entry_price: 1 },
    ];

    const allocation = computePortfolioAllocation(positions);
    expect(allocation.stocks).toBeCloseTo(0.7);
    expect(allocation.cash).toBeCloseTo(0.3);
  });

  test('compute allocation with no CASH row', () => {
    const positions: PortfolioPosition[] = [
      { instrument: 'AAPL', name: 'Apple', weight: 0.4, allocation: 40000, signal: 'Buy', entry_price: 180 },
      { instrument: 'GOOGL', name: 'Google', weight: 0.3, allocation: 30000, signal: 'Strong Buy', entry_price: 140 },
    ];

    const allocation = computePortfolioAllocation(positions);
    expect(allocation.stocks).toBe(0.7);
    expect(allocation.cash).toBe(0);
  });

  test('compute allocation with empty positions returns all cash', () => {
    const positions: PortfolioPosition[] = [];
    const allocation = computePortfolioAllocation(positions);
    expect(allocation.stocks).toBe(0);
    expect(allocation.cash).toBe(1);
  });

  test('compute allocation with only CASH row', () => {
    const positions: PortfolioPosition[] = [
      { instrument: 'CASH', name: '现金', weight: 1.0, allocation: 100000, signal: 'Buy', entry_price: 1 },
    ];

    const allocation = computePortfolioAllocation(positions);
    expect(allocation.stocks).toBe(0);
    expect(allocation.cash).toBe(1);
  });
});