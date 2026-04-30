import { open } from '@evan/duckdb';
import type { PortfolioPosition, WatchlistSummary } from '../types.js';

export function loadPortfolio(filePath: string): PortfolioPosition[] {
  const db = open(filePath);
  const conn = db.connect();
  try {
    const result = conn.query(`SELECT instrument, name, weight, allocation, signal, entry_price FROM '${filePath}'`);
    return result as PortfolioPosition[];
  } finally {
    conn.close();
    db.close();
  }
}

export function parseWatchlistSummary(rawData: Record<string, unknown>): WatchlistSummary {
  return {
    total_stocks: Number(rawData.total_stocks) || 0,
    bullish_count: Number(rawData.bullish_count) || 0,
    bearish_count: Number(rawData.bearish_count) || 0,
    avg_score: Number(rawData.avg_score) || 0,
    best_stock: rawData.best_stock != null ? String(rawData.best_stock) : null,
    worst_stock: rawData.worst_stock != null ? String(rawData.worst_stock) : null,
  };
}

export function computePortfolioAllocation(positions: PortfolioPosition[]): { stocks: number; cash: number } {
  if (positions.length === 0) {
    return { stocks: 0, cash: 1 };
  }

  let stocksWeight = 0;
  let hasCash = false;

  for (const pos of positions) {
    if (pos.instrument === 'CASH') {
      hasCash = true;
    } else {
      stocksWeight += pos.weight;
    }
  }

  const stocks = stocksWeight;
  const cash = hasCash ? 1 - stocksWeight : 0;
  return { stocks, cash };
}