import { describe, expect, test, beforeAll, afterAll } from 'bun:test';
import { unlinkSync, mkdirSync } from 'fs';
import { join } from 'path';
import { open } from '@evan/duckdb';
import type { BacktestReport } from '../../src/types.js';
import {
  loadBacktestReport,
  computeCumulativeReturns,
  computeDrawdownSeries,
} from '../../src/loaders/backtest.js';

const TEST_DIR = '/tmp/backtest_test_parquet';

describe('loaders/backtest', () => {
  beforeAll(() => {
    mkdirSync(TEST_DIR, { recursive: true });
  });

  afterAll(() => {
    try {
      unlinkSync(join(TEST_DIR, 'test_report.parquet'));
      unlinkSync(join(TEST_DIR, 'test_analysis.parquet'));
      unlinkSync(join(TEST_DIR, 'nonexistent.parquet'));
    } catch {
      // ignore cleanup errors
    }
  });

  test('loadBacktestReport with valid parquet data', () => {
    const db = open(':memory:');
    const conn = db.connect();

    conn.query(`
      CREATE TABLE backtest_data AS SELECT
        '2024-01-01' as date,
        0.05 as return_val,
        0.03 as bench,
        0.001 as cost,
        0.1 as turnover
      UNION ALL
      SELECT '2024-01-02', 0.02, 0.01, 0.001, 0.05
      UNION ALL
      SELECT '2024-01-03', -0.01, 0.005, 0.001, 0.08
    `);

    const reportPath = join(TEST_DIR, 'test_report.parquet');
    conn.query(`COPY (SELECT date, return_val, bench, cost, turnover FROM backtest_data) TO '${reportPath}' (FORMAT PARQUET)`);
    conn.close();
    db.close();

    const report = loadBacktestReport(reportPath);

    expect(report.rows).toHaveLength(3);
    expect(report.rows[0].date).toBe('2024-01-01');
    expect(report.rows[0].return).toBe(0.05);
    expect(report.rows[0].bench).toBe(0.03);
    expect(report.rows[0].cost).toBe(0.001);
    expect(report.rows[0].turnover).toBe(0.1);
    expect(report.rows[1].return).toBe(0.02);
    expect(report.rows[2].return).toBe(-0.01);
  });

  test('loadBacktestReport throws for missing file', () => {
    const nonexistentPath = join(TEST_DIR, 'nonexistent.parquet');

    expect(() => loadBacktestReport(nonexistentPath)).toThrow(
      `Backtest report file not found: ${nonexistentPath}`
    );
  });

  test('computeCumulativeReturns with returns [0.1, 0.05, -0.02]', () => {
    const report: BacktestReport = {
      rows: [
        { date: '2024-01-01', return: 0.1, bench: 0.03, cost: 0, turnover: 0 },
        { date: '2024-01-02', return: 0.05, bench: 0.02, cost: 0, turnover: 0 },
        { date: '2024-01-03', return: -0.02, bench: 0.01, cost: 0, turnover: 0 },
      ],
    };

    const cumulative = computeCumulativeReturns(report);

    expect(cumulative).toHaveLength(3);
    expect(cumulative[0]).toBeCloseTo(1.1, 5);
    expect(cumulative[1]).toBeCloseTo(1.155, 5);
    expect(cumulative[2]).toBeCloseTo(1.1319, 3);
  });

  test('computeDrawdownSeries with cumulative [1.1, 1.2, 1.05, 1.15]', () => {
    const report: BacktestReport = {
      rows: [
        { date: '2024-01-01', return: 0.1, bench: 0.03, cost: 0, turnover: 0 },
        { date: '2024-01-02', return: 0.0909, bench: 0.03, cost: 0, turnover: 0 },
        { date: '2024-01-03', return: -0.125, bench: 0.03, cost: 0, turnover: 0 },
        { date: '2024-01-04', return: 0.0952, bench: 0.03, cost: 0, turnover: 0 },
      ],
    };

    const drawdowns = computeDrawdownSeries(report);

    expect(drawdowns).toHaveLength(4);
    expect(drawdowns[0]).toBe(0);
    expect(drawdowns[1]).toBe(0);
    expect(drawdowns[2]).toBeLessThan(0);
    expect(drawdowns[3]).toBeLessThan(0);
  });

  test('computeCumulativeReturns returns empty array for empty report', () => {
    const report: BacktestReport = { rows: [] };
    const cumulative = computeCumulativeReturns(report);
    expect(cumulative).toHaveLength(0);
  });

  test('computeDrawdownSeries returns empty array for empty report', () => {
    const report: BacktestReport = { rows: [] };
    const drawdowns = computeDrawdownSeries(report);
    expect(drawdowns).toHaveLength(0);
  });
});