import { describe, expect, test, beforeAll, afterAll } from 'bun:test';
import { writeFileSync, unlinkSync, mkdirSync } from 'fs';
import { join } from 'path';
import { open } from '@evan/duckdb';
import type { KronosPrediction, LightGBMPrediction } from '../../src/types.js';
import {
  loadKronosSignals,
  loadLightGBMPredictions,
  rankByScore,
  mergeModelScores,
} from '../../src/loaders/prediction.js';

const TEST_DIR = '/tmp/prediction_test';

describe('loaders/prediction', () => {
  beforeAll(() => {
    mkdirSync(TEST_DIR, { recursive: true });
  });

  afterAll(() => {
    try {
      unlinkSync(join(TEST_DIR, 'kronos_test.csv'));
      unlinkSync(join(TEST_DIR, 'lightgbm_test.parquet'));
    } catch {
      // ignore cleanup errors
    }
  });

  test('loadKronosSignals with valid CSV', () => {
    const csvContent =
      'signal_date,instrument,score,score_pct\n2024-01-01,600000.SH,0.75,0.85\n2024-01-01,600519.SH,0.65,0.72\n2024-01-01,000001.SZ,0.55,0.60';
    writeFileSync(join(TEST_DIR, 'kronos_test.csv'), csvContent);

    const result = loadKronosSignals(join(TEST_DIR, 'kronos_test.csv'));

    expect(result).toHaveLength(3);
    expect(result[0].signal_date).toBe('2024-01-01');
    expect(result[0].instrument).toBe('600000.SH');
    expect(result[0].score).toBe(0.75);
    expect(result[0].score_pct).toBe(0.85);
    expect(result[1].instrument).toBe('600519.SH');
    expect(result[2].instrument).toBe('000001.SZ');
  });

  test('loadKronosSignals with missing file throws', () => {
    expect(() => loadKronosSignals('/nonexistent/path.csv')).toThrow();
  });

  test('loadLightGBMPredictions with valid parquet', () => {
    const db = open(':memory:');
    const conn = db.connect();

    conn.query(`
      CREATE TABLE lightgbm_data AS SELECT
        '2024-01-01' as date,
        '600000.SH' as instrument,
        0.8 as score
      UNION ALL
      SELECT '2024-01-01', '600519.SH', 0.7
      UNION ALL
      SELECT '2024-01-02', '600000.SH', 0.85
    `);

    const parquetPath = join(TEST_DIR, 'lightgbm_test.parquet');
    conn.query(`COPY (SELECT * FROM lightgbm_data) TO '${parquetPath}' (FORMAT PARQUET)`);
    conn.close();
    db.close();

    const result = loadLightGBMPredictions(parquetPath);

    expect(result).toHaveLength(3);
    expect(result[0].date).toBe('2024-01-01');
    expect(result[0].instrument).toBe('600000.SH');
    expect(result[0].score).toBe(0.8);
  });

  test('rankByScore ascending', () => {
    const predictions: KronosPrediction[] = [
      { signal_date: '2024-01-01', instrument: 'A', score: 0.8, score_pct: 0.9 },
      { signal_date: '2024-01-01', instrument: 'B', score: 0.3, score_pct: 0.4 },
      { signal_date: '2024-01-01', instrument: 'C', score: 0.6, score_pct: 0.7 },
    ];

    const result = rankByScore(predictions, 'asc');

    expect(result[0].instrument).toBe('B');
    expect(result[1].instrument).toBe('C');
    expect(result[2].instrument).toBe('A');
  });

  test('rankByScore descending', () => {
    const predictions: LightGBMPrediction[] = [
      { date: '2024-01-01', instrument: 'A', score: 0.8 },
      { date: '2024-01-01', instrument: 'B', score: 0.3 },
      { date: '2024-01-01', instrument: 'C', score: 0.6 },
    ];

    const result = rankByScore(predictions, 'desc');

    expect(result[0].instrument).toBe('A');
    expect(result[1].instrument).toBe('C');
    expect(result[2].instrument).toBe('B');
  });

  test('mergeModelScores with both models present', () => {
    const kronos: KronosPrediction[] = [
      { signal_date: '2024-01-01', instrument: 'A', score: 0.8, score_pct: 0.9 },
      { signal_date: '2024-01-01', instrument: 'B', score: 0.2, score_pct: 0.3 },
    ];

    const lightgbm: LightGBMPrediction[] = [
      { date: '2024-01-01', instrument: 'A', score: 0.6 },
      { date: '2024-01-01', instrument: 'B', score: 0.4 },
    ];

    const result = mergeModelScores(kronos, lightgbm);

    expect(result).toHaveLength(2);

    const aResult = result.find((r) => r.instrument === 'A')!;
    expect(aResult.kronos_score).toBe(0.8);
    expect(aResult.lightgbm_score).toBe(0.6);

    const bResult = result.find((r) => r.instrument === 'B')!;
    expect(bResult.kronos_score).toBe(0.2);
    expect(bResult.lightgbm_score).toBe(0.4);
  });

  test('mergeModelScores with only one model (empty other)', () => {
    const kronos: KronosPrediction[] = [
      { signal_date: '2024-01-01', instrument: 'A', score: 0.8, score_pct: 0.9 },
    ];

    const result = mergeModelScores(kronos, []);

    expect(result).toHaveLength(1);
    expect(result[0].instrument).toBe('A');
    expect(result[0].kronos_score).toBe(0.8);
    expect(result[0].lightgbm_score).toBe(0);
  });

  test('mergeModelScores normalization correctness', () => {
    const kronos: KronosPrediction[] = [
      { signal_date: '2024-01-01', instrument: 'MIN', score: 0, score_pct: 0.1 },
      { signal_date: '2024-01-01', instrument: 'MAX', score: 1, score_pct: 0.9 },
    ];

    const lightgbm: LightGBMPrediction[] = [
      { date: '2024-01-01', instrument: 'MIN', score: 0 },
      { date: '2024-01-01', instrument: 'MAX', score: 1 },
    ];

    const result = mergeModelScores(kronos, lightgbm);

    const minResult = result.find((r) => r.instrument === 'MIN')!;
    const maxResult = result.find((r) => r.instrument === 'MAX')!;

    // MIN: both normalized to 0, combined=0, mapped to -1
    expect(minResult.combined_score).toBeCloseTo(-1, 5);
    expect(minResult.signal).toBe('Strong Sell');

    // MAX: both normalized to 1, combined=1, mapped to 1
    expect(maxResult.combined_score).toBeCloseTo(1, 5);
    expect(maxResult.signal).toBe('Strong Buy');
  });

  test('mergeModelScores signal thresholds', () => {
    const kronos: KronosPrediction[] = [
      { signal_date: '2024-01-01', instrument: 'SELL', score: 0.1, score_pct: 0.2 },
      { signal_date: '2024-01-01', instrument: 'BUY', score: 0.9, score_pct: 0.95 },
    ];

    const lightgbm: LightGBMPrediction[] = [
      { date: '2024-01-01', instrument: 'SELL', score: 0.1 },
      { date: '2024-01-01', instrument: 'BUY', score: 0.9 },
    ];

    const result = mergeModelScores(kronos, lightgbm);

    const sellResult = result.find((r) => r.instrument === 'SELL')!;
    const buyResult = result.find((r) => r.instrument === 'BUY')!;

    // SELL: kronos=(0.1-0.1)/(0.9-0.1)=0, lightgbm=(0.1-0.1)/(0.9-0.1)=0, avg=0, mapped=-1
    expect(sellResult.combined_score).toBeCloseTo(-1, 5);
    expect(sellResult.signal).toBe('Strong Sell');

    // BUY: kronos=(0.9-0.1)/(0.9-0.1)=1, lightgbm=(0.9-0.1)/(0.9-0.1)=1, avg=1, mapped=1
    expect(buyResult.combined_score).toBeCloseTo(1, 5);
    expect(buyResult.signal).toBe('Strong Buy');
  });
});