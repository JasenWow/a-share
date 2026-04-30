import { readFileSync } from 'fs';
import { open } from '@evan/duckdb';
import type { KronosPrediction, LightGBMPrediction, MergedScore } from '../types.js';

/**
 * Load Kronos signals from CSV file.
 * CSV format: signal_date,instrument,score,score_pct (with header row)
 */
export function loadKronosSignals(filePath: string): KronosPrediction[] {
  const text = readFileSync(filePath, 'utf-8');
  const lines = text.trim().split('\n').slice(1); // skip header

  return lines.map((line) => {
    const [signal_date, instrument, score, score_pct]: string[] = line.split(',');
    return {
      signal_date,
      instrument,
      score: parseFloat(score),
      score_pct: parseFloat(score_pct),
    };
  });
}

/**
 * Load LightGBM predictions from parquet file.
 * Parquet has MultiIndex (datetime, instrument), column: score
 * DuckDB will flatten this into columns we can read.
 */
export function loadLightGBMPredictions(filePath: string): LightGBMPrediction[] {
  const db = open(':memory:');
  const conn = db.connect();

  // Read parquet - DuckDB will handle MultiIndex flattening automatically
  const result = conn.query(
    `SELECT CAST(date AS VARCHAR) as date, CAST(instrument AS VARCHAR) as instrument, CAST(score AS DOUBLE) as score FROM '${filePath}'`
  ) as Record<string, unknown>[];

  const rows: LightGBMPrediction[] = result.map((row) => ({
    date: String(row.date),
    instrument: String(row.instrument),
    score: Number(row.score),
  }));

  conn.close();
  db.close();

  return rows;
}

/**
 * Sort predictions by score in given direction.
 */
export function rankByScore<T extends KronosPrediction | LightGBMPrediction>(
  predictions: T[],
  direction: 'asc' | 'desc' = 'desc'
): T[] {
  return [...predictions].sort((a, b) => {
    if (direction === 'asc') {
      return a.score - b.score;
    }
    return b.score - a.score;
  });
}

/**
 * Merge Kronos + LightGBM scores into combined rankings.
 * Normalizes both scores to 0-1 range, averages them, maps to -1..1.
 * Signal: Strong Buy (>0.5), Buy (0-0.5), Sell (-0.5-0), Strong Sell (<-0.5)
 */
export function mergeModelScores(
  kronos: KronosPrediction[],
  lightgbm: LightGBMPrediction[]
): MergedScore[] {
  // Build lookup maps by instrument
  const kronosMap = new Map(kronos.map((k) => [k.instrument, k.score]));
  const lightgbmMap = new Map(lightgbm.map((l) => [l.instrument, l.score]));

  // Get all instruments
  const allInstruments = new Set([...kronosMap.keys(), ...lightgbmMap.keys()]);

  // Get score arrays for normalization
  const kronosScores = Array.from(kronosMap.values());
  const lightgbmScores = Array.from(lightgbmMap.values());

  const normalize = (score: number, min: number, max: number): number => {
    if (max === min) return 0.5;
    return (score - min) / (max - min);
  };

  const kronosMin = kronosScores.length > 0 ? Math.min(...kronosScores) : 0;
  const kronosMax = kronosScores.length > 0 ? Math.max(...kronosScores) : 1;
  const lightgbmMin = lightgbmScores.length > 0 ? Math.min(...lightgbmScores) : 0;
  const lightgbmMax = lightgbmScores.length > 0 ? Math.max(...lightgbmScores) : 1;

  const merged: MergedScore[] = [];

  for (const instrument of allInstruments) {
    const kronosScore = kronosMap.get(instrument) ?? null;
    const lightgbmScore = lightgbmMap.get(instrument) ?? null;

    const kronosNorm =
      kronosScore !== null ? normalize(kronosScore, kronosMin, kronosMax) : null;
    const lightgbmNorm =
      lightgbmScore !== null ? normalize(lightgbmScore, lightgbmMin, lightgbmMax) : null;

    let combinedScore: number;
    let combinedNorm: number;

    if (kronosNorm !== null && lightgbmNorm !== null) {
      combinedNorm = (kronosNorm + lightgbmNorm) / 2;
    } else if (kronosNorm !== null) {
      combinedNorm = kronosNorm;
    } else {
      combinedNorm = lightgbmNorm!;
    }

    // Map from 0-1 to -1 to 1
    combinedScore = combinedNorm * 2 - 1;

    // Determine signal based on combinedScore
    let signal: 'Strong Buy' | 'Buy' | 'Sell' | 'Strong Sell';
    if (combinedScore > 0.5) {
      signal = 'Strong Buy';
    } else if (combinedScore > 0) {
      signal = 'Buy';
    } else if (combinedScore > -0.5) {
      signal = 'Sell';
    } else {
      signal = 'Strong Sell';
    }

    merged.push({
      instrument,
      kronos_score: kronosScore ?? 0,
      lightgbm_score: lightgbmScore ?? 0,
      combined_score: combinedScore,
      signal,
    });
  }

  return merged;
}