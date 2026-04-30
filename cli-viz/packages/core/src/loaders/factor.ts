import { open } from '@evan/duckdb';
import type { FactorValue, CorrelationMatrix, FactorReturn } from '../types.js';

/**
 * Load factor data from parquet file.
 * Factor data typically has columns: date, instrument, factor1_name, factor2_name, ...
 * Reshape from wide to long format: one row per (date, instrument, factor_name, value)
 */
export function loadFactorData(filePath: string): FactorValue[] {
  const db = open(':memory:');
  const conn = db.connect();
  try {
    const columns = conn.query(`SELECT name FROM parquet_schema('${filePath}')`) as Record<
      string,
      unknown
    >[];
    const colNames = columns
      .map((row) => String(row.name))
      .filter((name) => name !== 'duckdb_schema');

    const factorColumns = colNames.filter((c) => c !== 'date' && c !== 'instrument');

    if (factorColumns.length === 0) {
      return [];
    }

    const selectClause = colNames
      .map((col) => `CAST(${col} AS VARCHAR) AS ${col}`)
      .join(', ');
    const result = conn.query(`SELECT ${selectClause} FROM '${filePath}'`) as Record<
      string,
      unknown
    >[];

    if (!result || result.length === 0) {
      return [];
    }

    const resultList: FactorValue[] = [];
    for (const row of result) {
      const date = String(row.date ?? '');
      const instrument = String(row.instrument ?? '');
      for (const factorName of factorColumns) {
        const value = row[factorName];
        if (value !== null && value !== undefined) {
          resultList.push({
            date,
            instrument,
            factor_name: factorName,
            value: Number(value),
          });
        }
      }
    }

    return resultList;
  } finally {
    conn.close();
    db.close();
  }
}

/**
 * Compute Pearson correlation matrix between factors.
 * Input: FactorValue[] in long format
 * Output: CorrelationMatrix with factor_names and square values matrix
 * Diagonal must be 1.0, matrix must be symmetric
 */
export function computeCorrelationMatrix(factors: FactorValue[]): CorrelationMatrix {
  if (factors.length === 0) {
    return { factor_names: [], values: [] };
  }

  // Get unique factor names
  const factorNames = [...new Set(factors.map((f) => f.factor_name))].sort();

  if (factorNames.length === 0) {
    return { factor_names: [], values: [] };
  }

  // Group factor values by factor name
  const factorValues: Map<string, Map<string, number>> = new Map();
  for (const f of factors) {
    if (!factorValues.has(f.factor_name)) {
      factorValues.set(f.factor_name, new Map());
    }
    // Use date+instrument as key for pairing
    const key = `${f.date}|${f.instrument}`;
    factorValues.get(f.factor_name)!.set(key, f.value);
  }

  // Compute Pearson correlation for each pair
  const n = factorNames.length;
  const values: number[][] = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        values[i][j] = 1.0;
      } else if (j < i) {
        values[i][j] = values[j][i]; // Symmetric
      } else {
        const name1 = factorNames[i];
        const name2 = factorNames[j];
        const vals1 = factorValues.get(name1)!;
        const vals2 = factorValues.get(name2)!;

        // Find common keys (same date+instrument)
        const commonKeys: string[] = [];
        for (const key of vals1.keys()) {
          if (vals2.has(key)) {
            commonKeys.push(key);
          }
        }

        if (commonKeys.length < 2) {
          values[i][j] = 0; // Not enough data points
        } else {
          const x = commonKeys.map((k) => vals1.get(k)!);
          const y = commonKeys.map((k) => vals2.get(k)!);
          values[i][j] = pearsonCorrelation(x, y);
        }
      }
    }
  }

  return { factor_names: factorNames, values };
}

/**
 * Compute Pearson correlation coefficient
 */
function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n === 0) return 0;

  // Compute means
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
  }
  const meanX = sumX / n;
  const meanY = sumY / n;

  // Compute covariance and standard deviations
  let cov = 0;
  let varX = 0;
  let varY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    cov += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }

  const stdX = Math.sqrt(varX);
  const stdY = Math.sqrt(varY);

  if (stdX === 0 || stdY === 0) return 0;

  return cov / (stdX * stdY);
}

/**
 * Compute each factor's return contribution.
 * Simple approach: for each factor, compute average value weighted by returns.
 */
export function computeFactorReturns(factors: FactorValue[], returns: number[]): FactorReturn[] {
  if (factors.length === 0 || returns.length === 0) {
    return [];
  }

  // Group factors by name
  const factorGroups: Map<string, FactorValue[]> = new Map();
  for (const f of factors) {
    if (!factorGroups.has(f.factor_name)) {
      factorGroups.set(f.factor_name, []);
    }
    factorGroups.get(f.factor_name)!.push(f);
  }

  // Compute return contribution per factor
  // Match by date+instrument order in the original factor list
  const results: FactorReturn[] = [];

  for (const [factorName, factorVals] of factorGroups) {
    let totalContribution = 0;
    let count = 0;

    for (let i = 0; i < factorVals.length && i < returns.length; i++) {
      totalContribution += factorVals[i].value * returns[i];
      count++;
    }

    const returnContribution = count > 0 ? totalContribution / count : 0;
    results.push({ factor_name: factorName, return_contribution: returnContribution });
  }

  return results;
}