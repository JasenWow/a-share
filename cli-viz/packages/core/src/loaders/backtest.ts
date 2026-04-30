import { open } from '@evan/duckdb';
import { readFileSync } from 'fs';
import type { BacktestReport, BacktestReportRow, BacktestAnalysis } from '../types.js';

function getParquetColumns(filePath: string, conn: { query: (sql: string) => unknown[] }): string[] {
  try {
    const schema = conn.query(`SELECT name FROM parquet_schema('${filePath}')`) as Record<string, unknown>[];
    return schema
      .map((row: Record<string, unknown>) => String(row.name))
      .filter((name) => name !== 'duckdb_schema');
  } catch {
    return [];
  }
}

export function loadBacktestReport(filePath: string): BacktestReport {
  try {
    readFileSync(filePath);
  } catch {
    throw new Error(`Backtest report file not found: ${filePath}`);
  }

  const db = open(':memory:');
  const conn = db.connect();

  try {
    const columns = getParquetColumns(filePath, conn);
    if (columns.length === 0) {
      return { rows: [] };
    }

    const numericColumns = ['return', 'bench', 'cost', 'turnover'];
    const selectClause = columns
      .map((col) => {
        if (numericColumns.includes(col)) {
          return `CAST(CAST(${col} AS VARCHAR) AS DOUBLE) AS ${col}`;
        }
        return `CAST(${col} AS VARCHAR) AS ${col}`;
      })
      .join(', ');
    const query = `SELECT ${selectClause} FROM '${filePath}'`;
    const rows = conn.query(query);

    if (!rows || rows.length === 0) {
      return { rows: [] };
    }

    const result: BacktestReportRow[] = [];

    for (const row of rows as Record<string, unknown>[]) {
      result.push({
        date: String(row.date ?? row.DatetimeIndex ?? ''),
        return: Number(row['return'] ?? row.return_val ?? 0),
        bench: Number(row.bench ?? 0),
        cost: Number(row.cost ?? 0),
        turnover: Number(row.turnover ?? 0),
      });
    }

    return { rows: result };
  } catch {
    throw new Error(`Invalid parquet file: ${filePath}`);
  } finally {
    conn.close();
    db.close();
  }
}

export function loadBacktestAnalysis(filePath: string): BacktestAnalysis {
  try {
    readFileSync(filePath);
  } catch {
    throw new Error(`Backtest report file not found: ${filePath}`);
  }

  const db = open(':memory:');
  const conn = db.connect();

  try {
    const columns = getParquetColumns(filePath, conn);
    if (columns.length === 0) {
      return createEmptyAnalysis();
    }

    const selectClause = columns
      .map((col) => {
        const cast = `CAST(${col} AS VARCHAR)`;
        return col === 'value' || col === '0' || col.endsWith('_id') || col.endsWith('_type')
          ? `${cast} AS ${col}`
          : `CAST(${cast} AS DOUBLE) AS ${col}`;
      })
      .join(', ');
    const query = `SELECT ${selectClause} FROM '${filePath}'`;
    const rows = conn.query(query);

    if (!rows || rows.length === 0) {
      return createEmptyAnalysis();
    }

    const metrics: Record<string, Record<string, number>> = {};

    for (const row of rows as Record<string, unknown>[]) {
      const category = String(row.category ?? row.index ?? '');
      const metric = String(row.metric ?? row.metric_name ?? '');
      const value = Number(row.value ?? row['0'] ?? 0);

      if (!metrics[category]) {
        metrics[category] = {};
      }
      metrics[category][metric] = value;
    }

    return extractAnalysisMetrics(metrics);
  } catch {
    throw new Error(`Invalid parquet file: ${filePath}`);
  } finally {
    conn.close();
    db.close();
  }
}

export function computeCumulativeReturns(report: BacktestReport): number[] {
  if (report.rows.length === 0) {
    return [];
  }

  const cumulative: number[] = [];
  let running = 1.0;

  for (const row of report.rows) {
    running *= 1 + row.return;
    cumulative.push(running);
  }

  return cumulative;
}

export function computeDrawdownSeries(report: BacktestReport): number[] {
  const cumulative = computeCumulativeReturns(report);
  if (cumulative.length === 0) {
    return [];
  }

  const drawdowns: number[] = [];
  let peak = cumulative[0];

  for (const value of cumulative) {
    if (value > peak) {
      peak = value;
    }
    drawdowns.push((value - peak) / peak);
  }

  return drawdowns;
}

function createEmptyAnalysis(): BacktestAnalysis {
  return {
    annualized_return: 0,
    annualized_benchmark: 0,
    excess_return: 0,
    sharpe_ratio: 0,
    information_ratio: 0,
    max_drawdown: 0,
    drawdown_duration_days: 0,
    total_cost: 0,
    mean_turnover: 0,
    max_turnover: 0,
    monthly_return_distribution: {},
    n_trading_days: 0,
    start_date: '',
    end_date: '',
  };
}

function extractAnalysisMetrics(metrics: Record<string, Record<string, number>>): BacktestAnalysis {
  const getMetric = (category: string, metric: string): number => {
    return metrics[category]?.[metric] ?? 0;
  };

  return {
    annualized_return: getMetric('annualized_return', 'mean') || getMetric('return', 'annualized'),
    annualized_benchmark: getMetric('annualized_benchmark', 'mean') || getMetric('bench', 'annualized'),
    excess_return: getMetric('excess_return_without_cost', 'mean') || getMetric('excess_return', 'mean'),
    sharpe_ratio: getMetric('sharpe_ratio', 'mean') || getMetric('sharpe', 'ratio'),
    information_ratio: getMetric('information_ratio', 'mean') || getMetric('ir', 'mean'),
    max_drawdown: getMetric('max_drawdown', 'mean') || getMetric('drawdown', 'max'),
    drawdown_duration_days: getMetric('drawdown_duration', 'max') || getMetric('drawdown_duration_days', 'mean'),
    total_cost: getMetric('total_cost', 'mean') || getMetric('cost', 'total'),
    mean_turnover: getMetric('mean_turnover', 'mean') || getMetric('turnover', 'mean'),
    max_turnover: getMetric('max_turnover', 'max') || getMetric('turnover', 'max'),
    monthly_return_distribution: {},
    n_trading_days: getMetric('n_trading_days', 'mean') || getMetric('days', 'n'),
    start_date: '',
    end_date: '',
  };
}