import { open } from '@evan/duckdb';
import { writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const dataDir = join(__dirname, 'data');

// Create data directory
mkdirSync(dataDir, { recursive: true });

// Use DuckDB to create parquet files
const db = open(':memory:');
const conn = db.connect();

// 1. Backtest Report (10 rows)
conn.query(`
  CREATE TABLE backtest_report AS SELECT * FROM (
    VALUES
      ('2024-01-02', 0.012, 0.008, 0.001, 0.15),
      ('2024-01-03', -0.005, 0.002, 0.001, 0.12),
      ('2024-01-04', 0.020, 0.015, 0.001, 0.18),
      ('2024-01-05', -0.010, -0.003, 0.001, 0.10),
      ('2024-01-08', 0.008, 0.005, 0.001, 0.14),
      ('2024-01-09', 0.015, 0.010, 0.001, 0.16),
      ('2024-01-10', -0.003, 0.001, 0.001, 0.11),
      ('2024-01-11', 0.025, 0.018, 0.001, 0.20),
      ('2024-01-12', 0.010, 0.007, 0.001, 0.13),
      ('2024-01-15', 0.005, 0.003, 0.001, 0.09)
  ) AS t(date, "return", bench, cost, turnover)
`);
conn.query(`COPY backtest_report TO '${join(dataDir, 'backtest_report.parquet')}' (FORMAT PARQUET)`);

// 2. Backtest Analysis
conn.query(`
  CREATE TABLE backtest_analysis AS SELECT * FROM (
    VALUES
      ('annualized_return', 'mean', 0.15),
      ('sharpe_ratio', 'mean', 1.5),
      ('max_drawdown', 'mean', -0.08),
      ('information_ratio', 'mean', 0.9),
      ('total_cost', 'mean', 0.01),
      ('mean_turnover', 'mean', 0.13),
      ('n_trading_days', 'mean', 10.0)
  ) AS t(category, metric_name, value)
`);
conn.query(`COPY backtest_analysis TO '${join(dataDir, 'backtest_analysis.parquet')}' (FORMAT PARQUET)`);

// 3. Kronos Signals (CSV)
const kronosCSV = `signal_date,instrument,score,score_pct
2024-01-15,600519,0.85,85.0
2024-01-15,000858,0.72,72.0
2024-01-15,601318,0.45,45.0
2024-01-15,000333,-0.30,-30.0
2024-01-15,600036,-0.65,-65.0`;
writeFileSync(join(dataDir, 'kronos_signals.csv'), kronosCSV);

// 4. LightGBM Predictions (parquet)
conn.query(`
  CREATE TABLE lightgbm_preds AS SELECT * FROM (
    VALUES
      ('2024-01-15', '600519', 0.78),
      ('2024-01-15', '000858', 0.65),
      ('2024-01-15', '601318', 0.40),
      ('2024-01-15', '000333', -0.25),
      ('2024-01-15', '600036', -0.55)
  ) AS t(date, instrument, score)
`);
conn.query(`COPY lightgbm_preds TO '${join(dataDir, 'predictions.parquet')}' (FORMAT PARQUET)`);

// 5. Factor data (parquet)
conn.query(`
  CREATE TABLE factor_data AS SELECT * FROM (
    VALUES
      ('2024-01-15', '600519', 0.5, 0.3, -0.2),
      ('2024-01-15', '000858', 0.4, 0.1, -0.1),
      ('2024-01-15', '601318', -0.2, 0.6, 0.3),
      ('2024-01-15', '000333', -0.3, -0.4, 0.5),
      ('2024-01-15', '600036', 0.1, -0.2, -0.3)
  ) AS t(date, instrument, momentum, value, volatility)
`);
conn.query(`COPY factor_data TO '${join(dataDir, 'factors.parquet')}' (FORMAT PARQUET)`);

conn.close();
db.close();
console.log('✅ All fixtures generated in', dataDir);