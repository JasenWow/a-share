import React from 'react';
import { render, Box, Text } from 'ink';
import { existsSync } from 'fs';
import { resolve, join } from 'path';
import {
  loadBacktestReport,
  loadBacktestAnalysis,
  computeCumulativeReturns,
  computeDrawdownSeries,
} from '@viz/core';
import { BacktestView } from '../views/BacktestView.js';

interface BacktestOptions {
  dataDir: string;
  file?: string;
}

export function runBacktestCommand(options: BacktestOptions) {
  const dataDir = resolve(options.dataDir);

  if (!existsSync(dataDir)) {
    console.error(`Error: Data directory not found: ${dataDir}`);
    process.exit(1);
  }

  const reportPath = options.file ? resolve(options.file) : join(dataDir, 'backtest_report.parquet');

  if (!existsSync(reportPath)) {
    console.error(`Error: Backtest report not found: ${reportPath}`);
    process.exit(1);
  }

  let report;
  try {
    report = loadBacktestReport(reportPath);
  } catch (err) {
    console.error(`Error loading backtest report: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }

  let analysis = null;
  const analysisPath = join(dataDir, 'backtest_analysis.parquet');
  if (existsSync(analysisPath)) {
    try {
      analysis = loadBacktestAnalysis(analysisPath);
    } catch {
      // Analysis is optional, continue without it
    }
  }

  const cumulativeReturns = computeCumulativeReturns(report);
  const drawdownSeries = computeDrawdownSeries(report);

  const { unmount } = render(
    <BacktestView
      report={report}
      analysis={analysis}
      cumulativeReturns={cumulativeReturns}
      drawdownSeries={drawdownSeries}
    />,
  );

  process.stdout.write('\x1b[?25h');
  unmount();
}
