import React from 'react';
import { Box, Text } from 'ink';
import asciichart from 'asciichart';
import type { BacktestReport, BacktestAnalysis } from '@viz/core';

interface BacktestViewProps {
  report: BacktestReport;
  analysis: BacktestAnalysis | null;
  cumulativeReturns: number[];
  drawdownSeries: number[];
}

function formatPercent(value: number): string {
  return (value * 100).toFixed(2) + '%';
}

function colorForSharpe(value: number): 'green' | 'yellow' | 'red' {
  if (value > 1) return 'green';
  if (value >= 0) return 'yellow';
  return 'red';
}

function MetricsTable({ analysis }: { analysis: BacktestAnalysis }) {
  const rows: Array<{ label: string; value: string; color: string }> = [
    { label: 'Annualized Return', value: formatPercent(analysis.annualized_return), color: analysis.annualized_return >= 0 ? 'green' : 'red' },
    { label: 'Sharpe Ratio', value: analysis.sharpe_ratio.toFixed(2), color: colorForSharpe(analysis.sharpe_ratio) },
    { label: 'Max Drawdown', value: formatPercent(analysis.max_drawdown), color: 'red' },
    { label: 'Information Ratio', value: analysis.information_ratio.toFixed(2), color: colorForSharpe(analysis.information_ratio) },
    { label: 'Total Cost', value: analysis.total_cost.toFixed(4), color: 'yellow' },
    { label: 'Mean Turnover', value: formatPercent(analysis.mean_turnover), color: 'cyan' },
    { label: 'Trading Period', value: analysis.start_date && analysis.end_date ? `${analysis.start_date} → ${analysis.end_date}` : 'N/A', color: 'white' },
    { label: 'Trading Days', value: String(analysis.n_trading_days), color: 'white' },
  ];

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="cyan">Performance Metrics</Text>
      {rows.map((row) => (
        <Box key={row.label}>
          <Box width={22}>
            <Text dimColor>{row.label}</Text>
          </Box>
          <Text color={row.color as 'green' | 'red' | 'yellow' | 'cyan' | 'white'}>{row.value}</Text>
        </Box>
      ))}
    </Box>
  );
}

function CumulativeReturnsChart({ report, cumulativeReturns }: { report: BacktestReport; cumulativeReturns: number[] }) {
  if (cumulativeReturns.length === 0) return null;

  const benchCumulative: number[] = [];
  let running = 1.0;
  for (const row of report.rows) {
    running *= 1 + row.bench;
    benchCumulative.push(running);
  }

  const chart = asciichart.plot([cumulativeReturns, benchCumulative], {
    height: 15,
    colors: [asciichart.green, asciichart.blue],
  });

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="cyan">Cumulative Returns</Text>
      <Text dimColor>Green = Strategy | Blue = Benchmark</Text>
      <Text>{chart}</Text>
    </Box>
  );
}

function DrawdownChart({ drawdownSeries }: { drawdownSeries: number[] }) {
  if (drawdownSeries.length === 0) return null;

  const chart = asciichart.plot(drawdownSeries, {
    height: 10,
    colors: [asciichart.red],
  });

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="red">Drawdown</Text>
      <Text>{chart}</Text>
    </Box>
  );
}

function MonthlyReturnsChart({ monthlyReturnDistribution }: { monthlyReturnDistribution: Record<string, number> }) {
  const months = Object.keys(monthlyReturnDistribution);
  if (months.length === 0) return null;

  const sortedMonths = months.sort();
  const values = sortedMonths.map((m) => monthlyReturnDistribution[m]);
  const maxAbsValue = Math.max(...values.map(Math.abs), 0.001);
  const maxBarLength = 40;

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="cyan">Monthly Returns</Text>
      {sortedMonths.map((month) => {
        const value = monthlyReturnDistribution[month];
        const barLength = Math.round((Math.abs(value) / maxAbsValue) * maxBarLength);
        const bar = '█'.repeat(Math.max(barLength, 1));
        const color = value >= 0 ? 'green' : 'red';
        const sign = value >= 0 ? '+' : '';
        return (
          <Box key={month}>
            <Box width={8}>
              <Text>{month}</Text>
            </Box>
            <Text color={color}>{bar}</Text>
            <Text> {sign}{formatPercent(value)}</Text>
          </Box>
        );
      })}
    </Box>
  );
}

export function BacktestView({ report, analysis, cumulativeReturns, drawdownSeries }: BacktestViewProps) {
  if (report.rows.length === 0) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text bold color="yellow">No backtest data available</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      {analysis && <MetricsTable analysis={analysis} />}
      <CumulativeReturnsChart report={report} cumulativeReturns={cumulativeReturns} />
      <DrawdownChart drawdownSeries={drawdownSeries} />
      {analysis && <MonthlyReturnsChart monthlyReturnDistribution={analysis.monthly_return_distribution} />}
    </Box>
  );
}
