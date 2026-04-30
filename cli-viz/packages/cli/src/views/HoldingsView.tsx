import React from 'react';
import { Box, Text } from 'ink';
import type { PortfolioPosition, WatchlistSummary } from '@viz/core';

export interface HoldingsViewProps {
  positions: PortfolioPosition[];
  summary: WatchlistSummary | null;
  allocation: { stocks: number; cash: number };
}

function formatWeight(w: number): string {
  return `${(w * 100).toFixed(1)}%`;
}

function formatAllocation(a: number): string {
  return a.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

function signalColor(signal: string, isCash: boolean): string | undefined {
  if (isCash) return 'gray';
  switch (signal) {
    case 'Strong Buy': return 'green';
    case 'Buy': return 'green';
    case 'Sell': return 'yellow';
    case 'Strong Sell': return 'red';
    default: return undefined;
  }
}

function padRight(s: string, len: number): string {
  const stripped = s.replace(/\x1b\[[0-9;]*m/g, '');
  const pad = Math.max(0, len - stripped.length);
  return s + ' '.repeat(pad);
}

function padLeft(s: string, len: number): string {
  const stripped = s.replace(/\x1b\[[0-9;]*m/g, '');
  const pad = Math.max(0, len - stripped.length);
  return ' '.repeat(pad) + s;
}

const BAR_WIDTH = 30;

function WeightBar({ weight, isCash }: { weight: number; isCash: boolean }) {
  const filled = Math.round(weight * BAR_WIDTH);
  const empty = BAR_WIDTH - filled;
  const bar = '█'.repeat(filled) + '░'.repeat(empty);
  const color = isCash ? 'gray' : 'green';
  return (
    <Text>
      <Text color={color}>{bar}</Text> {formatWeight(weight)}
    </Text>
  );
}

function SummaryHeader({ summary }: { summary: WatchlistSummary }) {
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="cyan">Portfolio Summary</Text>
      <Box gap={2}>
        <Text>Total Stocks: <Text bold>{summary.total_stocks}</Text></Text>
        <Text>Bullish: <Text color="green" bold>{summary.bullish_count}</Text></Text>
        <Text>Bearish: <Text color="red" bold>{summary.bearish_count}</Text></Text>
      </Box>
      <Box gap={2}>
        <Text>Avg Score: <Text bold>{summary.avg_score.toFixed(2)}</Text></Text>
        {summary.best_stock && <Text>Best: <Text color="green">{summary.best_stock}</Text></Text>}
        {summary.worst_stock && <Text>Worst: <Text color="red">{summary.worst_stock}</Text></Text>}
      </Box>
    </Box>
  );
}

function HoldingsTable({ positions }: { positions: PortfolioPosition[] }) {
  const sorted = [...positions].sort((a, b) => {
    if (a.instrument === 'CASH') return 1;
    if (b.instrument === 'CASH') return -1;
    return b.weight - a.weight;
  });

  const colInstrument = 10;
  const colName = 20;
  const colWeight = 9;
  const colAlloc = 12;
  const colSignal = 13;
  const colPrice = 12;

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="cyan">Holdings</Text>
      <Box>
        <Text bold>{padRight('Instrument', colInstrument)}{padRight('Name', colName)}{padRight('Weight', colWeight)}{padRight('Allocation', colAlloc)}{padRight('Signal', colSignal)}{'Entry Price'}</Text>
      </Box>
      <Text color="gray">{'─'.repeat(76)}</Text>
      {sorted.map((pos) => {
        const isCash = pos.instrument === 'CASH';
        const sigCol = signalColor(pos.signal, isCash);
        const signalText = sigCol ? <Text color={sigCol}>{pos.signal}</Text> : <Text>{pos.signal}</Text>;

        return (
          <Box key={pos.instrument}>
            <Text>
              {padRight(isCash ? 'CASH' : pos.instrument, colInstrument)}
              {padRight(pos.name.slice(0, colName - 1), colName)}
              {padRight(formatWeight(pos.weight), colWeight)}
              {padRight(formatAllocation(pos.allocation), colAlloc)}
            </Text>
            <Text>{padRight('', 0)}</Text>
            {signalText}
            <Text>{' '.repeat(Math.max(0, colSignal - pos.signal.length - 1))}{isCash ? '' : formatAllocation(pos.entry_price)}</Text>
          </Box>
        );
      })}
    </Box>
  );
}

function WeightDistribution({ positions }: { positions: PortfolioPosition[] }) {
  const sorted = [...positions].sort((a, b) => {
    if (a.instrument === 'CASH') return 1;
    if (b.instrument === 'CASH') return -1;
    return b.weight - a.weight;
  });

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="cyan">Weight Distribution</Text>
      {sorted.map((pos) => {
        const isCash = pos.instrument === 'CASH';
        const label = isCash ? 'CASH  ' : pos.instrument;
        return (
          <Box key={pos.instrument}>
            <Text>{padLeft(label, 8)} </Text>
            <WeightBar weight={pos.weight} isCash={isCash} />
          </Box>
        );
      })}
    </Box>
  );
}

function AllocationSummary({ allocation }: { allocation: { stocks: number; cash: number } }) {
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="cyan">Allocation</Text>
      <Text>Stocks: <Text color="green" bold>{formatWeight(allocation.stocks)}</Text></Text>
      <Text>Cash:   <Text color="gray" bold>{formatWeight(allocation.cash)}</Text></Text>
    </Box>
  );
}

export function HoldingsView({ positions, summary, allocation }: HoldingsViewProps) {
  if (positions.length === 0) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color="yellow">No holdings data available.</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      {summary && <SummaryHeader summary={summary} />}
      <HoldingsTable positions={positions} />
      <WeightDistribution positions={positions} />
      <AllocationSummary allocation={allocation} />
    </Box>
  );
}
