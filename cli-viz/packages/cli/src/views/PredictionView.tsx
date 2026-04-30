import React from 'react';
import { Box, Text } from 'ink';
import type { KronosPrediction, LightGBMPrediction, MergedScore } from '@viz/core';

interface PredictionViewProps {
  kronos: KronosPrediction[];
  lightgbm: LightGBMPrediction[];
  merged: MergedScore[];
}

const MAX_ROWS = 20;

function scoreColor(value: number): string {
  return value >= 0 ? 'green' : 'red';
}

function signalArrow(signal: string): { arrow: string; color: string } {
  switch (signal) {
    case 'Strong Buy':
      return { arrow: '\u25B2\u25B2', color: 'green' };
    case 'Buy':
      return { arrow: '\u25B2', color: 'green' };
    case 'Sell':
      return { arrow: '\u25BC', color: 'yellow' };
    case 'Strong Sell':
      return { arrow: '\u25BC\u25BC', color: 'red' };
    default:
      return { arrow: '?', color: 'white' };
  }
}

function KronosTable({ data }: { data: KronosPrediction[] }) {
  const ranked = [...data].sort((a, b) => b.score_pct - a.score_pct).slice(0, MAX_ROWS);

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color="cyan" bold>
        Kronos Predictions
      </Text>
      <Box>
        <Text dimColor>
          {'Rank'.padEnd(6)}{'Instrument'.padEnd(14)}{'Score'.padStart(10)}{'Score%'.padStart(10)}  Signal
        </Text>
      </Box>
      {ranked.map((row, i) => (
        <Box key={`${row.instrument}-${i}`}>
          <Text>{String(i + 1).padEnd(6)}</Text>
          <Text>{row.instrument.padEnd(14)}</Text>
          <Text color={scoreColor(row.score)}>{row.score.toFixed(4).padStart(10)}</Text>
          <Text color={scoreColor(row.score_pct)}>  {row.score_pct.toFixed(2).padStart(8)}%</Text>
          <Text color={scoreColor(row.score_pct)}>  {row.score_pct >= 0 ? 'Bullish' : 'Bearish'}</Text>
        </Box>
      ))}
    </Box>
  );
}

function LightGBMTable({ data }: { data: LightGBMPrediction[] }) {
  const ranked = [...data].sort((a, b) => b.score - a.score).slice(0, MAX_ROWS);

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color="cyan" bold>
        LightGBM Predictions
      </Text>
      <Box>
        <Text dimColor>
          {'Rank'.padEnd(6)}{'Instrument'.padEnd(14)}{'Score'.padStart(10)}
        </Text>
      </Box>
      {ranked.map((row, i) => (
        <Box key={`${row.instrument}-${i}`}>
          <Text>{String(i + 1).padEnd(6)}</Text>
          <Text>{row.instrument.padEnd(14)}</Text>
          <Text color={scoreColor(row.score)}>{row.score.toFixed(4).padStart(10)}</Text>
        </Box>
      ))}
    </Box>
  );
}

function MergedTable({ data }: { data: MergedScore[] }) {
  const ranked = [...data].sort((a, b) => b.combined_score - a.combined_score).slice(0, MAX_ROWS);

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color="cyan" bold>
        Combined Model Rankings
      </Text>
      <Box>
        <Text dimColor>
          {'Rank'.padEnd(6)}{'Instrument'.padEnd(14)}{'Kronos'.padStart(10)}{'LightGBM'.padStart(10)}{'Combined'.padStart(10)}  Signal
        </Text>
      </Box>
      {ranked.map((row, i) => {
        const { arrow, color } = signalArrow(row.signal);
        return (
          <Box key={row.instrument}>
            <Text>{String(i + 1).padEnd(6)}</Text>
            <Text>{row.instrument.padEnd(14)}</Text>
            <Text color={scoreColor(row.kronos_score)}>{row.kronos_score.toFixed(4).padStart(10)}</Text>
            <Text color={scoreColor(row.lightgbm_score)}> {row.lightgbm_score.toFixed(4).padStart(9)}</Text>
            <Text color={scoreColor(row.combined_score)}> {row.combined_score.toFixed(4).padStart(9)}</Text>
            <Text color={color}>  {arrow} {row.signal}</Text>
          </Box>
        );
      })}
    </Box>
  );
}

export function PredictionView({ kronos, lightgbm, merged }: PredictionViewProps) {
  const hasKronos = kronos.length > 0;
  const hasLightgbm = lightgbm.length > 0;
  const hasMerged = merged.length > 0;

  if (!hasKronos && !hasLightgbm && !hasMerged) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color="red">No prediction data available</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      {hasKronos ? <KronosTable data={kronos} /> : <Text color="yellow">Kronos: No data available</Text>}
      {hasLightgbm ? <LightGBMTable data={lightgbm} /> : <Text color="yellow">LightGBM: No data available</Text>}
      {merged.length > 0 && <MergedTable data={merged} />}
    </Box>
  );
}
