import React from 'react';
import { Box, Text } from 'ink';
import type { FactorValue, CorrelationMatrix, FactorReturn } from '@viz/core';

export interface FactorViewProps {
  factors: FactorValue[];
  correlationMatrix: CorrelationMatrix;
  factorReturns: FactorReturn[];
}

/** Pick a color for a correlation value */
function correlationColor(val: number): string | undefined {
  if (val > 0.7) return 'blue';
  if (val > 0.3) return 'cyan';
  if (val > -0.3) return undefined; // will use dimColor
  if (val > -0.7) return 'yellow';
  return 'red';
}

function FactorValuesTable({ factors }: { factors: FactorValue[] }) {
  // Get unique dates and pick the latest
  const dates = [...new Set(factors.map((f) => f.date))].sort();
  const latestDate = dates[dates.length - 1];

  const latestFactors = factors.filter((f) => f.date === latestDate);

  // Unique factor names (preserving first-seen order)
  const allFactorNames = [...new Set(latestFactors.map((f) => f.factor_name))];
  const factorNames = allFactorNames.slice(0, 6);

  // Unique instruments
  const allInstruments = [...new Set(latestFactors.map((f) => f.instrument))];

  // Build a lookup: key "instrument|factor_name" → value
  const lookup = new Map<string, number>();
  for (const f of latestFactors) {
    if (factorNames.includes(f.factor_name)) {
      lookup.set(`${f.instrument}|${f.factor_name}`, f.value);
    }
  }

  // Sort instruments by first factor value descending, take top 15
  const sortedInstruments = allInstruments.sort((a, b) => {
    const va = lookup.get(`${a}|${factorNames[0]}`) ?? 0;
    const vb = lookup.get(`${b}|${factorNames[0]}`) ?? 0;
    return vb - va;
  });
  const instruments = sortedInstruments.slice(0, 15);

  // Column widths
  const instrumentWidth = 12;
  const factorColWidth = 12;

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text bold color="cyan">
        Factor Values (Latest)
      </Text>
      <Text dimColor>
        Date: {latestDate}
      </Text>

      {/* Header row */}
      <Box>
        <Text bold>{'Instrument'.padEnd(instrumentWidth)}</Text>
        {factorNames.map((fn) => (
          <Text key={fn} bold>
            {fn.padEnd(factorColWidth).slice(0, factorColWidth)}
          </Text>
        ))}
      </Box>

      {/* Data rows */}
      {instruments.map((inst) => (
        <Box key={inst}>
          <Text>{inst.padEnd(instrumentWidth).slice(0, instrumentWidth)}</Text>
          {factorNames.map((fn) => {
            const val = lookup.get(`${inst}|${fn}`);
            if (val === undefined) {
              return (
                <Text key={fn}>{'—'.padEnd(factorColWidth)}</Text>
              );
            }
            const formatted = val.toFixed(4).padStart(factorColWidth);
            return (
              <Text key={fn} color={val >= 0 ? 'green' : 'red'}>
                {formatted}
              </Text>
            );
          })}
        </Box>
      ))}
    </Box>
  );
}

function CorrelationMatrixView({ matrix }: { matrix: CorrelationMatrix }) {
  const { factor_names, values } = matrix;

  if (!factor_names.length || !values.length) {
    return (
      <Box marginTop={1}>
        <Text color="yellow">Correlation data not available</Text>
      </Box>
    );
  }

  const labelWidth = 12;
  const cellWidth = 8;

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text bold color="cyan">
        Factor Correlation Matrix
      </Text>

      {/* Header row with factor names */}
      <Box>
        <Text>{''.padEnd(labelWidth)}</Text>
        {factor_names.map((fn) => (
          <Text key={fn} bold>
            {fn.padEnd(cellWidth).slice(0, cellWidth)}
          </Text>
        ))}
      </Box>

      {/* Data rows */}
      {factor_names.map((rowName, i) => (
        <Box key={rowName}>
          <Text bold>{rowName.padEnd(labelWidth).slice(0, labelWidth)}</Text>
          {factor_names.map((_colName, j) => {
            const val = values[i]?.[j] ?? 0;
            const display = val.toFixed(2).padStart(6);
            const color = correlationColor(val);
            const isDiag = i === j;
            const isMid = val > -0.3 && val <= 0.3;

            if (isMid && !isDiag) {
              return (
                <Text key={j} dimColor>
                  {display.padEnd(cellWidth)}
                </Text>
              );
            }
            return (
              <Text key={j} color={color}>
                {display.padEnd(cellWidth)}
              </Text>
            );
          })}
        </Box>
      ))}
    </Box>
  );
}

function FactorReturnBars({ factorReturns }: { factorReturns: FactorReturn[] }) {
  if (factorReturns.length === 0) return null;

  const maxBarWidth = 40;

  // Sort by absolute return contribution descending
  const sorted = [...factorReturns].sort(
    (a, b) => Math.abs(b.return_contribution) - Math.abs(a.return_contribution),
  );

  // Find the max absolute value for scaling
  const maxAbs = Math.max(...sorted.map((f) => Math.abs(f.return_contribution)), 0.0001);

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text bold color="cyan">
        Factor Return Contribution
      </Text>

      {sorted.map((fr) => {
        const scaled = Math.round((Math.abs(fr.return_contribution) / maxAbs) * maxBarWidth);
        const barChars = Math.max(scaled, 1);
        const isPositive = fr.return_contribution >= 0;
        const bar = isPositive ? '█'.repeat(barChars) : '░'.repeat(barChars);
        const nameCol = fr.factor_name.padEnd(14).slice(0, 14);

        return (
          <Box key={fr.factor_name}>
            <Text>{nameCol}</Text>
            <Text color={isPositive ? 'green' : 'red'}>{bar}</Text>
            <Text> {fr.return_contribution.toFixed(4)}</Text>
          </Box>
        );
      })}
    </Box>
  );
}

export function FactorView({ factors, correlationMatrix, factorReturns }: FactorViewProps) {
  if (factors.length === 0) {
    return (
      <Box padding={1}>
        <Text color="yellow">No factor data available</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      <FactorValuesTable factors={factors} />
      <CorrelationMatrixView matrix={correlationMatrix} />
      <FactorReturnBars factorReturns={factorReturns} />
    </Box>
  );
}
