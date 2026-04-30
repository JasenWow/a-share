import { describe, expect, test } from 'bun:test';
import React from 'react';
import { render } from 'ink-testing-library';
import { FactorView } from '../../src/views/FactorView.js';
import type { FactorValue, CorrelationMatrix, FactorReturn } from '@viz/core';

const sampleFactors: FactorValue[] = [
  { date: '2025-01-10', instrument: 'SH600000', factor_name: 'momentum', value: 0.1234 },
  { date: '2025-01-10', instrument: 'SH600000', factor_name: 'value', value: -0.0567 },
  { date: '2025-01-10', instrument: 'SH600000', factor_name: 'volatility', value: 0.0091 },
  { date: '2025-01-10', instrument: 'SZ000001', factor_name: 'momentum', value: 0.0456 },
  { date: '2025-01-10', instrument: 'SZ000001', factor_name: 'value', value: 0.0789 },
  { date: '2025-01-10', instrument: 'SZ000001', factor_name: 'volatility', value: -0.0234 },
  { date: '2025-01-10', instrument: 'SH601318', factor_name: 'momentum', value: -0.0112 },
  { date: '2025-01-10', instrument: 'SH601318', factor_name: 'value', value: 0.0345 },
  { date: '2025-01-10', instrument: 'SH601318', factor_name: 'volatility', value: 0.0678 },
];

const sampleCorrelation: CorrelationMatrix = {
  factor_names: ['momentum', 'value', 'volatility'],
  values: [
    [1.0, 0.85, -0.12],
    [0.85, 1.0, 0.45],
    [-0.12, 0.45, 1.0],
  ],
};

const sampleReturns: FactorReturn[] = [
  { factor_name: 'momentum', return_contribution: 0.0234 },
  { factor_name: 'value', return_contribution: 0.0112 },
  { factor_name: 'volatility', return_contribution: -0.0089 },
];

describe('FactorView', () => {
  test('renders factor table with factor names and instrument codes', () => {
    const { lastFrame } = render(
      <FactorView
        factors={sampleFactors}
        correlationMatrix={sampleCorrelation}
        factorReturns={sampleReturns}
      />,
    );
    const output = lastFrame()!;

    expect(output).toContain('Factor Values (Latest)');
    expect(output).toContain('momentum');
    expect(output).toContain('value');
    expect(output).toContain('volatility');
    expect(output).toContain('SH600000');
    expect(output).toContain('SZ000001');
    expect(output).toContain('SH601318');
  });

  test('renders correlation matrix with 1.00 on diagonal', () => {
    const { lastFrame } = render(
      <FactorView
        factors={sampleFactors}
        correlationMatrix={sampleCorrelation}
        factorReturns={sampleReturns}
      />,
    );
    const output = lastFrame()!;

    expect(output).toContain('Factor Correlation Matrix');
    expect(output).toContain('1.00');
    expect(output).toContain('momentum');
    expect(output).toContain('value');
    expect(output).toContain('volatility');
  });

  test('renders return contribution bars with bar characters', () => {
    const { lastFrame } = render(
      <FactorView
        factors={sampleFactors}
        correlationMatrix={sampleCorrelation}
        factorReturns={sampleReturns}
      />,
    );
    const output = lastFrame()!;

    expect(output).toContain('Factor Return Contribution');
    expect(output).toContain('█');
    expect(output).toContain('░');
    expect(output).toContain('0.0234');
    expect(output).toContain('0.0112');
    expect(output).toContain('-0.0089');
  });

  test('renders empty state when factors array is empty', () => {
    const { lastFrame } = render(
      <FactorView
        factors={[]}
        correlationMatrix={{ factor_names: [], values: [] }}
        factorReturns={[]}
      />,
    );
    const output = lastFrame()!;

    expect(output).toContain('No factor data available');
  });

  test('correlation color coding renders all value ranges', () => {
    const colorCorrelation: CorrelationMatrix = {
      factor_names: ['a', 'b', 'c', 'd', 'e'],
      values: [
        [1.0, 0.85, 0.15, -0.5, -0.85],
        [0.85, 1.0, 0.15, -0.5, -0.85],
        [0.15, 0.15, 1.0, -0.5, -0.85],
        [-0.5, -0.5, -0.5, 1.0, -0.85],
        [-0.85, -0.85, -0.85, -0.85, 1.0],
      ],
    };
    const { lastFrame } = render(
      <FactorView
        factors={sampleFactors}
        correlationMatrix={colorCorrelation}
        factorReturns={sampleReturns}
      />,
    );
    const output = lastFrame()!;

    expect(output).toContain('0.85');
    expect(output).toContain('0.15');
    expect(output).toContain('-0.50');
    expect(output).toContain('-0.85');
    expect(output).toContain('1.00');
  });

  test('shows "Correlation data not available" for empty matrix', () => {
    const { lastFrame } = render(
      <FactorView
        factors={sampleFactors}
        correlationMatrix={{ factor_names: [], values: [] }}
        factorReturns={sampleReturns}
      />,
    );
    const output = lastFrame()!;

    expect(output).toContain('Correlation data not available');
  });
});
