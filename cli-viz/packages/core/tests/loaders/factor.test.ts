import { describe, expect, test } from 'bun:test';
import type { FactorValue } from '../../src/types.js';
import { computeCorrelationMatrix, computeFactorReturns } from '../../src/loaders/factor.js';

describe('computeCorrelationMatrix', () => {
  test('with perfect correlation', () => {
    const factors: FactorValue[] = [
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_a', value: 1.0 },
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_b', value: 2.0 },
      { date: '2024-01-01', instrument: 'TSLA', factor_name: 'alpha_a', value: 3.0 },
      { date: '2024-01-01', instrument: 'TSLA', factor_name: 'alpha_b', value: 6.0 },
    ];

    const result = computeCorrelationMatrix(factors);

    expect(result.factor_names).toContain('alpha_a');
    expect(result.factor_names).toContain('alpha_b');
    expect(result.values[0][0]).toBe(1.0);
    expect(result.values[1][1]).toBe(1.0);
    expect(Math.abs(result.values[0][1] - 1.0)).toBeLessThan(0.0001);
    expect(result.values[0][1]).toBe(result.values[1][0]);
  });

  test('with anti-correlation', () => {
    const factors: FactorValue[] = [
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_a', value: 1.0 },
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_b', value: -1.0 },
      { date: '2024-01-01', instrument: 'TSLA', factor_name: 'alpha_a', value: 2.0 },
      { date: '2024-01-01', instrument: 'TSLA', factor_name: 'alpha_b', value: -2.0 },
    ];

    const result = computeCorrelationMatrix(factors);

    expect(Math.abs(result.values[0][1] + 1.0)).toBeLessThan(0.0001);
  });

  test('diagonal is 1.0', () => {
    const factors: FactorValue[] = [
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_a', value: 1.0 },
      { date: '2024-01-01', instrument: 'TSLA', factor_name: 'alpha_a', value: 2.0 },
    ];

    const result = computeCorrelationMatrix(factors);

    for (let i = 0; i < result.factor_names.length; i++) {
      expect(result.values[i][i]).toBe(1.0);
    }
  });

  test('is symmetric', () => {
    const factors: FactorValue[] = [
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_a', value: 1.0 },
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_b', value: 0.5 },
      { date: '2024-01-01', instrument: 'TSLA', factor_name: 'alpha_a', value: 2.0 },
      { date: '2024-01-01', instrument: 'TSLA', factor_name: 'alpha_b', value: 0.8 },
    ];

    const result = computeCorrelationMatrix(factors);

    for (let i = 0; i < result.values.length; i++) {
      for (let j = 0; j < result.values[i].length; j++) {
        expect(result.values[i][j]).toBe(result.values[j][i]);
      }
    }
  });

  test('empty data handling', () => {
    const result = computeCorrelationMatrix([]);
    expect(result.factor_names).toEqual([]);
    expect(result.values).toEqual([]);
  });
});

describe('computeFactorReturns', () => {
  test('with known data', () => {
    const factors: FactorValue[] = [
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_momentum', value: 0.5 },
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_momentum', value: 0.6 },
      { date: '2024-01-01', instrument: 'AAPL', factor_name: 'alpha_momentum', value: 0.7 },
    ];
    const returns = [0.01, 0.02, 0.03];

    const result = computeFactorReturns(factors, returns);

    expect(result.length).toBe(1);
    expect(result[0].factor_name).toBe('alpha_momentum');
    const expected = (0.5 * 0.01 + 0.6 * 0.02 + 0.7 * 0.03) / 3;
    expect(result[0].return_contribution).toBeCloseTo(expected, 5);
  });

  test('empty data handling', () => {
    expect(computeFactorReturns([], [])).toEqual([]);
    expect(computeFactorReturns([], [0.01])).toEqual([]);
  });
});