import { describe, expect, test } from 'bun:test';
import React from 'react';
import { render } from 'ink-testing-library';
import { PredictionView } from '../../src/views/PredictionView.js';
import type { KronosPrediction, LightGBMPrediction, MergedScore } from '@viz/core';

const sampleKronos: KronosPrediction[] = [
  { signal_date: '2026-04-30', instrument: 'SH600000', score: 0.85, score_pct: 12.5 },
  { signal_date: '2026-04-30', instrument: 'SZ000001', score: -0.3, score_pct: -4.2 },
  { signal_date: '2026-04-30', instrument: 'SH601398', score: 0.12, score_pct: 1.8 },
];

const sampleLightgbm: LightGBMPrediction[] = [
  { date: '2026-04-30', instrument: 'SH600000', score: 0.72 },
  { date: '2026-04-30', instrument: 'SZ000001', score: -0.15 },
  { date: '2026-04-30', instrument: 'SH601398', score: 0.44 },
];

const sampleMerged: MergedScore[] = [
  { instrument: 'SH600000', kronos_score: 0.85, lightgbm_score: 0.72, combined_score: 0.9, signal: 'Strong Buy' },
  { instrument: 'SH601398', kronos_score: 0.12, lightgbm_score: 0.44, combined_score: 0.15, signal: 'Buy' },
  { instrument: 'SZ000001', kronos_score: -0.3, lightgbm_score: -0.15, combined_score: -0.25, signal: 'Sell' },
];

describe('PredictionView', () => {
  test('renders merged ranking table with instruments, scores, and signal arrows', () => {
    const { lastFrame } = render(
      <PredictionView kronos={sampleKronos} lightgbm={sampleLightgbm} merged={sampleMerged} />
    );
    const output = lastFrame()!;

    // Instruments appear
    expect(output).toContain('SH600000');
    expect(output).toContain('SH601398');
    expect(output).toContain('SZ000001');

    // Combined scores appear (4 decimal places)
    expect(output).toContain('0.9000');
    expect(output).toContain('0.1500');
    expect(output).toContain('-0.2500');

    // Signal arrows appear
    expect(output).toContain('\u25B2'); // ▲ for Buy/Strong Buy
    expect(output).toContain('\u25BC'); // ▼ for Sell

    // Signal labels appear
    expect(output).toContain('Strong Buy');
    expect(output).toContain('Buy');
    expect(output).toContain('Sell');

    // Section headers
    expect(output).toContain('Kronos Predictions');
    expect(output).toContain('LightGBM Predictions');
    expect(output).toContain('Combined Model Rankings');
  });

  test('renders only kronos data when lightgbm/merged are empty', () => {
    const { lastFrame } = render(
      <PredictionView kronos={sampleKronos} lightgbm={[]} merged={[]} />
    );
    const output = lastFrame()!;

    // Kronos table present
    expect(output).toContain('Kronos Predictions');
    expect(output).toContain('SH600000');

    // LightGBM empty state
    expect(output).toContain('LightGBM: No data available');
  });

  test('renders empty state when all arrays are empty', () => {
    const { lastFrame } = render(
      <PredictionView kronos={[]} lightgbm={[]} merged={[]} />
    );
    const output = lastFrame()!;

    expect(output).toContain('No prediction data available');
  });

  test('score color coding distinguishes positive and negative scores', () => {
    const { lastFrame } = render(
      <PredictionView kronos={sampleKronos} lightgbm={sampleLightgbm} merged={sampleMerged} />
    );
    const output = lastFrame()!;

    expect(output).toContain('0.8500');
    expect(output).toContain('-0.3000');
    expect(output).toContain('12.50%');
    expect(output).toContain('-4.20%');
    expect(output).toContain('Bullish');
    expect(output).toContain('Bearish');
  });

  test('ranking order is descending by score', () => {
    const unsortedMerged: MergedScore[] = [
      { instrument: 'SZ000001', kronos_score: -0.3, lightgbm_score: -0.15, combined_score: -0.25, signal: 'Sell' },
      { instrument: 'SH600000', kronos_score: 0.85, lightgbm_score: 0.72, combined_score: 0.9, signal: 'Strong Buy' },
      { instrument: 'SH601398', kronos_score: 0.12, lightgbm_score: 0.44, combined_score: 0.15, signal: 'Buy' },
    ];

    const { lastFrame } = render(
      <PredictionView kronos={[]} lightgbm={[]} merged={unsortedMerged} />
    );
    const output = lastFrame()!;

    // SH600000 (highest combined_score 0.9) should come first in output
    const pos600000 = output.indexOf('SH600000');
    const pos601398 = output.indexOf('SH601398');
    const pos000001 = output.indexOf('SZ000001');

    // Descending order: SH600000 > SH601398 > SZ000001
    expect(pos600000).toBeLessThan(pos601398);
    expect(pos601398).toBeLessThan(pos000001);
  });
});
