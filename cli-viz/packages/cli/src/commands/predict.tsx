import React from 'react';
import { render } from 'ink';
import { existsSync } from 'fs';
import { resolve, join } from 'path';
import { loadKronosSignals, loadLightGBMPredictions, mergeModelScores } from '@viz/core';
import { PredictionView } from '../views/PredictionView.js';

interface PredictOptions {
  dataDir: string;
  file?: string;
}

export function runPredictCommand(options: PredictOptions) {
  const dataDir = resolve(options.dataDir);

  if (!existsSync(dataDir)) {
    console.error(`Error: Data directory not found: ${dataDir}`);
    process.exit(1);
  }

  const kronosPath = join(dataDir, 'kronos_signals.csv');
  const lightgbmPath = join(dataDir, 'predictions.parquet');

  const kronos = existsSync(kronosPath) ? loadKronosSignals(kronosPath) : [];
  const lightgbm = existsSync(lightgbmPath) ? loadLightGBMPredictions(lightgbmPath) : [];
  const merged = mergeModelScores(kronos, lightgbm);

  const { unmount } = render(<PredictionView kronos={kronos} lightgbm={lightgbm} merged={merged} />);
  process.stdout.write('\x1b[?25h');
  unmount();
}