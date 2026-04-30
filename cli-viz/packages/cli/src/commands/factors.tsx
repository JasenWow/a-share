import React from 'react';
import { render } from 'ink';
import { existsSync } from 'fs';
import { resolve, join } from 'path';
import { loadFactorData, computeCorrelationMatrix, computeFactorReturns } from '@viz/core';
import { FactorView } from '../views/FactorView.js';

interface FactorsOptions {
  dataDir: string;
  file?: string;
}

export function runFactorsCommand(options: FactorsOptions) {
  const dataDir = resolve(options.dataDir);

  if (!existsSync(dataDir)) {
    console.error(`Error: Data directory not found: ${dataDir}`);
    process.exit(1);
  }

  const factorFile = options.file ? resolve(options.file) : join(dataDir, 'factors.parquet');

  if (!existsSync(factorFile)) {
    render(<FactorView factors={[]} correlationMatrix={{ factor_names: [], values: [] }} factorReturns={[]} />);
    return;
  }

  const factors = loadFactorData(factorFile);
  const correlationMatrix = computeCorrelationMatrix(factors);
  const factorReturns = factors.length > 0 ? computeFactorReturns(factors, [0.01]) : [];

  render(
    <FactorView
      factors={factors}
      correlationMatrix={correlationMatrix}
      factorReturns={factorReturns}
    />,
  );
}