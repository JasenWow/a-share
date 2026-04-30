import React from 'react';
import { render, Box, Text } from 'ink';
import { existsSync } from 'fs';
import { resolve, join } from 'path';
import { loadPortfolio, parseWatchlistSummary, computePortfolioAllocation } from '@viz/core';
import { readFile } from 'fs/promises';
import { HoldingsView } from '../views/HoldingsView.js';

interface HoldingsOptions {
  dataDir: string;
  file?: string;
}

export async function runHoldingsCommand(options: HoldingsOptions) {
  const dataDir = resolve(options.dataDir);

  if (!existsSync(dataDir)) {
    console.error(`Error: Data directory not found: ${dataDir}`);
    process.exit(1);
  }

  const portfolioPath = options.file ? resolve(options.file) : join(dataDir, 'portfolio.parquet');
  const summaryPath = join(dataDir, 'watchlist_summary.json');

  const positions = existsSync(portfolioPath) ? loadPortfolio(portfolioPath) : [];

  let summary = null;
  if (existsSync(summaryPath)) {
    try {
      const raw = JSON.parse(await readFile(summaryPath, 'utf-8'));
      summary = parseWatchlistSummary(raw);
    } catch {
      render(
        <Box flexDirection="column" padding={1}>
          <Text color="yellow">Warning: Could not parse watchlist summary.</Text>
        </Box>,
      );
    }
  }

  const allocation = computePortfolioAllocation(positions);

  const { unmount } = render(<HoldingsView positions={positions} summary={summary} allocation={allocation} />);
  process.stdout.write('\x1b[?25h');
  unmount();
}
