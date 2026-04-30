#!/usr/bin/env bun
import { Command } from 'commander';
import { runBacktestCommand } from './commands/backtest.js';
import { runHoldingsCommand } from './commands/holdings.js';
import { runPredictCommand } from './commands/predict.js';
import { runFactorsCommand } from './commands/factors.js';

const program = new Command();

program
  .name('cli-viz')
  .description('CLI visualization toolkit for stock trading analysis')
  .version('0.1.0');

program
  .command('backtest')
  .description('Visualize backtest results')
  .option('-d, --data-dir <path>', 'Data directory', '../big-a/output')
  .option('-f, --file <path>', 'Specific backtest report file')
  .action(runBacktestCommand);

program
  .command('holdings')
  .description('Display portfolio holdings')
  .option('-d, --data-dir <path>', 'Data directory', '../big-a/output')
  .option('-f, --file <path>', 'Specific portfolio file')
  .action(runHoldingsCommand);

program
  .command('predict')
  .description('Show model prediction rankings')
  .option('-d, --data-dir <path>', 'Data directory', '../big-a/output')
  .option('-f, --file <path>', 'Specific prediction file')
  .action(runPredictCommand);

program
  .command('factors')
  .description('Analyze factor values and correlations')
  .option('-d, --data-dir <path>', 'Data directory', '../big-a/output')
  .option('-f, --file <path>', 'Specific factor data file')
  .action(runFactorsCommand);

program.parse();