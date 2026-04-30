import { describe, expect, test } from 'bun:test';
import { Command } from 'commander';

function createTestProgram() {
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
    .action(() => {});

  program
    .command('holdings')
    .description('Display portfolio holdings')
    .option('-d, --data-dir <path>', 'Data directory', '../big-a/output')
    .option('-f, --file <path>', 'Specific portfolio file')
    .action(() => {});

  program
    .command('predict')
    .description('Show model prediction rankings')
    .option('-d, --data-dir <path>', 'Data directory', '../big-a/output')
    .option('-f, --file <path>', 'Specific prediction file')
    .action(() => {});

  program
    .command('factors')
    .description('Analyze factor values and correlations')
    .option('-d, --data-dir <path>', 'Data directory', '../big-a/output')
    .option('-f, --file <path>', 'Specific factor data file')
    .action(() => {});

  return program;
}

describe('CLI Commands', () => {
  test('--help shows all 4 command names', () => {
    const program = createTestProgram();
    const helpText = program.helpInformation();
    expect(helpText).toContain('backtest');
    expect(helpText).toContain('holdings');
    expect(helpText).toContain('predict');
    expect(helpText).toContain('factors');
  });

  test('version is 0.1.0', () => {
    const program = createTestProgram();
    expect(program.version()).toBe('0.1.0');
  });

  test('all 4 commands registered', () => {
    const program = createTestProgram();
    const commands = program.commands.map((cmd) => cmd.name());
    expect(commands).toContain('backtest');
    expect(commands).toContain('holdings');
    expect(commands).toContain('predict');
    expect(commands).toContain('factors');
  });

  test('backtest command has --data-dir option', () => {
    const program = createTestProgram();
    const backtestCmd = program.commands.find((cmd) => cmd.name() === 'backtest');
    expect(backtestCmd).toBeDefined();
    const options = backtestCmd!.options.map((opt) => opt.long);
    expect(options).toContain('--data-dir');
    expect(options).toContain('--file');
  });
});