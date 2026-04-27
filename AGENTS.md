# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-27
**Commit:** cb01193
**Branch:** main

---

name: karpathy-guidelines
description: Behavioral guidelines to reduce common LLM coding mistakes. Use when writing, reviewing, or refactoring code to avoid overcomplication, make surgical changes, surface assumptions, and define verifiable success criteria.
license: MIT

---

# Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## OVERVIEW

Quantitative stock trading system with Qlib integration. Models: Kronos (transformer) + LightGBM (factor-based). Supports backtesting, real trading simulation, and multi-model portfolio scoring.

## STRUCTURE

```
stock-big-a/
├── AGENTS.md              # This file (root knowledge)
├── big-a/                 # Main package
│   ├── src/big_a/         # Source modules (11 subpackages)
│   ├── scripts/           # CLI entry points (18 scripts)
│   ├── tests/             # Test suite (20+ files)
│   ├── configs/            # YAML configs (backtest, data, model)
│   └── pyproject.toml     # hatchling build, ruff linting
├── docs/                  # Documentation
└── mlruns/                # MLflow experiment tracking
```

## WHERE TO LOOK

| Task               | Location              | Notes                                         |
| ------------------ | --------------------- | --------------------------------------------- |
| Add/Edit model     | `src/big_a/models/`   | kronos_model/, lightgbm_model.py, hedge_fund/ |
| Backtest logic     | `src/big_a/backtest/` | engine.py, rolling.py, evaluation.py          |
| Factor definitions | `src/big_a/factors/`  | alpha_factors.py                              |
| Report generation  | `src/big_a/report/`   | formatter.py, scorer.py                       |
| Trading strategy   | `src/big_a/strategy/` | real_trading.py                               |
| Data handling      | `src/big_a/data/`     | updater, loader, validator                    |
| Run experiment     | `scripts/run.py`      | Main CLI with subcommands                     |
| Tests              | `big-a/tests/`        | pytest, conftest.py, skip_if_no_data marker   |

## CONVENTIONS

- **Python**: 3.12+, type hints not enforced
- **Linter**: ruff (line-length=120, py312 target)
- **Build**: hatchling with src-layout
- **CLI**: typer scripts in `scripts/`, invoke via `python scripts/x.py`
- **Testing**: pytest with `skip_if_no_data` marker (requires Qlib CN data)
- **No pyproject.toml entry points**: Scripts not installable as commands

## ANTI-PATTERNS (THIS PROJECT)

- Do not run real-time updates in Phase 1 (README)
- Do not modify files outside `src/big_a/data/updater.py`, `scripts/update_data.py`, `README.md` for data updater task
- Scripts/ is not a package (no `__init__.py`) - cannot `python -m scripts.x`

## UNIQUE STYLES

- Qlib integration for CN stock data (akshare data source)
- MLflow experiment tracking in `mlruns/`
- Prefect workflow orchestration (workflow/ module)
- Multi-model portfolio scoring: WatchlistScorer combines Kronos + LightGBM

## COMMANDS

```bash
cd big-a
python -m pytest tests/ -m "not skip_if_no_data"  # Run tests
ruff check src/                                     # Lint
python scripts/run.py --help                       # Main CLI
python scripts/e2e.py                               # End-to-end pipeline
```

## NOTES

- Data cache in `data/qlib_data/` (~GBs, excluded from git)
- `big-a/.venv` for local venv (not used - uses system python)
- `src/big_a/__init__.py` is minimal (docstring only)
- Root `main.py` is dead placeholder
