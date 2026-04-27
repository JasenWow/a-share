# Models Module Knowledge Base

**Generated:** 2026-04-27

## OVERVIEW

ML model implementations: Kronos transformer (OHLCV prediction), LightGBM (Alpha158 factor model), and HedgeFund multi-agent LLM system.

## STRUCTURE

```
models/
├── kronos.py                  # KronosSignalGenerator wrapper (HuggingFace inference)
├── kronos_model/              # Kronos core package
│   ├── kronos.py              # Kronos, KronosPredictor classes
│   └── module.py              # Transformer architecture
├── lightgbm_model.py           # LightGBM model (train/predict/save/load)
└── hedge_fund/                # LLM multi-agent portfolio system
    ├── agents/                # Named agents (warren_buffett, bill_ackman, etc.)
    ├── graph/                  # LangGraph workflow definitions
    ├── tools/                  # News and Qlib data tools
    ├── signal_generator.py    # HedgeFundSignalGenerator (main entry)
    ├── llm.py                 # LLM provider configuration
    └── types.py               # Pydantic types (HedgeFundState, AgentSignal, etc.)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Kronos inference | `kronos.py` | KronosSignalGenerator, OHLCV signal generation |
| Kronos architecture | `kronos_model/kronos.py` | KronosPredictor, Transformer model |
| LightGBM training | `lightgbm_model.py` | train/predict/save/load functions |
| Multi-model scoring | `report/scorer.py` | WatchlistScorer combines Kronos + LightGBM |
| HedgeFund agents | `hedge_fund/agents/` | 11 investor agents (Buffett, Ackman, etc.) |
| Agent state | `hedge_fund/types.py` | HedgeFundState, AgentSignal, PortfolioDecision |

## CONVENTIONS

- Kronos: lookback=90 days, pred_len=10 days, signal_mode="mean" by default
- LightGBM: uses Alpha158 features via Qlib DatasetH, output score in [-1, 1]
- All models output DataFrame with MultiIndex (datetime, instrument) and "score" column
- WatchlistScorer aggregates Kronos rolling trend + LightGBM factor scores
- AgentSignal uses Literal["bullish", "bearish", "neutral"] for direction

## ANTI-PATTERNS

- Do not call kronos.predict directly on raw DataFrame; use KronosSignalGenerator.preprocess first
- Do not instantiate LLM directly; use hedge_fund/llm.py provider abstraction
- Do not mix Qlib DatasetH with pandas DataFrame in model input pipelines