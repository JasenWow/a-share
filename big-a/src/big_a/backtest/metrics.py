"""Evaluation metric thresholds and success criteria."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Success thresholds
# ---------------------------------------------------------------------------

SUCCESS_IC: float = 0.03
"""Minimum mean Information Coefficient (IC) to consider a model useful."""

SUCCESS_SHARPE: float = 1.0
"""Minimum annualized Sharpe ratio for a strategy to be considered viable."""

MAX_DRAWDOWN_THRESHOLD: float = 0.20
"""Maximum allowed max-drawdown (fraction). Strategies exceeding this are flagged."""

# ---------------------------------------------------------------------------
# Display labels (used by compare_models and CLI)
# ---------------------------------------------------------------------------

METRIC_LABELS: dict[str, str] = {
    "mean_ic": "Mean IC",
    "mean_rank_ic": "Mean Rank IC",
    "icir": "ICIR",
    "sharpe": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown",
    "turnover": "Turnover",
}
