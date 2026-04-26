"""Hedge Fund Signal Generator - Qlib-compatible signal generation using multi-agent LLM workflow."""
from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from big_a.models.hedge_fund.graph.workflow import create_workflow, run_workflow
from big_a.models.hedge_fund.types import HedgeFundState


class HedgeFundSignalGenerator:
    """Generate trading signals using hedge fund multi-agent LLM workflow.

    Usage::

        gen = HedgeFundSignalGenerator(config)
        signals = gen.generate_signals(
            instruments=["SH600000", "SZ000001"],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        # signals: DataFrame(index=[datetime, instrument], columns=["score"])

    The workflow runs multiple LLM agents in parallel (technicals, valuation, sentiment,
    and 10 investor personas), aggregates their signals through a risk manager, and
    produces a final portfolio decision with a score in [-1, 1] range.

    Parameters
    ----------
    config : dict or None
        Configuration dict with agent settings, LLM settings, etc.
        If None, uses default config from workflow.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def generate_signals(
        self,
        instruments: list[str] | str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Generate trading signals using the hedge fund multi-agent workflow.

        Runs the full LangGraph workflow for each instrument:
        1. Fan-out to parallel agents (technicals, valuation, sentiment, 10 investors)
        2. Risk manager aggregates all signals
        3. Portfolio manager makes final decision with score [-1, 1]
        4. Returns Qlib-compatible DataFrame

        Parameters
        ----------
        instruments : list[str] or str
            Qlib instrument codes (e.g. "SH600000" or ["SH600000", "SZ000001"]).
        start_date, end_date : str
            Date range for analysis (YYYY-MM-DD format).

        Returns
        -------
        pd.DataFrame
            MultiIndex (datetime, instrument), column "score".
            Score range: [-1, 1] for Qlib backtest compatibility.
            datetime is the end_date when the signal is available.
        """
        if isinstance(instruments, str):
            instruments = [instruments]

        all_scores = []

        for instrument in instruments:
            logger.info(f"Generating signal for {instrument} from {start_date} to {end_date}")

            try:
                workflow = create_workflow(config=self.config)
                result = run_workflow(
                    graph=workflow,
                    ticker=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    config=self.config,
                )

                decision = result.get("data", {}).get("portfolio_decision", {})
                score = self._extract_score(decision)

                all_scores.append({
                    "datetime": pd.Timestamp(end_date),
                    "instrument": instrument,
                    "score": float(score),
                })

                logger.info(f"Signal for {instrument}: {score:.3f}")

            except Exception as e:
                logger.warning(f"Workflow failed for {instrument}, returning neutral score: {e}")
                all_scores.append({
                    "datetime": pd.Timestamp(end_date),
                    "instrument": instrument,
                    "score": 0.0,
                })

        if not all_scores:
            logger.warning("No signals generated")
            return pd.DataFrame(
                columns=["score"],
                index=pd.MultiIndex.from_tuples([], names=["datetime", "instrument"]),
            )

        df = pd.DataFrame(all_scores)
        df = df.set_index(["datetime", "instrument"])
        result = df[["score"]]
        result.index.names = ["datetime", "instrument"]
        logger.info(f"Generated {len(result)} signals")
        return result

    def _extract_score(self, decision: Any) -> float:
        """Extract score from portfolio decision.

        Handles both PortfolioDecision objects and dict representations.

        Parameters
        ----------
        decision : PortfolioDecision or dict
            Portfolio decision from workflow.

        Returns
        -------
        float
            Score value in [-1, 1] range, or 0.0 if extraction fails.
        """
        if isinstance(decision, dict):
            score = decision.get("score", 0.0)
        elif hasattr(decision, "score"):
            score = getattr(decision, "score", 0.0)
        elif hasattr(decision, "model_dump"):
            score = decision.model_dump().get("score", 0.0)
        elif hasattr(decision, "dict"):
            score = decision.dict().get("score", 0.0)
        else:
            score = 0.0

        try:
            score = float(score)
        except (TypeError, ValueError):
            logger.warning(f"Invalid score value: {score}, using 0.0")
            score = 0.0

        if score < -1.0 or score > 1.0:
            logger.warning(f"Score {score} out of range [-1, 1], clamping")
            score = max(-1.0, min(1.0, score))

        return score
