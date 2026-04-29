"""Signal fusion module combining LLM and quantitative model scores.

Provides SignalFusion class for weighted combination of LLM signals
and quantitative model (Kronos, LightGBM) scores.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from big_a.simulation.types import SignalSource, SignalStrength, StockSignal


class SignalFusion:
    """Fuses LLM and quantitative model scores into combined signals.

    Supports two fusion modes:
    - Two-way: LLM + single quant source (Kronos or LightGBM)
    - Three-way: Kronos + LightGBM + LLM

    Weights are validated to sum to ~1.0 (tolerance 0.01).
    Combined scores are min-max normalized to [-1, 1] range.

    Example::

        fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)
        signals = fusion.fuse(llm_signals, quant_scores)
    """

    def __init__(
        self,
        llm_weight: float = 0.5,
        quant_weight: float = 0.5,
    ) -> None:
        """Initialize SignalFusion.

        Parameters
        ----------
        llm_weight : float
            Weight for LLM signals in [0, 1].
        quant_weight : float
            Weight for quantitative signals in [0, 1].

        Raises
        ------
        ValueError
            If weights don't sum to ~1.0 (tolerance 0.01).
        """
        total = llm_weight + quant_weight
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(
                f"Weights must sum to 1.0, got llm_weight={llm_weight}, quant_weight={quant_weight}"
            )
        self.llm_weight = llm_weight
        self.quant_weight = quant_weight

    def _assign_strength(self, score: float) -> SignalStrength:
        """Assign SignalStrength based on normalized score.

        Parameters
        ----------
        score : float
            Combined score in [-1, 1].

        Returns
        -------
        SignalStrength
            Signal strength enum value.
        """
        if score > 0.5:
            return SignalStrength.STRONG_BUY
        elif score > 0:
            return SignalStrength.BUY
        elif score > -0.5:
            return SignalStrength.HOLD
        elif score > -1.0:
            return SignalStrength.SELL
        else:
            return SignalStrength.STRONG_SELL

    def _normalize_scores(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Min-max normalize scores to [-1, 1] range.

        If all scores are identical, returns all zeros to avoid division by zero.

        Parameters
        ----------
        scores : dict[str, float]
            Mapping of stock codes to raw combined scores.

        Returns
        -------
        dict[str, float]
            Mapping of stock codes to normalized scores in [-1, 1].
        """
        if not scores:
            return {}

        values = np.array(list(scores.values()))
        min_val = values.min()
        max_val = values.max()

        if np.isclose(min_val, max_val):
            logger.debug("All scores identical, returning zeros to avoid division by zero")
            return {k: 0.0 for k in scores}

        normalized = (values - min_val) / (max_val - min_val)
        normalized = normalized * 2 - 1

        return dict(zip(scores.keys(), normalized.tolist()))

    def fuse(
        self,
        llm_signals: list[StockSignal],
        quant_scores: dict[str, float],
    ) -> list[StockSignal]:
        """Fuse LLM signals with quantitative scores.

        If llm_signals is empty, uses 100% quant_weight (pure quant fallback).
        For each stock in the union of inputs:
        - llm_score = matching signal's score (0.0 if no LLM signal)
        - quant_score = from quant_scores dict (0.0 if not present)
        - combined = llm_weight * llm_score + quant_weight * quant_score

        Results are min-max normalized to [-1, 1] and sorted descending.

        Parameters
        ----------
        llm_signals : list[StockSignal]
            List of LLM-generated signals.
        quant_scores : dict[str, float]
            Mapping of stock codes to quantitative scores.

        Returns
        -------
        list[StockSignal]
            Sorted list of fused signals (highest score first).
        """
        if not llm_signals and not quant_scores:
            return []

        llm_score_map: dict[str, float] = {s.stock_code: s.score for s in llm_signals}

        all_stocks = set(llm_score_map.keys()) | set(quant_scores.keys())

        raw_combined: dict[str, float] = {}

        if not llm_signals:
            effective_llm_weight = 0.0
            effective_quant_weight = 1.0
        else:
            effective_llm_weight = self.llm_weight
            effective_quant_weight = self.quant_weight

        for stock in all_stocks:
            llm_score = llm_score_map.get(stock, 0.0)
            quant_score = quant_scores.get(stock, 0.0)
            combined = effective_llm_weight * llm_score + effective_quant_weight * quant_score
            raw_combined[stock] = combined

        normalized = self._normalize_scores(raw_combined)

        result: list[StockSignal] = []
        for stock_code, score in normalized.items():
            signal = StockSignal(
                stock_code=stock_code,
                score=score,
                signal=self._assign_strength(score),
                source=SignalSource.fused,
                reasoning=f"Fused (llm_w={effective_llm_weight:.2f}, quant_w={effective_quant_weight:.2f})",
            )
            result.append(signal)

        result.sort(key=lambda x: x.score, reverse=True)

        return result

    def fuse_three_way(
        self,
        kronos_scores: dict[str, float],
        lightgbm_scores: dict[str, float],
        llm_signals: list[StockSignal],
        kronos_weight: float = 0.35,
        lightgbm_weight: float = 0.15,
        llm_weight: float = 0.5,
    ) -> list[StockSignal]:
        """Fuse three-way: Kronos + LightGBM + LLM.

        First merges Kronos and LightGBM into combined_quant using their weights
        (normalized by combined weight = 0.5). Then fuses combined_quant with LLM
        using self.llm_weight / self.quant_weight ratio.

        Parameters
        ----------
        kronos_scores : dict[str, float]
            Mapping of stock codes to Kronos model scores.
        lightgbm_scores : dict[str, float]
            Mapping of stock codes to LightGBM model scores.
        llm_signals : list[StockSignal]
            List of LLM-generated signals.
        kronos_weight : float
            Weight for Kronos scores (default 0.35).
        lightgbm_weight : float
            Weight for LightGBM scores (default 0.15).
        llm_weight : float
            Weight for LLM signals (default 0.5).

        Returns
        -------
        list[StockSignal]
            Sorted list of fused three-way signals (highest score first).
        """
        quant_sum = kronos_weight + lightgbm_weight
        if not np.isclose(quant_sum, 0.5, atol=0.01):
            raise ValueError(
                f"Kronos + LightGBM weights must sum to 0.5, got {quant_sum}"
            )

        all_quant_stocks = set(kronos_scores.keys()) | set(lightgbm_scores.keys())
        combined_quant: dict[str, float] = {}

        for stock in all_quant_stocks:
            kronos = kronos_scores.get(stock, 0.0)
            lightgbm = lightgbm_scores.get(stock, 0.0)
            combined_quant[stock] = (kronos_weight * kronos + lightgbm_weight * lightgbm) / 0.5

        return self.fuse(llm_signals, combined_quant)