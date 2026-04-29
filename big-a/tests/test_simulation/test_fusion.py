from __future__ import annotations

import pytest

from big_a.simulation.fusion import SignalFusion
from big_a.simulation.types import SignalSource, SignalStrength, StockSignal


def make_llm_signal(stock_code: str, score: float) -> StockSignal:
    return StockSignal(
        stock_code=stock_code,
        score=score,
        signal=SignalStrength.BUY if score > 0 else SignalStrength.SELL,
        source=SignalSource.llm,
    )


class TestEqualWeightFusion:
    def test_equal_weight_fusion(self):
        fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)
        llm_signals = [make_llm_signal("A", 0.8), make_llm_signal("B", 0.6)]
        quant_scores = {"A": 0.6, "B": 0.8}
        result = fusion.fuse(llm_signals, quant_scores)
        assert len(result) == 2
        scores = {s.stock_code: s.score for s in result}
        assert scores["A"] == scores["B"]
        assert result[0].source == SignalSource.fused


class TestLlmFallback:
    def test_llm_fallback_to_pure_quant(self):
        fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)
        llm_signals: list[StockSignal] = []
        quant_scores = {"A": 0.8, "B": 0.6}
        result = fusion.fuse(llm_signals, quant_scores)
        assert len(result) == 2
        codes = [s.stock_code for s in result]
        assert "A" in codes
        assert "B" in codes
        assert result[0].source == SignalSource.fused


class TestEmptyInputs:
    def test_empty_inputs(self):
        fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)
        result = fusion.fuse([], {})
        assert result == []


class TestNormalization:
    def test_normalization(self):
        fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)
        llm_signals = [make_llm_signal("A", 0.3), make_llm_signal("B", 0.6), make_llm_signal("C", 0.9)]
        quant_scores = {"A": 0.3, "B": 0.6, "C": 0.9}
        result = fusion.fuse(llm_signals, quant_scores)
        scores = [s.score for s in result]
        assert all(-1.0 <= s <= 1.0 for s in scores)
        assert result[0].score == 1.0
        assert result[-1].score == -1.0


class TestThreeWayFusion:
    def test_three_way_fusion(self):
        fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)
        kronos_scores = {"A": 0.8, "B": 0.4}
        lightgbm_scores = {"A": 0.6, "B": 0.2}
        llm_signals = [make_llm_signal("A", 0.7), make_llm_signal("B", 0.5)]
        result = fusion.fuse_three_way(
            kronos_scores, lightgbm_scores, llm_signals,
            kronos_weight=0.35, lightgbm_weight=0.15
        )
        assert len(result) == 2
        assert result[0].source == SignalSource.fused


class TestSignalStrengthAssignment:
    def test_signal_strength_assignment(self):
        fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)
        llm_signals = [
            make_llm_signal("STRONG_BUY", 0.9),
            make_llm_signal("BUY", 0.3),
            make_llm_signal("HOLD", -0.3),
            make_llm_signal("SELL", -0.6),
            make_llm_signal("STRONG_SELL", -0.9),
        ]
        quant_scores = {code: 0.0 for code in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]}
        result = fusion.fuse(llm_signals, quant_scores)
        signal_map = {s.stock_code: s.signal for s in result}
        assert signal_map["STRONG_BUY"] == SignalStrength.STRONG_BUY
        assert signal_map["BUY"] == SignalStrength.BUY
        assert signal_map["HOLD"] == SignalStrength.HOLD
        assert signal_map["SELL"] == SignalStrength.SELL
        assert signal_map["STRONG_SELL"] == SignalStrength.STRONG_SELL


class TestMissingLlmSignal:
    def test_missing_llm_signal_uses_zero(self):
        fusion = SignalFusion(llm_weight=0.5, quant_weight=0.5)
        llm_signals = [make_llm_signal("A", 0.8)]
        quant_scores = {"A": 0.8, "B": 0.6}
        result = fusion.fuse(llm_signals, quant_scores)
        assert len(result) == 2
        scores = {s.stock_code: s.score for s in result}
        assert "A" in scores
        assert "B" in scores
        assert scores["A"] > scores["B"]


class TestWeightValidation:
    def test_invalid_weights(self):
        with pytest.raises(ValueError):
            SignalFusion(llm_weight=0.3, quant_weight=0.3)

    def test_valid_weights(self):
        fusion = SignalFusion(llm_weight=0.6, quant_weight=0.4)
        assert fusion.llm_weight == 0.6
        assert fusion.quant_weight == 0.4