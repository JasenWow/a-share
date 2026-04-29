from __future__ import annotations

from unittest.mock import MagicMock

from big_a.llm.client import LLMClient, LLMError
from big_a.llm.decision import LLMAnalysisOutput, LLMTradingDecision, StockAnalysis
from big_a.simulation.types import SignalSource, SignalStrength


class TestLLMTradingDecision:
    """Tests for LLMTradingDecision class."""

    def test_analyze_stocks_success(self):
        """Mock LLM returns valid JSON with 3 stock analyses."""
        mock_client = MagicMock(spec=LLMClient)
        mock_output = LLMAnalysisOutput(
            analyses=[
                StockAnalysis(stock_code="000001", score=0.8, signal=SignalStrength.BUY, reasoning="Test 1"),
                StockAnalysis(stock_code="000002", score=-0.6, signal=SignalStrength.SELL, reasoning="Test 2"),
                StockAnalysis(stock_code="000003", score=0.1, signal=SignalStrength.HOLD, reasoning="Test 3"),
            ],
            market_view="市场震荡",
        )
        mock_client.chat_structured.return_value = mock_output

        decision = LLMTradingDecision(mock_client)
        market_data = {
            "000001": {"ohlc": [{"open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}], "name": "平安银行"},
            "000002": {"ohlc": [{"open": 20, "high": 21, "low": 19, "close": 19.5, "volume": 2000}], "name": "万科A"},
            "000003": {"ohlc": [{"open": 30, "high": 31, "low": 29, "close": 30.2, "volume": 1500}], "name": "格力电器"},
        }
        quant_scores = {"000001": 0.7, "000002": -0.5, "000003": 0.2}

        signals = decision.analyze_stocks(market_data, quant_scores)

        assert len(signals) == 3
        assert signals[0].stock_code == "000001"
        assert signals[0].score == 0.8
        assert signals[0].signal == SignalStrength.BUY
        assert signals[0].source == SignalSource.llm
        assert signals[1].stock_code == "000002"
        assert signals[1].score == -0.6
        assert signals[1].signal == SignalStrength.SELL
        assert signals[2].stock_code == "000003"
        assert signals[2].score == 0.1
        mock_client.chat_structured.assert_called_once()

    def test_analyze_stocks_empty_data(self):
        """Empty market_data and quant_scores returns empty list."""
        mock_client = MagicMock(spec=LLMClient)
        mock_output = LLMAnalysisOutput(analyses=[], market_view="市场观望")
        mock_client.chat_structured.return_value = mock_output

        decision = LLMTradingDecision(mock_client)
        signals = decision.analyze_stocks({}, {})

        assert signals == []

    def test_invalid_json_fallback(self):
        """Mock LLM raises LLMError returns empty list, no exception."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat_structured.side_effect = LLMError("Parse failed")

        decision = LLMTradingDecision(mock_client)
        signals = decision.analyze_stocks(
            {"000001": {"ohlc": [], "name": "Test"}},
            {"000001": 0.5},
        )

        assert signals == []

    def test_api_failure_fallback(self):
        """Mock LLM raises Exception returns empty list."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client.chat_structured.side_effect = RuntimeError("API unavailable")

        decision = LLMTradingDecision(mock_client)
        signals = decision.analyze_stocks(
            {"000001": {"ohlc": [], "name": "Test"}},
            {"000001": 0.5},
        )

        assert signals == []

    def test_score_clamping(self):
        """Mock LLM returns score=1.5 is clamped to 1.0."""
        mock_client = MagicMock(spec=LLMClient)
        mock_output = LLMAnalysisOutput(
            analyses=[
                StockAnalysis(stock_code="000001", score=1.5, signal=SignalStrength.STRONG_BUY, reasoning="Overbought"),
            ],
            market_view="看多",
        )
        mock_client.chat_structured.return_value = mock_output

        decision = LLMTradingDecision(mock_client)
        market_data = {"000001": {"ohlc": [], "name": "Test"}}
        quant_scores = {"000001": 0.9}

        signals = decision.analyze_stocks(market_data, quant_scores)

        assert len(signals) == 1
        assert signals[0].score == 1.0
        assert signals[0].stock_code == "000001"

    def test_build_analysis_prompt_content(self):
        """Verify prompt contains stock codes and quant scores."""
        mock_client = MagicMock(spec=LLMClient)
        mock_output = LLMAnalysisOutput(analyses=[], market_view="")
        mock_client.chat_structured.return_value = mock_output

        decision = LLMTradingDecision(mock_client)
        market_data = {
            "000001": {
                "ohlc": [
                    {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000},
                ],
                "name": "平安银行",
            },
        }
        quant_scores = {"000001": 0.75}

        decision.analyze_stocks(market_data, quant_scores)

        call_args = mock_client.chat_structured.call_args
        system_prompt = call_args[0][0]
        user_message = call_args[0][1]

        assert "000001" in user_message
        assert "平安银行" in user_message
        assert "0.750" in user_message
        assert "A股量化分析师" in system_prompt