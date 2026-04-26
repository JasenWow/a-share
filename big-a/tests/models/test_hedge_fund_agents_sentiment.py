"""Tests for sentiment analysis agent."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from big_a.models.hedge_fund.agents.sentiment import sentiment_agent
from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState


class TestSentimentAgent:
    """Tests for sentiment_agent function."""

    @patch("big_a.models.hedge_fund.agents.sentiment.get_stock_news")
    @patch("big_a.models.hedge_fund.agents.sentiment.call_llm")
    def test_sentiment_agent_with_positive_news(self, mock_call_llm: MagicMock, mock_get_news: MagicMock) -> None:
        """Test that positive news generates bullish signal."""
        mock_get_news.return_value = [
            {"title": "公司发布重大利好消息", "content": "公司宣布与大型企业签署战略合作协议", "date": "2024-01-01", "source": "test"},
            {"title": "业绩超预期", "content": "公司净利润同比增长50%", "date": "2024-01-02", "source": "test"},
        ]
        mock_call_llm.return_value = AgentSignal(
            agent_name="llm_response",
            signal="bullish",
            confidence=0.85,
            reasoning="Positive news indicates strong growth prospects",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        }

        result = sentiment_agent(state)

        assert "data" in result
        assert "analyst_signals" in result["data"]
        assert "sentiment_agent" in result["data"]["analyst_signals"]
        signal = result["data"]["analyst_signals"]["sentiment_agent"]
        assert isinstance(signal, AgentSignal)
        assert signal.signal == "bullish"
        assert signal.confidence == 0.85
        assert signal.agent_name == "sentiment_agent"
        mock_get_news.assert_called_once_with("SH600000", "2024-01-01", "2024-01-02", None)
        mock_call_llm.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.sentiment.get_stock_news")
    @patch("big_a.models.hedge_fund.agents.sentiment.call_llm")
    def test_sentiment_agent_with_negative_news(self, mock_call_llm: MagicMock, mock_get_news: MagicMock) -> None:
        """Test that negative news generates bearish signal."""
        mock_get_news.return_value = [
            {"title": "公司面临监管处罚", "content": "因违规操作被监管部门处罚", "date": "2024-01-01", "source": "test"},
            {"title": "业绩大幅下滑", "content": "公司净利润同比下降30%", "date": "2024-01-02", "source": "test"},
        ]
        mock_call_llm.return_value = AgentSignal(
            agent_name="llm_response",
            signal="bearish",
            confidence=0.9,
            reasoning="Negative news indicates significant downside risk",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"ticker": "SZ000001", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        }

        result = sentiment_agent(state)

        signal = result["data"]["analyst_signals"]["sentiment_agent"]
        assert signal.signal == "bearish"
        assert signal.confidence == 0.9
        assert signal.agent_name == "sentiment_agent"

    @patch("big_a.models.hedge_fund.agents.sentiment.get_stock_news")
    @patch("big_a.models.hedge_fund.agents.sentiment.call_llm")
    def test_sentiment_agent_with_mixed_news(self, mock_call_llm: MagicMock, mock_get_news: MagicMock) -> None:
        """Test that mixed news generates neutral signal."""
        mock_get_news.return_value = [
            {"title": "公司发布新产品", "content": "新产品市场反响良好", "date": "2024-01-01", "source": "test"},
            {"title": "成本上升压力", "content": "原材料价格上涨影响毛利率", "date": "2024-01-02", "source": "test"},
        ]
        mock_call_llm.return_value = AgentSignal(
            agent_name="llm_response",
            signal="neutral",
            confidence=0.5,
            reasoning="Mixed news with both positive and negative factors",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"ticker": "SH600519", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        }

        result = sentiment_agent(state)

        signal = result["data"]["analyst_signals"]["sentiment_agent"]
        assert signal.signal == "neutral"
        assert signal.confidence == 0.5
        assert signal.agent_name == "sentiment_agent"

    @patch("big_a.models.hedge_fund.agents.sentiment.get_stock_news")
    def test_sentiment_agent_with_no_news(self, mock_get_news: MagicMock) -> None:
        """Test that no news returns neutral signal with zero confidence."""
        mock_get_news.return_value = []

        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        }

        result = sentiment_agent(state)

        signal = result["data"]["analyst_signals"]["sentiment_agent"]
        assert signal.signal == "neutral"
        assert signal.confidence == 0.0
        assert "No news available for analysis" in signal.reasoning
        assert signal.agent_name == "sentiment_agent"
        mock_get_news.assert_called_once()

    @patch("big_a.models.hedge_fund.agents.sentiment.get_stock_news")
    def test_sentiment_agent_with_news_returning_none(self, mock_get_news: MagicMock) -> None:
        """Test that news returning None is handled gracefully."""
        mock_get_news.return_value = None

        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        }

        result = sentiment_agent(state)

        signal = result["data"]["analyst_signals"]["sentiment_agent"]
        assert signal.signal == "neutral"
        assert signal.confidence == 0.0
        assert "No news available for analysis" in signal.reasoning

    def test_sentiment_agent_missing_ticker(self) -> None:
        """Test that missing ticker returns neutral signal."""
        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"start_date": "2024-01-01", "end_date": "2024-01-02"},
        }

        result = sentiment_agent(state)

        signal = result["data"]["analyst_signals"]["sentiment_agent"]
        assert signal.signal == "neutral"
        assert signal.confidence == 0.0
        assert "Insufficient data" in signal.reasoning

    def test_sentiment_agent_missing_dates(self) -> None:
        """Test that missing dates returns neutral signal."""
        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"ticker": "SH600000"},
        }

        result = sentiment_agent(state)

        signal = result["data"]["analyst_signals"]["sentiment_agent"]
        assert signal.signal == "neutral"
        assert signal.confidence == 0.0
        assert "Insufficient data" in signal.reasoning

    @patch("big_a.models.hedge_fund.agents.sentiment.get_stock_news")
    @patch("big_a.models.hedge_fund.agents.sentiment.call_llm")
    def test_sentiment_agent_with_config(self, mock_call_llm: MagicMock, mock_get_news: MagicMock) -> None:
        """Test that config is passed to news_tools and call_llm."""
        mock_get_news.return_value = [
            {"title": "测试新闻", "content": "测试内容", "date": "2024-01-01", "source": "test"},
        ]
        mock_call_llm.return_value = AgentSignal(
            agent_name="llm_response",
            signal="neutral",
            confidence=0.5,
            reasoning="Test reasoning",
        )

        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        }

        config = {"news": {"source": "test"}, "llm": {"model": "test-model"}}

        result = sentiment_agent(state, config)

        mock_get_news.assert_called_once_with("SH600000", "2024-01-01", "2024-01-02", config)
        mock_call_llm.assert_called_once()
        call_args = mock_call_llm.call_args
        assert call_args[0][2] == config

    @patch("big_a.models.hedge_fund.agents.sentiment.get_stock_news")
    @patch("big_a.models.hedge_fund.agents.sentiment.call_llm")
    def test_sentiment_agent_llm_error_handling(self, mock_call_llm: MagicMock, mock_get_news: MagicMock) -> None:
        """Test that LLM errors are handled gracefully."""
        mock_get_news.return_value = [
            {"title": "测试新闻", "content": "测试内容", "date": "2024-01-01", "source": "test"},
        ]
        mock_call_llm.side_effect = Exception("LLM API error")

        state: HedgeFundState = {
            "messages": [],
            "data": {},
            "metadata": {"ticker": "SH600000", "start_date": "2024-01-01", "end_date": "2024-01-02"},
        }

        result = sentiment_agent(state)

        signal = result["data"]["analyst_signals"]["sentiment_agent"]
        assert signal.signal == "neutral"
        assert signal.confidence == 0.0
        assert "Error in sentiment analysis" in signal.reasoning
        assert signal.agent_name == "sentiment_agent"
