"""LLM-based trading decision module using MiniMax for stock analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

from big_a.simulation.types import SignalSource, SignalStrength, StockSignal

if TYPE_CHECKING:
    from big_a.llm.client import LLMClient


class StockAnalysis(BaseModel):
    """Pydantic output model for individual stock analysis from LLM."""

    stock_code: str
    score: float
    signal: SignalStrength
    reasoning: str


class LLMAnalysisOutput(BaseModel):
    analyses: list[StockAnalysis]
    market_view: str


class LLMTradingDecision:
    """LLM-based trading decision maker using MiniMax to analyze stocks and produce signals."""

    def __init__(self, llm_client: LLMClient, temperature: float = 0.3) -> None:
        """Initialize LLMTradingDecision.

        Args:
            llm_client: LLM client instance for making API calls.
            temperature: Sampling temperature for LLM (default 0.3).
        """
        self._llm_client = llm_client
        self._temperature = temperature

    def analyze_stocks(
        self, market_data: dict[str, dict], quant_scores: dict[str, float]
    ) -> list[StockSignal]:
        """Analyze stocks using LLM and return trading signals.

        Args:
            market_data: Market data dict with stock codes as keys.
                Format: {stock_code: {"ohlc": list[dict], "name": str}}
                Each ohlc dict has: open, high, low, close, volume.
            quant_scores: Quant model scores with stock codes as keys.
                Format: {stock_code: float} — score from -1 to 1.

        Returns:
            List of StockSignal objects with LLM analysis.
            Returns empty list if LLM call fails (fallback to pure quant).
        """
        try:
            system_prompt, user_message = self._build_analysis_prompt(market_data, quant_scores)
            output = self._llm_client.chat_structured(
                system_prompt, user_message, LLMAnalysisOutput
            )
            return self._parse_to_signals(output)
        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to pure quant: {e}")
            return []

    def _build_analysis_prompt(
        self, market_data: dict[str, dict], quant_scores: dict[str, float]
    ) -> tuple[str, str]:
        """Build system and user prompts for stock analysis.

        Args:
            market_data: Market data dict with OHLC data per stock.
            quant_scores: Quant model scores per stock.

        Returns:
            Tuple of (system_prompt, user_message).
        """
        system_prompt = (
            "你是一位专业的A股量化分析师。根据提供的技术面数据和量化模型评分，对以下股票进行分析，"
            "判断每个股票的涨跌信号和强度。你的输出必须符合指定的JSON格式。"
        )

        # Build stock table
        lines: list[str] = ["股票代码\t股票名称\t近期行情\t量化评分"]
        lines.append("-" * 60)

        for stock_code, data in market_data.items():
            name = data.get("name", stock_code)
            ohlc_list = data.get("ohlc", [])

            if ohlc_list:
                ohlc_strs = []
                for bar in ohlc_list[-5:]:
                    ohlc_strs.append(
                        f"开:{bar.get('open', 0):.2f} 高:{bar.get('high', 0):.2f} "
                        f"低:{bar.get('low', 0):.2f} 收:{bar.get('close', 0):.2f} 量:{bar.get('volume', 0):.0f}"
                    )
                ohlc_formatted = " | ".join(ohlc_strs)
            else:
                ohlc_formatted = "无数据"

            quant_score = quant_scores.get(stock_code, 0.0)
            lines.append(f"{stock_code}\t{name}\t{ohlc_formatted}\t{quant_score:.3f}")

        user_message = "请分析以下股票并给出JSON格式的评分和信号：\n\n" + "\n".join(lines)

        return system_prompt, user_message

    def _parse_to_signals(self, output: LLMAnalysisOutput) -> list[StockSignal]:
        """Parse LLMAnalysisOutput into list of StockSignal.

        Args:
            output: Parsed LLM analysis output.

        Returns:
            List of StockSignal with clamped scores.
        """
        signals = []
        for analysis in output.analyses:
            clamped_score = max(-1.0, min(1.0, analysis.score))
            signal = StockSignal(
                stock_code=analysis.stock_code,
                score=clamped_score,
                signal=analysis.signal,
                source=SignalSource.llm,
                reasoning=analysis.reasoning,
            )
            signals.append(signal)
        return signals