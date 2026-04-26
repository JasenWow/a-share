"""Risk manager agent for aggregating analyst signals and assessing risk."""
from __future__ import annotations

from typing import Any

from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState, RiskAssessment


def risk_manager_agent(state: HedgeFundState) -> HedgeFundState:
    """Aggregate analyst signals and produce risk assessment.

    Collects all signals from state["data"]["analyst_signals"], computes
    risk metrics (signal agreement, confidence-weighted aggregate, volatility assessment),
    and uses LLM to produce a RiskAssessment.

    Args:
        state: HedgeFundState containing analyst_signals in data field

    Returns:
        Updated HedgeFundState with risk_assessment in data field
    """
    data = state.get("data", {})
    analyst_signals = data.get("analyst_signals", {})
    config = state.get("metadata", {}).get("config")

    if not analyst_signals:
        neutral_assessment = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.0,
            max_position_weight=0.0,
            reasoning="No analyst signals available for risk assessment",
        )
        state.setdefault("data", {})["risk_assessment"] = neutral_assessment.model_dump()
        return state

    signals = _extract_signals(analyst_signals)

    if not signals:
        neutral_assessment = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.0,
            max_position_weight=0.0,
            reasoning="No valid analyst signals available for risk assessment",
        )
        state.setdefault("data", {})["risk_assessment"] = neutral_assessment.model_dump()
        return state

    risk_metrics = _compute_risk_metrics(signals)
    prompt = _build_risk_prompt(signals, risk_metrics)

    try:
        assessment = call_llm(prompt, RiskAssessment, config)
    except Exception as e:
        assessment = RiskAssessment(
            adjusted_signal="neutral",
            confidence=0.0,
            max_position_weight=0.0,
            reasoning=f"Error during risk assessment: {str(e)}",
        )

    state.setdefault("data", {})["risk_assessment"] = assessment.model_dump()
    return state


def _extract_signals(analyst_signals: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and normalize signals from analyst_signals dict.

    Handles both AgentSignal objects and dict representations.

    Args:
        analyst_signals: Dict of agent_name -> signal (dict or AgentSignal)

    Returns:
        List of normalized signal dicts
    """
    signals = []

    for agent_name, signal_data in analyst_signals.items():
        if signal_data is None:
            continue

        if isinstance(signal_data, dict):
            signal_dict = signal_data
        elif hasattr(signal_data, "model_dump"):
            signal_dict = signal_data.model_dump()
        elif hasattr(signal_data, "dict"):
            signal_dict = signal_data.dict()
        else:
            continue

        if "signal" not in signal_dict or "confidence" not in signal_dict:
            continue

        signals.append(signal_dict)

    return signals


def _compute_risk_metrics(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute risk metrics from aggregated signals.

    Args:
        signals: List of signal dicts

    Returns:
        Dict of risk metrics
    """
    if not signals:
        return {
            "total_agents": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "average_confidence": 0.0,
            "signal_agreement_ratio": 0.0,
        }

    bullish_count = sum(1 for s in signals if s.get("signal") == "bullish")
    bearish_count = sum(1 for s in signals if s.get("signal") == "bearish")
    neutral_count = sum(1 for s in signals if s.get("signal") == "neutral")
    total_agents = len(signals)

    confidences = [s.get("confidence", 0.0) for s in signals]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    max_count = max(bullish_count, bearish_count, neutral_count)
    signal_agreement_ratio = max_count / total_agents if total_agents > 0 else 0.0

    return {
        "total_agents": total_agents,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "average_confidence": average_confidence,
        "signal_agreement_ratio": signal_agreement_ratio,
    }


def _build_risk_prompt(signals: list[dict[str, Any]], risk_metrics: dict[str, Any]) -> str:
    """Build LLM prompt for risk assessment.

    Args:
        signals: List of signal dicts
        risk_metrics: Computed risk metrics

    Returns:
        Formatted prompt string
    """
    signal_summary_lines = []
    for signal in signals:
        agent_name = signal.get("agent_name", "Unknown")
        signal_direction = signal.get("signal", "neutral")
        confidence = signal.get("confidence", 0.0)
        reasoning = signal.get("reasoning", "")

        signal_summary_lines.append(
            f"- {agent_name}: {signal_direction} (confidence: {confidence:.2f}) - {reasoning}"
        )

    signal_summary = "\n".join(signal_summary_lines)

    prompt = f"""你是专业的A股风险管理专家。请基于以下分析师信号和风险指标，提供风险调整后的交易建议。

**分析师信号汇总** ({risk_metrics['total_agents']} 个代理):
{signal_summary}

**风险指标**:
- 看涨信号数量: {risk_metrics['bullish_count']}
- 看跌信号数量: {risk_metrics['bearish_count']}
- 中性信号数量: {risk_metrics['neutral_count']}
- 平均置信度: {risk_metrics['average_confidence']:.2f}
- 信号一致性比例: {risk_metrics['signal_agreement_ratio']:.2f} (多数信号占比)

**风险评估原则** (A股市场):
1. **高一致性** (一致性比例 > 0.7): 可以考虑较高仓位，但需结合置信度
2. **中等一致性** (0.4 < 一致性比例 <= 0.7): 谨慎建仓，控制仓位
3. **低一致性** (一致性比例 <= 0.4): 观望为主，最小仓位或空仓
4. **混合信号**: 当看涨和看跌信号数量接近时，倾向于中性，降低风险敞口
5. **置信度加权**: 高置信度信号应获得更大权重

**输出要求**:
1. adjusted_signal: 调整后的交易信号 (bullish/bearish/neutral)
2. confidence: 调整后的置信度 (0.0-1.0)，考虑信号一致性和平均置信度
3. max_position_weight: 建议的最大仓位权重 (0.0-1.0)，基于风险水平
   - 高一致性 + 高置信度: 0.5-1.0
   - 中等一致性: 0.2-0.5
   - 低一致性或混合信号: 0.0-0.2
4. reasoning: 风险调整的详细理由，说明如何从原始信号得出最终建议

请综合考虑所有分析师的观点和A股市场特性，给出风险调整后的建议。"""

    return prompt
