"""Portfolio manager agent for final investment decisions."""
from __future__ import annotations

from typing import Any

from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.types import HedgeFundState, PortfolioDecision


def portfolio_manager_agent(state: HedgeFundState) -> HedgeFundState:
    """Make final investment decision based on risk assessment and analyst signals.

    Reads risk_assessment and analyst_signals from state, uses LLM to produce
    a PortfolioDecision with action, score, and reasoning. The score is in
    [-1, 1] range for Qlib backtest compatibility.

    Args:
        state: HedgeFundState containing risk_assessment and analyst_signals

    Returns:
        Updated HedgeFundState with portfolio_decision in data field
    """
    data = state.get("data", {})
    risk_assessment_data = data.get("risk_assessment")
    analyst_signals = data.get("analyst_signals", {})
    config = state.get("metadata", {}).get("config")

    if not risk_assessment_data:
        neutral_decision = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning="No risk assessment available for portfolio decision",
        )
        state.setdefault("data", {})["portfolio_decision"] = neutral_decision.model_dump()
        return state

    risk_assessment = _extract_risk_assessment(risk_assessment_data)

    if not risk_assessment:
        neutral_decision = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning="Invalid risk assessment data for portfolio decision",
        )
        state.setdefault("data", {})["portfolio_decision"] = neutral_decision.model_dump()
        return state

    signals = _extract_signals(analyst_signals)
    prompt = _build_decision_prompt(risk_assessment, signals)

    try:
        decision = call_llm(prompt, PortfolioDecision, config)
    except Exception as e:
        decision = PortfolioDecision(
            action="hold",
            score=0.0,
            reasoning=f"Error during portfolio decision: {str(e)}",
        )

    state.setdefault("data", {})["portfolio_decision"] = decision.model_dump()
    return state


def _extract_risk_assessment(risk_assessment_data: Any) -> dict[str, Any] | None:
    """Extract and normalize risk assessment from state.

    Handles both RiskAssessment objects and dict representations.

    Args:
        risk_assessment_data: RiskAssessment object or dict

    Returns:
        Normalized risk assessment dict or None if invalid
    """
    if isinstance(risk_assessment_data, dict):
        risk_dict = risk_assessment_data
    elif hasattr(risk_assessment_data, "model_dump"):
        risk_dict = risk_assessment_data.model_dump()
    elif hasattr(risk_assessment_data, "dict"):
        risk_dict = risk_assessment_data.dict()
    else:
        return None

    required_fields = ["adjusted_signal", "confidence", "max_position_weight", "reasoning"]
    if not all(field in risk_dict for field in required_fields):
        return None

    return risk_dict


def _extract_signals(analyst_signals: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and normalize analyst signals for context.

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


def _build_decision_prompt(risk_assessment: dict[str, Any], signals: list[dict[str, Any]]) -> str:
    """Build LLM prompt for portfolio decision.

    Args:
        risk_assessment: Normalized risk assessment dict
        signals: List of normalized signal dicts

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

    adjusted_signal = risk_assessment.get("adjusted_signal", "neutral")
    confidence = risk_assessment.get("confidence", 0.0)
    max_position_weight = risk_assessment.get("max_position_weight", 0.0)
    risk_reasoning = risk_assessment.get("reasoning", "")

    prompt = f"""你是专业的A股投资组合经理。请基于风险管理和分析师信号，做出最终投资决策。

**风险评估摘要**:
- 调整后信号: {adjusted_signal}
- 风险置信度: {confidence:.2f}
- 最大仓位权重: {max_position_weight:.2f}
- 风险评估理由: {risk_reasoning}

**分析师信号汇总** ({len(signals)} 个代理):
{signal_summary if signal_summary else "无分析师信号"}

**投资决策原则** (A股市场):
1. **风险优先**: 风险管理器的建议是核心考量因素
2. **仓位控制**: score 的绝对值不应超过 max_position_weight
3. **信号一致性**: 当分析师信号与风险评估一致时，增强决策信心
4. **量化评分**: 输出 score 用于量化回测，范围 [-1, 1]
   - 看涨/高置信度: score 接近 1.0 (但不超过 max_position_weight)
   - 看跌/高置信度: score 接近 -1.0 (但不超过 max_position_weight)
   - 中性: score 接近 0.0
5. **动态调整**: 
   - bullish → 正分数 (建议 0.5-1.0，受 max_position_weight 限制)
   - bearish → 负分数 (建议 -0.5 到 -1.0，受 max_position_weight 限制)
   - neutral → 接近 0 (建议 -0.2 到 0.2)

**输出要求**:
1. action: 交易动作 (buy/sell/hold)
   - buy: 当风险评估为 bullish 且分析师支持看涨时
   - sell: 当风险评估为 bearish 且分析师支持看跌时
   - hold: 当风险评估为 neutral 或信号不一致时
2. score: 量化评分 [-1, 1]，用于 Qlib 回测
   - 必须是浮点数，不能是 NaN 或 inf
   - 正数表示多头，负数表示空头，0 表示中性
   - 绝对值受 max_position_weight 限制: abs(score) <= max_position_weight
3. reasoning: 决策的详细理由，说明如何综合风险管理和分析师信号

请综合考虑风险评估和所有分析师信号，给出最终投资决策。"""

    return prompt
