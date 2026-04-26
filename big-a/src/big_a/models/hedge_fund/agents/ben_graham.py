from __future__ import annotations

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_market_data


def ben_graham_agent(state: HedgeFundState) -> HedgeFundState:
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    date = metadata.get("date")
    config = metadata.get("config", {})

    if not ticker or not date:
        signal = AgentSignal(
            agent_name="ben_graham",
            signal="neutral",
            confidence=0.0,
            reasoning="缺少必要参数：ticker 或 date"
        )
    else:
        market_data = get_market_data([ticker], date)
        data = market_data.get(ticker, {})

        prompt = f"""你是本杰明·格雷厄姆，深度价值投资之父。请基于以下A股市场数据进行分析，给出你的投资建议。

股票代码：{ticker}
分析日期：{date}
当前价格：{data.get('close', 'N/A')}
成交量：{data.get('volume', 'N/A')}
涨跌幅：{data.get('change', 'N/A')}

请从以下角度分析：
1. 价格是否远低于内在价值？（安全边际至少为三分之一）
2. 估值指标（如市盈率、市净率）是否处于历史低位？
3. 公司财务状况是否稳健，有足够的流动性？
4. 是否有持续的分红记录？
5. 考虑A股市场波动性较大的特点，当前价格是否足够便宜以提供安全边际？
6. 记住"市场先生"理论：利用市场波动，而不是被市场情绪左右

请给出你的判断：bullish（看涨）、bearish（看跌）或 neutral（中性），并说明理由。
"""

        signal = call_llm(prompt, AgentSignal, config)

    if "data" not in state:
        state["data"] = {}
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["ben_graham_agent"] = signal.model_dump()

    return state
