from __future__ import annotations

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_market_data


def peter_lynch_agent(state: HedgeFundState) -> HedgeFundState:
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    date = metadata.get("date")
    config = metadata.get("config", {})

    if not ticker or not date:
        signal = AgentSignal(
            agent_name="peter_lynch",
            signal="neutral",
            confidence=0.0,
            reasoning="缺少必要参数：ticker 或 date"
        )
    else:
        market_data = get_market_data([ticker], date)
        data = market_data.get(ticker, {})

        prompt = f"""你是彼得·林奇，成长投资大师。请基于以下A股市场数据进行分析，给出你的投资建议。

股票代码：{ticker}
分析日期：{date}
当前价格：{data.get('close', 'N/A')}
成交量：{data.get('volume', 'N/A')}
涨跌幅：{data.get('change', 'N/A')}

请从以下角度分析：
1. 这是否是你熟悉的行业或日常生活中能接触到的公司？
2. 盈利增长率是否高于估值水平（PEG比率）？
3. 这是否是一只潜在的十倍股？
4. 股票属于哪类：稳健增长型、快速成长型、周期型还是资产隐蔽型？
5. 考虑A股市场的涨跌停板和T+1交易特点，当前是否是买入时机？

请给出你的判断：bullish（看涨）、bearish（看跌）或 neutral（中性），并说明理由。
"""

        signal = call_llm(prompt, AgentSignal, config)

    if "data" not in state:
        state["data"] = {}
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["peter_lynch_agent"] = signal.model_dump()

    return state
