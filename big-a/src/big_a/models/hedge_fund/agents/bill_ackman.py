from __future__ import annotations

from typing import cast

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_market_data


def bill_ackman_agent(state: HedgeFundState) -> HedgeFundState:
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    date = metadata.get("date")
    config = metadata.get("config", {})

    if not ticker or not date:
        signal = AgentSignal(
            agent_name="bill_ackman",
            signal="neutral",
            confidence=0.0,
            reasoning="缺少必要参数：ticker 或 date"
        )
    else:
        market_data = get_market_data([ticker], date)
        data = cast(dict[str, object], market_data.get(ticker, {}))

        prompt = f"""你是比尔·阿克曼，激进投资大师。请基于以下A股市场数据进行分析，给出你的投资建议。

股票代码：{ticker}
分析日期：{date}
当前价格：{data.get('close', 'N/A')}
成交量：{data.get('volume', 'N/A')}
涨跌幅：{data.get('change', 'N/A')}

请从以下角度分析：
1. 公司是否具有强大的品牌护城河和消费者垄断地位？
2. 管理层是否有改进空间，是否需要推动变革？
3. 当前价格是否反映真实价值，是否适合集中持仓？
4. 公司业务是否具备长期竞争优势？
5. 在A股市场中，该行业是否具有集中投资的价值？

请给出你的判断：bullish（看涨）、bearish（看跌）或 neutral（中性），并说明理由。
"""

        signal = call_llm(prompt, AgentSignal, config)

    if "data" not in state:
        state["data"] = {}
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["bill_ackman_agent"] = signal.model_dump()

    return state
