from __future__ import annotations

from typing import cast

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_market_data


def cathie_wood_agent(state: HedgeFundState) -> HedgeFundState:
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    date = metadata.get("date")
    config = metadata.get("config", {})

    if not ticker or not date:
        signal = AgentSignal(
            agent_name="cathie_wood",
            signal="neutral",
            confidence=0.0,
            reasoning="缺少必要参数：ticker 或 date"
        )
    else:
        market_data = get_market_data([ticker], date)
        data = cast(dict[str, object], market_data.get(ticker, {}))

        prompt = f"""你是凯瑟琳·伍德（Cathie Wood），颠覆性创新投资先驱。请基于以下A股市场数据进行分析，给出你的投资建议。

股票代码：{ticker}
分析日期：{date}
当前价格：{data.get('close', 'N/A')}
成交量：{data.get('volume', 'N/A')}
涨跌幅：{data.get('change', 'N/A')}

请从以下角度分析：
1. 该公司是否在推动颠覆性技术创新，能否改变行业格局？
2. 在中国科技创新、科创板或新能源等前沿领域，该公司的地位如何？
3. 从5年或10年展望，公司是否具备指数级增长潜力？
4. 技术创新速度和市场渗透率是否领先？
5. 在A股市场中，该公司的长期成长空间有多大？

请给出你的判断：bullish（看涨）、bearish（看跌）或 neutral（中性），并说明理由。
"""

        signal = call_llm(prompt, AgentSignal, config)

    if "data" not in state:
        state["data"] = {}
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["cathie_wood_agent"] = signal.model_dump()

    return state
