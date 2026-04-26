from __future__ import annotations

from typing import cast

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_market_data


def aswath_damodaran_agent(state: HedgeFundState) -> HedgeFundState:
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    date = metadata.get("date")
    config = metadata.get("config", {})

    if not ticker or not date:
        signal = AgentSignal(
            agent_name="aswath_damodaran",
            signal="neutral",
            confidence=0.0,
            reasoning="缺少必要参数：ticker 或 date"
        )
    else:
        market_data = get_market_data([ticker], date)
        data = cast(dict[str, object], market_data.get(ticker, {}))

        prompt = f"""你是阿斯瓦特·达莫达兰（Aswath Damodaran），估值大师。请基于以下A股市场数据进行分析，给出你的投资建议。

股票代码：{ticker}
分析日期：{date}
当前价格：{data.get('close', 'N/A')}
成交量：{data.get('volume', 'N/A')}
涨跌幅：{data.get('change', 'N/A')}

请从以下角度分析：
1. 基于当前的盈利能力和增长率假设，DCF估值是否合理？
2. 考虑到A股市场的政策影响和散户比例高特征，估值模型如何调整？
3. 公司的故事是否与财务数据匹配，是否存在过度乐观或悲观？
4. 风险溢价和折现率是否反映真实风险水平？
5. 在当前市场环境下，该股票的估值是否具备吸引力？

请给出你的判断：bullish（看涨）、bearish（看跌）或 neutral（中性），并说明理由。
"""

        signal = call_llm(prompt, AgentSignal, config)

    if "data" not in state:
        state["data"] = {}
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["aswath_damodaran_agent"] = signal.model_dump()

    return state
