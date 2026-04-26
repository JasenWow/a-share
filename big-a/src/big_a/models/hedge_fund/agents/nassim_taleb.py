from __future__ import annotations

from typing import cast

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_market_data


def nassim_taleb_agent(state: HedgeFundState) -> HedgeFundState:
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    date = metadata.get("date")
    config = metadata.get("config", {})

    if not ticker or not date:
        signal = AgentSignal(
            agent_name="nassim_taleb",
            signal="neutral",
            confidence=0.0,
            reasoning="缺少必要参数：ticker 或 date"
        )
    else:
        market_data = get_market_data([ticker], date)
        data = cast(dict[str, object], market_data.get(ticker, {}))

        prompt = f"""你是纳西姆·塔勒布，风险分析和反脆弱理论创始人。请基于以下A股市场数据进行分析，给出你的投资建议。

股票代码：{ticker}
分析日期：{date}
当前价格：{data.get('close', 'N/A')}
成交量：{data.get('volume', 'N/A')}
涨跌幅：{data.get('change', 'N/A')}

请从以下角度分析：
1. 该投资在极端市场情况下的下行风险有多大？
2. 股票是否具有反脆弱特征，能否从波动中受益？
3. 在A股涨跌停板制度下，该股票的尾部风险如何？
4. 是否具备凸性策略的潜力，即有限损失、无限收益？
5. 如何防范黑天鹅事件，构建风险对冲组合？

请给出你的判断：bullish（看涨）、bearish（看跌）或 neutral（中性），并说明理由。
"""

        signal = call_llm(prompt, AgentSignal, config)

    if "data" not in state:
        state["data"] = {}
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["nassim_taleb_agent"] = signal.model_dump()

    return state
