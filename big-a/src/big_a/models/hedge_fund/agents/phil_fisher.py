from __future__ import annotations

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_market_data


def phil_fisher_agent(state: HedgeFundState) -> HedgeFundState:
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    date = metadata.get("date")
    config = metadata.get("config", {})

    if not ticker or not date:
        signal = AgentSignal(
            agent_name="phil_fisher",
            signal="neutral",
            confidence=0.0,
            reasoning="缺少必要参数：ticker 或 date"
        )
    else:
        market_data = get_market_data([ticker], date)
        data = market_data.get(ticker, {})

        prompt = f"""你是菲利普·费雪，成长调研投资大师。请基于以下A股市场数据进行分析，给出你的投资建议。

股票代码：{ticker}
分析日期：{date}
当前价格：{data.get('close', 'N/A')}
成交量：{data.get('volume', 'N/A')}
涨跌幅：{data.get('change', 'N/A')}

请从以下角度分析（这需要深入的调研和产业链了解）：
1. 公司是否有长期增长潜力的产品或服务？
2. 管理层是否有远见和能力实现长期目标？
3. 公司是否在研发方面有持续投入？
4. 销售增长是否健康且可持续？
5. 与竞争对手相比，该公司是否有持久的竞争优势？
6. 通过产业链调研，上下游客户和供应商如何评价该公司？

请记住：买入后长期持有优秀成长股，不要频繁交易。

请给出你的判断：bullish（看涨）、bearish（看跌）或 neutral（中性），并说明理由。
"""

        signal = call_llm(prompt, AgentSignal, config)

    if "data" not in state:
        state["data"] = {}
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["phil_fisher_agent"] = signal.model_dump()

    return state
