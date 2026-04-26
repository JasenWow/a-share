from __future__ import annotations

from big_a.models.hedge_fund.types import AgentSignal, HedgeFundState
from big_a.models.hedge_fund.llm import call_llm
from big_a.models.hedge_fund.tools.qlib_tools import get_market_data


def charlie_munger_agent(state: HedgeFundState) -> HedgeFundState:
    metadata = state.get("metadata", {})
    ticker = metadata.get("ticker")
    date = metadata.get("date")
    config = metadata.get("config", {})

    if not ticker or not date:
        signal = AgentSignal(
            agent_name="charlie_munger",
            signal="neutral",
            confidence=0.0,
            reasoning="缺少必要参数：ticker 或 date"
        )
    else:
        market_data = get_market_data([ticker], date)
        data = market_data.get(ticker, {})

        prompt = f"""你是查理·芒格，品质投资大师。请基于以下A股市场数据进行分析，给出你的投资建议。

股票代码：{ticker}
分析日期：{date}
当前价格：{data.get('close', 'N/A')}
成交量：{data.get('volume', 'N/A')}
涨跌幅：{data.get('change', 'N/A')}

请从以下角度分析：
1. 这是否是一家值得长期持有的伟大企业？
2. 商业模式是否具有可持续的竞争优势？
3. 管理层是否诚信、有能力且有远见？
4. 价格是否公平，而非仅仅是便宜？
5. 用你的思维模型（如逆向思维、多重因素分析）评估该公司

请给出你的判断：bullish（看涨）、bearish（看跌）或 neutral（中性），并说明理由。
"""

        signal = call_llm(prompt, AgentSignal, config)

    if "data" not in state:
        state["data"] = {}
    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}
    state["data"]["analyst_signals"]["charlie_munger_agent"] = signal.model_dump()

    return state
