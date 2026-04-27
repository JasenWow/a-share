"""LangGraph workflow for hedge fund multi-agent analysis."""
from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from big_a.models.hedge_fund.types import HedgeFundState

from big_a.models.hedge_fund.agents.technicals import technicals_agent
from big_a.models.hedge_fund.agents.valuation import valuation_agent
from big_a.models.hedge_fund.agents.sentiment import sentiment_agent
from big_a.models.hedge_fund.agents.warren_buffett import warren_buffett_agent
from big_a.models.hedge_fund.agents.charlie_munger import charlie_munger_agent
from big_a.models.hedge_fund.agents.peter_lynch import peter_lynch_agent
from big_a.models.hedge_fund.agents.ben_graham import ben_graham_agent
from big_a.models.hedge_fund.agents.phil_fisher import phil_fisher_agent
from big_a.models.hedge_fund.agents.bill_ackman import bill_ackman_agent
from big_a.models.hedge_fund.agents.michael_burry import michael_burry_agent
from big_a.models.hedge_fund.agents.nassim_taleb import nassim_taleb_agent
from big_a.models.hedge_fund.agents.cathie_wood import cathie_wood_agent
from big_a.models.hedge_fund.agents.aswath_damodaran import aswath_damodaran_agent
from big_a.models.hedge_fund.agents.risk_manager import risk_manager_agent
from big_a.models.hedge_fund.agents.portfolio_manager import portfolio_manager_agent

SLIM_AGENTS = [
    technicals_agent,
    valuation_agent,
    warren_buffett_agent,
]

FULL_AGENTS = [
    technicals_agent,
    valuation_agent,
    sentiment_agent,
    warren_buffett_agent,
    charlie_munger_agent,
    peter_lynch_agent,
    ben_graham_agent,
    phil_fisher_agent,
    bill_ackman_agent,
    michael_burry_agent,
    nassim_taleb_agent,
    cathie_wood_agent,
    aswath_damodaran_agent,
]


def _start_node(state: HedgeFundState) -> HedgeFundState:
    return state


def create_workflow(
    agent_list: list[Any] | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "slim",
) -> Any:
    """Create the LangGraph workflow with fan-out/fan-in pattern.

    Structure: start → [agent1, agent2, ...] (parallel) → risk_manager → portfolio_manager → END

    Parameters
    ----------
    agent_list : list of agent functions or None
        List of agent functions to include as parallel nodes.
        If None, uses mode to determine agents.
    config : dict or None
        Configuration dict (unused in workflow creation, passed to agents at runtime).
    mode : str
        "slim" for 3 analysts (default) or "full" for all 13.

    Returns
    -------
    CompiledGraph
        Compiled LangGraph workflow ready for execution.
    """
    if agent_list is None:
        cfg = config or {}
        effective_mode = cfg.get("mode", mode)
        agent_list = SLIM_AGENTS if effective_mode == "slim" else FULL_AGENTS

    workflow = StateGraph(HedgeFundState)

    workflow.add_node("start", _start_node)
    workflow.add_node("risk_manager", risk_manager_agent)
    workflow.add_node("portfolio_manager", portfolio_manager_agent)

    for agent_func in agent_list:
        agent_name = agent_func.__name__
        workflow.add_node(agent_name, agent_func)

    for agent_func in agent_list:
        agent_name = agent_func.__name__
        workflow.add_edge("start", agent_name)

    for agent_func in agent_list:
        agent_name = agent_func.__name__
        workflow.add_edge(agent_name, "risk_manager")

    workflow.add_edge("risk_manager", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start")

    return workflow.compile()


def run_workflow(
    graph: Any,
    ticker: str,
    start_date: str,
    end_date: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the compiled workflow graph.

    CRITICAL: Must populate BOTH state["data"] and state["metadata"] with ticker/dates
    because different agents read from different fields:
    - technicals_agent reads from state["data"]["ticker"]
    - valuation_agent reads from state["data"]["ticker"]
    - investor agents read from state["metadata"]["ticker"] and state["metadata"]["date"]
    - sentiment_agent reads from state["metadata"]["ticker"]

    Parameters
    ----------
    graph : CompiledGraph
        Compiled LangGraph from create_workflow().
    ticker : str
        Stock ticker to analyze.
    start_date, end_date : str
        Date range.
    config : dict or None
        Optional configuration passed to agents via metadata.

    Returns
    -------
    dict
        Final state from the workflow with portfolio_decision.
    """
    initial_state: HedgeFundState = {
        "messages": [],
        "data": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
        },
        "metadata": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "date": end_date,
            "config": config or {},
        },
    }
    result = graph.invoke(initial_state)
    return result
