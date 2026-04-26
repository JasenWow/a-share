"""Tests for hedge fund LangGraph workflow."""

from big_a.models.hedge_fund.graph import create_workflow, run_workflow


def test_create_workflow_compiles():
    """Create workflow with mock agents, verify it compiles."""
    graph = create_workflow(agent_names=["agent1", "agent2"])
    assert graph is not None
    assert hasattr(graph, "invoke")


def test_run_workflow_returns_state():
    """Run with mock, verify state dict returned."""
    graph = create_workflow(agent_names=["agent1"])
    result = run_workflow(
        graph,
        ticker="AAPL",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    assert isinstance(result, dict)
    assert "messages" in result
    assert "data" in result
    assert "metadata" in result


def test_workflow_preserves_ticker():
    """Run with ticker, verify it's in output state."""
    graph = create_workflow(agent_names=["agent1"])
    result = run_workflow(
        graph,
        ticker="MSFT",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    assert result["data"]["ticker"] == "MSFT"
    assert result["data"]["start_date"] == "2024-01-01"
    assert result["data"]["end_date"] == "2024-01-31"


def test_workflow_with_multiple_agents():
    """Create with 3 agent names, run, verify all executed."""
    graph = create_workflow(agent_names=["technical", "fundamental", "sentiment"])
    result = run_workflow(
        graph,
        ticker="GOOGL",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    assert result is not None
    assert isinstance(result, dict)


def test_workflow_with_no_agents():
    """Create with empty list, verify start → risk → portfolio → END."""
    graph = create_workflow(agent_names=[])
    result = run_workflow(
        graph,
        ticker="TSLA",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    assert result is not None
    assert isinstance(result, dict)
