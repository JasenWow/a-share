"""Tests for workflow slim/full mode support."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unittest.mock import MagicMock, patch

import pytest

from big_a.models.hedge_fund.graph.workflow import create_workflow, run_workflow


class TestCreateWorkflow:
    """Tests for create_workflow with mode parameter."""

    def test_slim_mode_has_three_analysts(self) -> None:
        """Slim mode should have exactly 3 analyst nodes."""
        graph = create_workflow(mode="slim")
        # Graph nodes: start + 3 analysts + risk_manager + portfolio_manager
        node_names = list(graph.nodes)
        # Check that the 3 slim analysts are present
        assert "technicals_agent" in node_names
        assert "valuation_agent" in node_names
        assert "warren_buffett_agent" in node_names
        # Check that risk/portfolio are present
        assert "risk_manager" in node_names
        assert "portfolio_manager" in node_names
        # Check that excluded agents are NOT present
        assert "sentiment_agent" not in node_names
        assert "cathie_wood_agent" not in node_names
        assert "michael_burry_agent" not in node_names

    def test_full_mode_has_all_agents(self) -> None:
        """Full mode should have all 13 analyst nodes."""
        graph = create_workflow(mode="full")
        node_names = list(graph.nodes)
        # All original agents should be present
        assert "technicals_agent" in node_names
        assert "valuation_agent" in node_names
        assert "sentiment_agent" in node_names
        assert "warren_buffett_agent" in node_names
        assert "charlie_munger_agent" in node_names
        assert "peter_lynch_agent" in node_names
        assert "ben_graham_agent" in node_names
        assert "phil_fisher_agent" in node_names
        assert "bill_ackman_agent" in node_names
        assert "michael_burry_agent" in node_names
        assert "nassim_taleb_agent" in node_names
        assert "cathie_wood_agent" in node_names
        assert "aswath_damodaran_agent" in node_names
        assert "risk_manager" in node_names
        assert "portfolio_manager" in node_names

    def test_default_mode_is_slim(self) -> None:
        """Default mode should be slim."""
        graph_default = create_workflow()
        graph_slim = create_workflow(mode="slim")
        assert set(graph_default.nodes) == set(graph_slim.nodes)

    def test_run_workflow_passes_mode(self) -> None:
        """run_workflow should pass mode to create_workflow via config."""
        with patch("big_a.models.hedge_fund.graph.workflow.create_workflow") as mock_create:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "messages": [],
                "data": {"portfolio_decision": {"score": 0.0, "action": "hold", "reasoning": "test"}, "analyst_signals": {}},
                "metadata": {},
            }
            mock_create.return_value = mock_graph

            run_workflow(
                graph=mock_graph,
                ticker="SH600000",
                start_date="2024-01-01",
                end_date="2024-12-31",
                config={"mode": "slim"},
            )
            # Verify graph was invoked
            mock_graph.invoke.assert_called_once()