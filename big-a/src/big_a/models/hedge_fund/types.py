"""Type definitions for the hedge fund multi-agent system."""
from __future__ import annotations
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import operator


def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two dicts (for LangGraph state annotation)."""
    return {**a, **b}


class HedgeFundState(TypedDict, total=False):
    """LangGraph state for the hedge fund workflow."""
    messages: Annotated[list[Any], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]


class AgentSignal(BaseModel):
    """Signal output from a single analysis agent."""
    agent_name: str = Field(description="Name of the agent producing this signal")
    signal: Literal["bullish", "bearish", "neutral"] = Field(description="Trading signal direction")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level 0-1")
    reasoning: str = Field(description="Explanation for the signal")


class RiskAssessment(BaseModel):
    """Risk assessment from the risk manager agent."""
    adjusted_signal: Literal["bullish", "bearish", "neutral"] = Field(description="Signal after risk adjustment")
    confidence: float = Field(ge=0.0, le=1.0)
    max_position_weight: float = Field(ge=0.0, le=1.0, description="Max weight for this position")
    reasoning: str = Field(description="Risk assessment reasoning")


class PortfolioDecision(BaseModel):
    """Final portfolio decision from the portfolio manager."""
    action: Literal["buy", "sell", "hold"] = Field(description="Trading action")
    score: float = Field(ge=-1.0, le=1.0, description="Score compatible with Qlib backtest [-1, 1]")
    reasoning: str = Field(description="Decision reasoning")


class MarketData(BaseModel):
    """Market data structure for agent consumption."""
    instrument: str = Field(description="Stock instrument code (e.g. SH600000)")
    date: str = Field(description="Trading date (YYYY-MM-DD)")
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
