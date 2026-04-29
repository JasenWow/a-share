"""Tests for simulation Prefect flow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_simulation_pipeline_importable():
    from big_a.scheduler.flows import simulation_pipeline
    assert simulation_pipeline.name == "simulation_trading_pipeline"


def test_daily_pipeline_still_works():
    from big_a.scheduler.flows import daily_pipeline
    assert daily_pipeline is not None


def test_flows_are_independent():
    from big_a.scheduler.flows import daily_pipeline, simulation_pipeline
    assert daily_pipeline.name != simulation_pipeline.name