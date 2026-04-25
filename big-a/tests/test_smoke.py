import importlib
import sys
from pathlib import Path
import pytest

# Ensure src/ package is on sys.path so that local packages under src/big_a can be imported
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_subpackages_import():
    # Ensure core package and its subpackages import without execution
    for mod in (
        "big_a",
        "big_a.data",
        "big_a.factors",
        "big_a.models",
        "big_a.strategy",
        "big_a.backtest",
        "big_a.workflow",
    ):
        __import__(mod)


def test_config_imports():
    # Ensure config module can be imported if present; skip if not available
    try:
        importlib.import_module("big_a.config")
    except ModuleNotFoundError:
        pytest.skip("big_a.config module not found; skipping")


def test_pyyaml_available():
    # PyYAML should be installed and importable
    import yaml  # noqa: F401


def test_qlib_installed_version():
    # Qlib should be installed and at least version 0.9.0
    import qlib

    v = getattr(qlib, "__version__", "0.0.0")
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    assert tuple(parts[:3]) >= (0, 9, 0), f"qlib version {v} is less than 0.9.0"


def test_lightgbm_installed():
    import importlib

    m = importlib.import_module("lightgbm")
    assert m is not None


def test_torch_installed():
    import torch  # noqa: F401
