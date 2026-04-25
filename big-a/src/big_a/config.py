"""Configuration loading and merging utility."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load and deep-merge one or more YAML config files."""
    merged: dict[str, Any] = {}
    for p in paths:
        path = PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)
        with open(path) as f:
            data = yaml.safe_load(f)
        if data:
            merged = _deep_merge(merged, data)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Validate that required config sections exist."""
    if "qlib_init" not in config:
        raise ValueError("Config missing 'qlib_init' section")
