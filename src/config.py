"""Experiment configuration loading and validation."""

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {"experiment", "data", "forecast", "features", "evaluation", "models"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config and ensure its top-level contract."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    missing = REQUIRED_SECTIONS - config.keys()
    if missing:
        raise ValueError(f"Configuration missing sections: {sorted(missing)}")
    if config["forecast"]["horizon_days"] <= 0:
        raise ValueError("forecast.horizon_days must be positive.")
    if config["evaluation"]["embargo_days"] < config["forecast"]["horizon_days"]:
        raise ValueError("evaluation.embargo_days must be at least forecast.horizon_days.")
    return config
