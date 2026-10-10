from pathlib import Path

import pytest

from src.config import load_config, validate_config


def test_experiment_config_respects_horizon_embargo() -> None:
    config = load_config(Path("configs/experiment.yaml"))
    assert config["evaluation"]["embargo_days"] >= config["forecast"]["horizon_days"]


def test_config_rejects_short_embargo() -> None:
    config = {
        "experiment": {}, "data": {}, "forecast": {"horizon_days": 5},
        "features": {}, "evaluation": {"embargo_days": 1}, "models": {},
    }
    with pytest.raises(ValueError, match="embargo"):
        validate_config(config)
