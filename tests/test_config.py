from pathlib import Path

import pytest

from src.config import load_config


def test_experiment_config_respects_horizon_embargo() -> None:
    config = load_config(Path("configs/experiment.yaml"))
    assert config["evaluation"]["embargo_days"] >= config["forecast"]["horizon_days"]


def test_config_rejects_short_embargo(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(
        "experiment: {}\ndata: {}\nforecast: {horizon_days: 5}\n"
        "features: {}\nevaluation: {embargo_days: 1}\nmodels: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="embargo"):
        load_config(config_file)
