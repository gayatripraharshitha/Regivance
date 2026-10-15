"""Single configuration source for the reproducible research workflow."""
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_research_config(path=None):
    path = Path(path or os.environ.get("REGIVANCE_CONFIG", ROOT / "configs/research.yaml"))
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported research configuration schema.")
    ev = config["evaluation"]
    for name in ("train_size", "validation_size", "test_size", "step_size", "development_size", "horizon", "embargo"):
        value = ev[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"evaluation.{name} must be a positive integer.")
    if ev["embargo"] < ev["horizon"]:
        raise ValueError("Embargo must cover the forward horizon.")
    if ev["test_size"] != ev["step_size"] or ev["development_size"] % ev["test_size"]:
        raise ValueError("Test blocks must cover the evaluation without gaps or overlaps.")
    # Existing feature names and comparison schemas encode these choices.
    if (ev["horizon"] != 5 or config["features"]["window"] != 21
            or config["features"]["n_regimes"] != 3 or config["features"]["seed"] != 42
            or config["models"]["rolling_window"] != 21
            or config["models"]["ewma_decay"] != 0.94):
        raise ValueError("This protocol fixes horizon=5, window=21, regimes=3, seed=42 and decay=0.94.")
    for name in ("lookback", "epochs", "hidden_size", "batch_size", "min_regime_validation"):
        value = config["models"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"models.{name} must be a positive integer.")
    final = config["final_evaluation"]
    if (not isinstance(final["observations"], int) or isinstance(final["observations"], bool)
            or final["observations"] < 1 or final["observations"] % ev["test_size"]):
        raise ValueError("Final observations must be positive complete test blocks.")
    if pd.Timestamp(final["freeze_date"]) < pd.Timestamp(final["exposed_data_through"]):
        raise ValueError("Freeze date cannot precede the exposed data boundary.")
    if final["primary_metric"] != "qlike" or final["reference"] != "GJR-GARCH-Student-t":
        raise ValueError("Frozen primary comparison is QLIKE against GJR-GARCH-Student-t.")
    return config


CONFIG = load_research_config()
