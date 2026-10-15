"""Classify existing statistical forecasts by training-fitted market regimes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_statistical_models import (
    TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, STEP_SIZE, FINAL_TEST_SIZE,
    EMBARGO, HORIZON,
)
from src.data.loader import load_industry_returns
from src.research_config import CONFIG
from src.data.portfolio import build_forecasting_dataset
from src.features.rolling_features import build_regime_features
from src.evaluation.splits import generate_walk_forward_splits
from src.evaluation.regime_analysis import (
    assign_walk_forward_regimes, summarize_regime_metrics,
)


def main():
    path = PROJECT_ROOT / "data/processed/walk_forward_statistical_models_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Run scripts/build_dataset.py and scripts/compare_statistical_models.py first."
        )
    predictions = pd.read_csv(path, index_col=0, parse_dates=True)
    returns = load_industry_returns(
        PROJECT_ROOT / CONFIG["data"]["raw_path"]
    )
    dataset = build_forecasting_dataset(returns, horizon=HORIZON)
    # Compute trailing features on full history, then align without shifting folds.
    features = build_regime_features(returns, window=21).reindex(dataset.index)
    splits = generate_walk_forward_splits(
        len(dataset), TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, STEP_SIZE,
        FINAL_TEST_SIZE, EMBARGO,
    )
    expected_dates = dataset.index[np.concatenate([
        np.arange(fold.test.start, fold.test.stop) for fold in splits
    ])]
    if not predictions.index.equals(expected_dates):
        raise ValueError("Saved forecasts do not match the configured evaluation dates.")
    if not np.allclose(predictions["actual"], dataset.loc[expected_dates, "target_volatility_5d"],
                       rtol=1e-10, atol=1e-12):
        raise ValueError("Saved targets differ from current source data; regenerate forecasts.")
    assignments, centers = assign_walk_forward_regimes(features, splits)
    metrics = summarize_regime_metrics(predictions, assignments)
    output = PROJECT_ROOT / "reports/tables"
    output.mkdir(parents=True, exist_ok=True)
    predictions.join(assignments).to_csv(output / "regime_forecasts.csv", index_label="date")
    centers.to_csv(output / "regime_centers_by_fold.csv", index=False)
    metrics.to_csv(output / "regime_model_metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print("Regimes are relative training-volatility ranks; this is descriptive analysis.")


if __name__ == "__main__":
    main()
