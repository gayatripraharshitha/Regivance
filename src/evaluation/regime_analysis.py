"""Training-only regime assignment and descriptive forecast breakdowns."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_volatility_metrics
from src.features.regimes import MarketRegimeDetector


def assign_walk_forward_regimes(features, splits, volatility_column="portfolio_volatility_21d"):
    """Use slices on the original modeling index; never drop rows before slicing.

    Rank labels by training-center volatility to make fold summaries comparable.
    These are relative ranks, not fixed economic regimes across time.
    """
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("Features require a DatetimeIndex.")
    if features.index.has_duplicates or not features.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique and chronological.")
    if volatility_column not in features:
        raise ValueError("Missing volatility feature.")
    assignments, centers = [], []
    for split in splits:
        for window in (split.train, split.test):
            if (window.start is None or window.stop is None
                    or not 0 <= window.start < window.stop <= len(features)
                    or window.step not in (None, 1)):
                raise ValueError("Invalid fold bounds.")
        if split.train.stop > split.test.start:
            raise ValueError("Training must precede testing.")
        training = features.iloc[split.train].dropna()
        testing = features.iloc[split.test]
        detector = MarketRegimeDetector().fit(training)
        center = detector.cluster_centers()
        ordered = center[volatility_column].sort_values(kind="stable").index
        mapping = dict(zip(ordered, ("lower", "middle", "higher")))
        labels = detector.predict(testing)
        assignments.append(pd.DataFrame({
            "fold": split.fold,
            "cluster_id": labels,
            "regime": labels.map(mapping),
        }, index=testing.index))
        center["regime"] = center.index.map(mapping)
        center["fold"] = split.fold
        center["training_start"] = training.index[0]
        center["training_end"] = training.index[-1]
        center["n_training"] = len(training)
        centers.append(center.rename_axis("cluster_id").reset_index())
    if not assignments:
        raise ValueError("At least one fold is required.")
    assigned = pd.concat(assignments)
    if assigned.index.has_duplicates:
        raise ValueError("Test folds overlap.")
    return assigned, pd.concat(centers, ignore_index=True)


def summarize_regime_metrics(predictions, assignments):
    """Require identical evaluation dates; never silently discard forecasts."""
    if predictions.index.has_duplicates or not predictions.index.equals(assignments.index):
        raise ValueError("Predictions and regimes must have identical unique dates.")
    if "actual" not in predictions or predictions.shape[1] < 2:
        raise ValueError("Expected actual and at least one model forecast.")
    if predictions.columns.has_duplicates or assignments["regime"].isna().any():
        raise ValueError("Invalid prediction columns or missing regimes.")
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must be finite.")
    joined = predictions.join(assignments)
    records = []
    for regime, rows in joined.groupby("regime"):
        for model in predictions.columns.drop("actual"):
            records.append({
                "regime": regime, "model": model, "n_test": len(rows),
                **calculate_volatility_metrics(rows["actual"], rows[model]),
            })
    return pd.DataFrame(records)
