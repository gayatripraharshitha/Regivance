import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import WalkForwardSplit
from src.evaluation.regime_analysis import assign_walk_forward_regimes, summarize_regime_metrics


def fixture():
    rng = np.random.default_rng(42)
    features = pd.DataFrame(rng.uniform(0.01, 0.1, (50, 3)),
        columns=["portfolio_volatility_21d", "correlation", "dispersion"],
        index=pd.date_range("2020-01-01", periods=50))
    folds = [WalkForwardSplit(1, slice(0, 20), slice(25, 30), slice(35, 40)),
             WalkForwardSplit(2, slice(5, 25), slice(30, 35), slice(40, 45))]
    return features, folds


def test_training_only_and_ranked_centers():
    features, folds = fixture()
    assigned, centers = assign_walk_forward_regimes(features, folds)
    assert assigned.index.equals(features.index[35:45])
    for _, group in centers.groupby("fold"):
        assert group.sort_values("portfolio_volatility_21d")["regime"].tolist() == ["lower", "middle", "higher"]
    changed = features.copy()
    changed.iloc[25:] *= 10
    _, changed_centers = assign_walk_forward_regimes(changed, folds)
    pd.testing.assert_frame_equal(centers, changed_centers)


def test_future_changes_preserve_earlier_labels():
    features, folds = fixture()
    original, _ = assign_walk_forward_regimes(features, folds)
    features.iloc[43:] *= 20
    changed, _ = assign_walk_forward_regimes(features, folds)
    pd.testing.assert_frame_equal(original.iloc[:8], changed.iloc[:8])


def test_warmup_does_not_shift_fold_dates():
    features, folds = fixture()
    features.iloc[:2] = np.nan
    assigned, centers = assign_walk_forward_regimes(features, folds)
    assert assigned.index.equals(features.index[35:45])
    assert centers.iloc[0]["n_training"] == 18


def test_missing_test_features_fail():
    features, folds = fixture()
    features.iloc[36] = np.nan
    with pytest.raises(ValueError, match="finite"):
        assign_walk_forward_regimes(features, folds)


def test_overlap_is_rejected():
    features, folds = fixture()
    with pytest.raises(ValueError, match="overlap"):
        assign_walk_forward_regimes(features, [folds[0], folds[0]])


def test_metrics_and_date_mismatch():
    features, folds = fixture()
    assigned, _ = assign_walk_forward_regimes(features, folds)
    predictions = pd.DataFrame({"actual": 0.02, "perfect": 0.02}, index=assigned.index)
    result = summarize_regime_metrics(predictions, assigned)
    assert result["n_test"].sum() == 10
    assert (result[["mae", "rmse", "qlike"]] == 0).all().all()
    with pytest.raises(ValueError, match="identical"):
        summarize_regime_metrics(predictions.iloc[1:], assigned)
