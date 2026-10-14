import numpy as np
import pandas as pd
import pytest

from src.features.regimes import MarketRegimeDetector


def sample_features():
    """Create three clearly separated groups for testing."""
    return pd.DataFrame(
        {
            "portfolio_volatility_21d": [
                0.004, 0.005, 0.006,
                0.014, 0.015, 0.016,
                0.040, 0.045, 0.050,
            ],
            "mean_industry_correlation_21d": [
                0.20, 0.22, 0.24,
                0.50, 0.52, 0.54,
                0.80, 0.82, 0.84,
            ],
            "cross_sectional_dispersion": [
                0.003, 0.004, 0.005,
                0.010, 0.011, 0.012,
                0.030, 0.035, 0.040,
            ],
        },
        index=pd.date_range("2024-01-01", periods=9, freq="B"),
    )


def test_predict_requires_fitting():
    detector = MarketRegimeDetector()

    with pytest.raises(RuntimeError, match="fit"):
        detector.predict(sample_features())


def test_predictions_preserve_dates_and_use_valid_labels():
    features = sample_features()
    detector = MarketRegimeDetector().fit(features)

    labels = detector.predict(features)

    assert labels.index.equals(features.index)
    assert labels.name == "regime"
    assert set(labels.unique()) == {0, 1, 2}


def test_prediction_does_not_change_fitted_state():
    training = sample_features()
    detector = MarketRegimeDetector().fit(training)

    original_mean = detector.scaler_.mean_.copy()
    original_scale = detector.scaler_.scale_.copy()
    original_centers = detector.model_.cluster_centers_.copy()

    later = training.iloc[:3].copy()
    later.index = pd.date_range("2024-02-01", periods=3, freq="B")
    later["portfolio_volatility_21d"] *= 10

    detector.predict(later)

    np.testing.assert_array_equal(
        detector.scaler_.mean_, original_mean
    )
    np.testing.assert_array_equal(
        detector.scaler_.scale_, original_scale
    )
    np.testing.assert_array_equal(
        detector.model_.cluster_centers_, original_centers
    )


def test_future_inputs_do_not_change_earlier_assignments():
    training = sample_features()
    detector = MarketRegimeDetector().fit(training)

    later = training.copy()
    later.index = pd.date_range("2024-02-01", periods=9, freq="B")

    changed = later.copy()
    changed.iloc[5:] = [0.50, 0.99, 0.30]

    original_labels = detector.predict(later)
    changed_labels = detector.predict(changed)

    pd.testing.assert_series_equal(
        original_labels.iloc[:5],
        changed_labels.iloc[:5],
    )


def test_centers_match_group_means_in_original_units():
    features = sample_features()
    detector = MarketRegimeDetector().fit(features)

    labels = detector.predict(features)
    centers = detector.cluster_centers()

    for regime in centers.index:
        expected = features.loc[labels == regime].mean()

        np.testing.assert_allclose(
            centers.loc[regime].to_numpy(),
            expected.to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        )


def test_missing_features_are_rejected():
    features = sample_features()
    features.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        MarketRegimeDetector().fit(features)


def test_changed_column_order_is_rejected():
    features = sample_features()
    detector = MarketRegimeDetector().fit(features)

    reordered = features[features.columns[::-1]]

    with pytest.raises(ValueError, match="names and order"):
        detector.predict(reordered)


def test_too_few_distinct_rows_are_rejected():
    features = sample_features()
    features.iloc[:] = features.iloc[0].to_numpy()

    with pytest.raises(ValueError, match="distinct"):
        MarketRegimeDetector(n_regimes=3).fit(features)


