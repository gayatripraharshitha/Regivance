
import numpy as np
import pandas as pd
import pytest

from src.models.lstm_data import create_lstm_sequences


def sample_data():
    dates = pd.date_range("2024-01-01", periods=12, freq="B")

    features = pd.DataFrame(
        {
            "return_a": np.arange(12, dtype=float) / 100,
            "return_b": np.arange(12, dtype=float) / 200,
        },
        index=dates,
    )

    targets = pd.Series(
        np.arange(12, dtype=float) / 100 + 0.01,
        index=dates,
        name="target_volatility_5d",
    )

    return features, targets


def test_sequences_have_correct_shape_values_and_dates():
    features, targets = sample_data()

    x, y, dates = create_lstm_sequences(
        features,
        targets,
        target_positions=slice(3, 6),
        lookback=3,
    )

    assert x.shape == (3, 3, 2)
    assert y.shape == (3,)
    assert dates.equals(features.index[3:6])

    np.testing.assert_allclose(
        x[0],
        features.iloc[1:4].to_numpy(dtype=np.float32),
    )
    np.testing.assert_allclose(
        y,
        targets.iloc[3:6].to_numpy(dtype=np.float32),
    )


def test_future_features_do_not_change_earlier_sequences():
    features, targets = sample_data()

    original = create_lstm_sequences(
        features,
        targets,
        target_positions=[5],
        lookback=4,
    )

    changed_features = features.copy()
    changed_features.iloc[6:] = 999.0

    changed = create_lstm_sequences(
        changed_features,
        targets,
        target_positions=[5],
        lookback=4,
    )

    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(original[1], changed[1])
    assert original[2].equals(changed[2])


def test_incomplete_lookback_is_rejected():
    features, targets = sample_data()

    with pytest.raises(ValueError, match="lookback"):
        create_lstm_sequences(
            features,
            targets,
            target_positions=[1],
            lookback=3,
        )


def test_feature_and_target_dates_must_match():
    features, targets = sample_data()

    altered_targets = targets.copy()
    altered_targets.index = altered_targets.index.shift(1, freq="D")

    with pytest.raises(ValueError, match="identical dates"):
        create_lstm_sequences(
            features,
            altered_targets,
            target_positions=[5],
            lookback=3,
        )


def test_nonpositive_targets_are_rejected():
    features, targets = sample_data()
    targets.iloc[5] = 0.0

    with pytest.raises(ValueError, match="positive"):
        create_lstm_sequences(
            features,
            targets,
            target_positions=[5],
            lookback=3,
        )


def test_missing_values_inside_sequence_are_rejected():
    features, targets = sample_data()
    features.iloc[4, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        create_lstm_sequences(
            features,
            targets,
            target_positions=[5],
            lookback=3,
        )


def test_invalid_positions_are_rejected():
    features, targets = sample_data()

    with pytest.raises(ValueError, match="outside"):
        create_lstm_sequences(
            features,
            targets,
            target_positions=[len(features)],
            lookback=3,
        )


def test_invalid_lookback_is_rejected():
    features, targets = sample_data()

    with pytest.raises(ValueError, match="lookback"):
        create_lstm_sequences(
            features,
            targets,
            target_positions=[5],
            lookback=0,
        )
