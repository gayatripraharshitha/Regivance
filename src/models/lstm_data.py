
from __future__ import annotations

import numpy as np
import pandas as pd


def create_lstm_sequences(
    features: pd.DataFrame,
    targets: pd.Series,
    target_positions: slice | np.ndarray | list[int],
    lookback: int = 21,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Construct trailing input sequences for volatility forecasting.

    Each sequence ending at position t contains feature rows:
        t-lookback+1, ..., t

    Its label is targets[t], which represents the future five-observation
    volatility target constructed elsewhere in the data pipeline.

    This function does not fit any preprocessing or model. The caller is
    responsible for fitting scalers and learned transformations using
    training data only.

    Parameters
    ----------
    features:
        Chronologically ordered, finite feature matrix with a DatetimeIndex.
    targets:
        Positive volatility targets indexed by the same dates and in the
        same order as features.
    target_positions:
        Positions of the target dates to include. Can be a Python slice
        or a sequence of integer positions.
    lookback:
        Number of historical feature rows in each sequence.

    Returns
    -------
    sequences:
        Float32 array of shape (samples, lookback, features).
    labels:
        Float32 array of shape (samples,).
    target_dates:
        Dates aligned with the labels.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")

    if not isinstance(targets, pd.Series):
        raise TypeError("targets must be a pandas Series.")

    if features.empty or features.shape[1] == 0:
        raise ValueError("features must contain rows and columns.")

    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must have a DatetimeIndex.")

    if (
        features.index.hasnans
        or features.index.has_duplicates
        or not features.index.is_monotonic_increasing
    ):
        raise ValueError(
            "Feature dates must be non-missing, unique and chronological."
        )

    if not features.columns.is_unique:
        raise ValueError("Feature names must be unique.")

    if not features.index.equals(targets.index):
        raise ValueError(
            "features and targets must have identical dates and ordering."
        )

    if (
        not isinstance(lookback, int)
        or isinstance(lookback, bool)
        or lookback < 1
    ):
        raise ValueError("lookback must be a positive integer.")

    if isinstance(target_positions, slice):
        if target_positions.step not in (None, 1):
            raise ValueError("Sliced target positions cannot have a step.")

        start = 0 if target_positions.start is None else target_positions.start
        stop = (
            len(features)
            if target_positions.stop is None
            else target_positions.stop
        )

        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
        ):
            raise ValueError("Slice bounds must be integers.")

        positions = np.arange(start, stop, dtype=np.int64)
    else:
        raw_positions = np.asarray(target_positions)

        if raw_positions.ndim != 1 or raw_positions.size == 0:
            raise ValueError("target_positions must be a non-empty 1D sequence.")

        if not np.issubdtype(raw_positions.dtype, np.integer):
            raise ValueError("target_positions must contain integers.")

        positions = raw_positions.astype(np.int64)

    if positions.size == 0:
        raise ValueError("No target positions were selected.")

    if (positions < 0).any() or (positions >= len(features)).any():
        raise ValueError("Target positions fall outside the feature data.")

    if np.any(np.diff(positions) <= 0):
        raise ValueError(
            "Target positions must be unique and strictly increasing."
        )

    if positions[0] < lookback - 1:
        raise ValueError(
            "A target position occurs before a complete lookback window."
        )

    all_targets = targets.to_numpy(dtype=np.float64)

    if not np.isfinite(all_targets[positions]).all():
        raise ValueError("Selected targets contain missing or non-finite values.")

    if (all_targets[positions] <= 0).any():
        raise ValueError("Selected volatility targets must be positive.")

    feature_values = features.to_numpy(dtype=np.float64)

    sequences = np.empty(
        (len(positions), lookback, features.shape[1]),
        dtype=np.float32,
    )

    for sample_index, target_position in enumerate(positions):
        start_position = target_position - lookback + 1
        end_position = target_position + 1

        historical_window = feature_values[start_position:end_position]

        if not np.isfinite(historical_window).all():
            raise ValueError(
                "An input sequence contains missing or non-finite features."
            )

        sequences[sample_index] = historical_window.astype(np.float32)

    labels = all_targets[positions].astype(np.float32)
    target_dates = pd.DatetimeIndex(features.index[positions])

    return sequences, labels, target_dates
