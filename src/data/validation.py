"""Validation rules for market-return inputs."""

import numpy as np
import pandas as pd


def validate_returns_frame(returns: pd.DataFrame) -> None:
    """Raise a clear error when a standardized return frame violates the contract."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns must have a DatetimeIndex.")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("Dates must be unique and increasing.")
    if returns.empty or returns.columns.empty:
        raise ValueError("Returns frame must contain dates and industry columns.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in returns.dtypes):
        raise TypeError("All industry return columns must be numeric.")
    if returns.isna().any().any():
        raise ValueError("Returns contain missing observations.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain non-finite observations.")
