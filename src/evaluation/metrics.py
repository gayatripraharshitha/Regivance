
import numpy as np
import pandas as pd


def calculate_volatility_metrics(
    actual: pd.Series,
    forecast: pd.Series,
) -> dict[str, float]:
    """
    Calculate MAE, RMSE, and ratio-based QLIKE for volatility forecasts.

    Actual and forecast values must use the same dates and units.
    QLIKE is calculated from squared volatility values.
    """
    if not isinstance(actual, pd.Series) or not isinstance(forecast, pd.Series):
        raise TypeError("actual and forecast must be pandas Series.")

    if actual.empty or forecast.empty:
        raise ValueError("actual and forecast must not be empty.")

    if not actual.index.equals(forecast.index):
        raise ValueError("actual and forecast must have identical indices.")

    actual_values = actual.to_numpy(dtype=float)
    forecast_values = forecast.to_numpy(dtype=float)

    if not np.isfinite(actual_values).all():
        raise ValueError("actual contains missing or non-finite values.")

    if not np.isfinite(forecast_values).all():
        raise ValueError("forecast contains missing or non-finite values.")

    if (actual_values <= 0).any():
        raise ValueError(
            "actual volatility must be positive for ratio-based QLIKE."
        )

    if (forecast_values <= 0).any():
        raise ValueError("forecast volatility must be positive.")

    errors = forecast_values - actual_values

    actual_variance = actual_values**2
    forecast_variance = forecast_values**2
    ratio = actual_variance / forecast_variance

    qlike = np.mean(ratio - np.log(ratio) - 1)

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "qlike": float(qlike),
    }
