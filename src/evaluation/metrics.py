
import numpy as np
import pandas as pd


def calculate_volatility_loss_contributions(
    actual: pd.Series,
    forecast: pd.Series,
) -> dict[str, pd.Series]:
    """
    Calculate additive loss contributions for volatility forecasts.

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

    return {
        "absolute_error": pd.Series(
            np.abs(errors), index=actual.index, name="absolute_error"
        ),
        "squared_error": pd.Series(
            errors**2, index=actual.index, name="squared_error"
        ),
        "qlike": pd.Series(
            ratio - np.log(ratio) - 1,
            index=actual.index,
            name="qlike",
        ),
    }


def calculate_volatility_metrics(
    actual: pd.Series,
    forecast: pd.Series,
) -> dict[str, float]:
    """Calculate MAE, RMSE, and ratio-based QLIKE for volatility forecasts."""
    losses = calculate_volatility_loss_contributions(actual, forecast)

    return {
        "mae": float(losses["absolute_error"].mean()),
        "rmse": float(np.sqrt(losses["squared_error"].mean())),
        "qlike": float(losses["qlike"].mean()),
    }
