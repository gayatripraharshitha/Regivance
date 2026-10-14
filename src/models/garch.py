
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model


Distribution = Literal["normal", "t"]


def forecast_garch_volatility_for_fold(
    training_returns: pd.Series,
    all_returns: pd.Series,
    forecast_dates: pd.DatetimeIndex,
    horizon: int = 5,
    scale: float = 100.0,
    distribution: Distribution = "normal",
    asymmetric: bool = False,
) -> pd.Series:
    """
    Fit GARCH(1,1) or GJR-GARCH(1,1) using training data only.

    Parameters
    ----------
    training_returns:
        Returns used to estimate model parameters.
    all_returns:
        Complete chronologically ordered return history. Returns after
        the training window update conditional variance as they become
        observable; they are not used to re-estimate model parameters.
    forecast_dates:
        Dates at which forecasts are required.
    horizon:
        Number of future observations covered by the volatility forecast.
    scale:
        Scale applied to returns during model estimation.
    distribution:
        Innovation distribution: "normal" or Student-t ("t").
    asymmetric:
        If True, use GJR-GARCH with a leverage/asymmetry term.

    Returns
    -------
    pd.Series
        Forecast volatility in the original return units, aligned to
        forecast_dates.

    Notes
    -----
    At forecast date t, the target covers returns t+1 through t+horizon.
    Forecasts use parameters estimated on training_returns and the
    conditional variance state implied by returns observed through t.
    """
    if not isinstance(training_returns, pd.Series):
        raise TypeError("training_returns must be a pandas Series.")

    if not isinstance(all_returns, pd.Series):
        raise TypeError("all_returns must be a pandas Series.")

    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon < 1
    ):
        raise ValueError("horizon must be a positive integer.")

    if (
        not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("scale must be a positive finite number.")

    if distribution not in ("normal", "t"):
        raise ValueError("distribution must be 'normal' or 't'.")

    if not isinstance(asymmetric, bool):
        raise TypeError("asymmetric must be a boolean.")

    if len(training_returns) < 100:
        raise ValueError(
            "At least 100 training observations are required for GARCH."
        )

    if training_returns.empty or all_returns.empty:
        raise ValueError("Return series must not be empty.")

    if (
        training_returns.index.has_duplicates
        or all_returns.index.has_duplicates
    ):
        raise ValueError("Return series contain duplicate dates.")

    if (
        not training_returns.index.is_monotonic_increasing
        or not all_returns.index.is_monotonic_increasing
    ):
        raise ValueError("Return dates must be sorted in ascending order.")

    train_values = training_returns.to_numpy(dtype=float)
    all_values = all_returns.to_numpy(dtype=float)

    if (
        not np.isfinite(train_values).all()
        or not np.isfinite(all_values).all()
    ):
        raise ValueError("Return series contain missing or non-finite values.")

    if not training_returns.index.isin(all_returns.index).all():
        raise ValueError("Training dates must be present in all_returns.")

    # Verify that the training values are consistent with the supplied
    # complete history, rather than silently fitting a different series.
    shared_training_values = all_returns.loc[training_returns.index].to_numpy(
        dtype=float
    )
    if not np.allclose(train_values, shared_training_values):
        raise ValueError(
            "training_returns values do not match all_returns."
        )

    forecast_dates = pd.DatetimeIndex(forecast_dates)

    if forecast_dates.empty:
        raise ValueError("forecast_dates must not be empty.")

    if forecast_dates.has_duplicates:
        raise ValueError("forecast_dates contain duplicates.")

    if not forecast_dates.is_monotonic_increasing:
        raise ValueError("forecast_dates must be sorted in ascending order.")

    training_end = training_returns.index[-1]

    if forecast_dates[0] <= training_end:
        raise ValueError(
            "All forecast dates must be after the final training date."
        )

    if not forecast_dates.isin(all_returns.index).all():
        raise ValueError(
            "Every forecast date must be present in all_returns."
        )

    # Fit the model using training returns only.
    scaled_training = training_returns.astype(float) * scale

    model = arch_model(
        scaled_training,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1 if asymmetric else 0,
        q=1,
        dist=distribution,
        rescale=False,
    )

    fitted = model.fit(disp="off", show_warning=False)

    if fitted.convergence_flag != 0:
        raise RuntimeError(
            "GARCH estimation did not converge successfully."
        )

    parameters = fitted.params

    omega = float(parameters["omega"])
    alpha = float(parameters["alpha[1]"])
    beta = float(parameters["beta[1]"])
    gamma = (
        float(parameters["gamma[1]"])
        if asymmetric
        else 0.0
    )

    # The fitted conditional variance at the final training date.
    conditional_volatility = np.asarray(fitted.conditional_volatility)
    conditional_variance = float(conditional_volatility[-1] ** 2)

    # Calculate the variance of the observation immediately after the
    # last training observation.
    last_training_return = float(scaled_training.iloc[-1])
    negative_shock_term = (
        gamma * last_training_return**2
        if last_training_return < 0
        else 0.0
    )

    conditional_variance = (
        omega
        + alpha * last_training_return**2
        + negative_shock_term
        + beta * conditional_variance
    )

    # Expected variance persistence for future, unobserved shocks.
    # The normal and Student-t innovation distributions are symmetric.
    expected_persistence = alpha + beta + gamma / 2.0

    after_training = all_returns.loc[
        (all_returns.index > training_end)
        & (all_returns.index <= forecast_dates[-1])
    ]

    requested_dates = set(forecast_dates)
    predictions: dict[pd.Timestamp, float] = {}

    for date, observed_return in after_training.items():
        scaled_return = float(observed_return) * scale

        negative_shock_term = (
            gamma * scaled_return**2
            if scaled_return < 0
            else 0.0
        )

        # Variance for the next observation after observing this return.
        next_variance = (
            omega
            + alpha * scaled_return**2
            + negative_shock_term
            + beta * conditional_variance
        )

        if date in requested_dates:
            future_variance = next_variance
            total_forecast_variance = 0.0

            for _ in range(horizon):
                total_forecast_variance += future_variance
                future_variance = (
                    omega + expected_persistence * future_variance
                )

            forecast = np.sqrt(total_forecast_variance) / scale

            if not np.isfinite(forecast) or forecast <= 0:
                raise RuntimeError(
                    f"Invalid GARCH volatility forecast on {date}."
                )

            predictions[date] = float(forecast)

        # Advance the state using the return now observed. Parameters stay
        # fixed throughout this fold.
        conditional_variance = next_variance

    if len(predictions) != len(forecast_dates):
        raise RuntimeError(
            "Failed to generate forecasts for all requested dates."
        )

    return pd.Series(
        predictions,
        index=forecast_dates,
        name=f"garch_forecast_{horizon}d",
        dtype=float,
    )
