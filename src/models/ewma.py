
import numpy as np
import pandas as pd


def forecast_ewma_volatility(
    portfolio_returns: pd.Series,
    decay: float = 0.94,
    horizon: int = 5,
    min_periods: int = 21,
) -> pd.Series:
    """
    Forecast next-horizon realized volatility using EWMA variance.

    Initialize variance using the mean squared return over the first
    min_periods observations. Then update variance recursively.

    At date t, the forecast uses returns available through date t.
    """
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series.")

    if not np.isfinite(decay) or not 0 < decay < 1:
        raise ValueError("decay must be strictly between 0 and 1.")

    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon < 1
    ):
        raise ValueError("horizon must be a positive integer.")

    if (
        not isinstance(min_periods, int)
        or isinstance(min_periods, bool)
        or min_periods < 1
    ):
        raise ValueError("min_periods must be a positive integer.")

    if portfolio_returns.empty:
        raise ValueError("portfolio_returns must not be empty.")

    if portfolio_returns.index.has_duplicates:
        raise ValueError("Portfolio returns contain duplicate dates.")

    if not portfolio_returns.index.is_monotonic_increasing:
        raise ValueError("Dates must be sorted in ascending order.")

    returns = portfolio_returns.to_numpy(dtype=float)

    if not np.isfinite(returns).all():
        raise ValueError(
            "Portfolio returns contain missing or non-finite values."
        )

    n = len(returns)
    variance = np.full(n, np.nan, dtype=float)

    # A forecast becomes available after min_periods observations.
    if n >= min_periods:
        variance[min_periods - 1] = np.mean(
            returns[:min_periods] ** 2
        )

        for i in range(min_periods, n):
            variance[i] = (
                decay * variance[i - 1]
                + (1 - decay) * returns[i] ** 2
            )

    forecast = np.sqrt(horizon * variance)

    return pd.Series(
        forecast,
        index=portfolio_returns.index,
        name=f"ewma_forecast_{horizon}d",
    )
