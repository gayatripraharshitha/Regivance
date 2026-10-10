
import numpy as np
import pandas as pd


def forecast_rolling_volatility(
    portfolio_returns: pd.Series,
    window: int = 21,
    horizon: int = 5,
) -> pd.Series:
    """
    Forecast next-horizon realized volatility using recent squared returns.

    At each date, use the current and previous window - 1 returns.
    Forecast = sqrt(horizon * mean of the squared returns).

    Returns NaN until a complete historical window is available.
    """
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series.")

    if (
        not isinstance(window, int)
        or isinstance(window, bool)
        or window < 1
    ):
        raise ValueError("window must be a positive integer.")

    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon < 1
    ):
        raise ValueError("horizon must be a positive integer.")

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

    mean_squared_return = portfolio_returns.pow(2).rolling(
        window=window,
        min_periods=window,
    ).mean()

    forecast = np.sqrt(horizon * mean_squared_return)

    return forecast.rename(
        f"rolling_volatility_{window}d_forecast_{horizon}d"
    )
