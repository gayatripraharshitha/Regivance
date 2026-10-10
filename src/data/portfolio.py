
import numpy as np
import pandas as pd


def build_forecasting_dataset(
    industry_returns: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Build an equal-weighted portfolio and its forward realized-volatility
    target.

    At date t:
      portfolio_return = equal-weighted return across all industries at t
      target = sqrt(sum of squared portfolio returns from t+1 to t+horizon)

    Returns a DataFrame indexed by date.
    The final `horizon` observations are dropped because their future
    targets cannot be calculated from the available data.
    """
    if industry_returns.empty:
        raise ValueError("Industry returns must not be empty.")

    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    if len(industry_returns) <= horizon:
        raise ValueError("Not enough observations for the requested horizon.")

    if industry_returns.index.has_duplicates:
        raise ValueError("Industry returns contain duplicate dates.")

    if not industry_returns.index.is_monotonic_increasing:
        raise ValueError("Dates must be sorted in ascending order.")

    if industry_returns.isna().any().any():
        raise ValueError("Industry returns contain missing values.")

    values = industry_returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Industry returns contain non-finite values.")

    # Give each industry an equal weight of 1 / number of industries.
    portfolio_return = industry_returns.mean(axis=1)

    # At date t, collect squared returns from t+1 through t+horizon.
    future_squared_returns = pd.concat(
        [
            portfolio_return.shift(-step).pow(2).rename(f"day_{step}")
            for step in range(1, horizon + 1)
        ],
        axis=1,
    )

    target_name = f"target_volatility_{horizon}d"

    target = np.sqrt(
        future_squared_returns.sum(axis=1, min_count=horizon)
    ).rename(target_name)

    dataset = pd.concat(
        [
            portfolio_return.rename("portfolio_return"),
            target,
        ],
        axis=1,
    )

    # The last horizon rows have no complete future target.
    dataset = dataset.dropna(subset=[target_name])

    return dataset
