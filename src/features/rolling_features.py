import numpy as np
import pandas as pd


def build_regime_features(
    industry_returns: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Build market-regime features using information available through date t.

    Input:
        Decimal industry returns with a chronological DatetimeIndex.

    Output:
        Trailing portfolio volatility, mean industry correlation,
        and same-day cross-sectional dispersion.

    Incomplete rolling windows remain NaN.
    """
    if not isinstance(industry_returns, pd.DataFrame):
        raise TypeError("industry_returns must be a pandas DataFrame.")

    if industry_returns.empty:
        raise ValueError("industry_returns must not be empty.")

    if not isinstance(industry_returns.index, pd.DatetimeIndex):
        raise TypeError("industry_returns must have a DatetimeIndex.")

    if industry_returns.index.hasnans:
        raise ValueError("Dates must not contain missing values.")

    if industry_returns.index.has_duplicates:
        raise ValueError("Dates must be unique.")

    if not industry_returns.index.is_monotonic_increasing:
        raise ValueError("Dates must be sorted in ascending order.")

    if industry_returns.columns.has_duplicates:
        raise ValueError("Industry column names must be unique.")

    if industry_returns.shape[1] < 2:
        raise ValueError("At least two industries are required.")

    if (
        not isinstance(window, int)
        or isinstance(window, bool)
        or window < 2
    ):
        raise ValueError("window must be an integer of at least 2.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in industry_returns.dtypes
    ):
        raise TypeError("All industry return columns must be numeric.")

    if not np.isfinite(industry_returns.to_numpy(dtype=float)).all():
        raise ValueError("Industry returns must be finite and non-missing.")

    returns = industry_returns.astype(float)

    portfolio_returns = returns.mean(axis=1)

    portfolio_volatility = (
        portfolio_returns
        .pow(2)
        .rolling(window=window, min_periods=window)
        .mean()
        .pow(0.5)
    )
    
    n_industries = returns.shape[1]

    # Select each distinct industry pair once, excluding the diagonal.
    upper_triangle = np.triu_indices(n_industries, k=1)

    average_correlation = np.full(len(returns), np.nan)

    for end_position in range(window - 1, len(returns)):
        start_position = end_position - window + 1

        historical_window = returns.iloc[
            start_position : end_position + 1
        ]

        correlation_matrix = historical_window.corr().to_numpy()
        pair_correlations = correlation_matrix[upper_triangle]

        # Undefined correlations are preserved as missing.
        # This can happen if an industry has constant returns in the window.
        if np.isfinite(pair_correlations).all():
            average_correlation[end_position] = pair_correlations.mean()

    mean_industry_correlation = pd.Series(
        average_correlation,
        index=returns.index,
    )

    cross_sectional_dispersion = returns.std(axis=1, ddof=0)

    features = pd.DataFrame(
        {
            f"portfolio_volatility_{window}d": portfolio_volatility,
            f"mean_industry_correlation_{window}d": (
                mean_industry_correlation
            ),
            "cross_sectional_dispersion": cross_sectional_dispersion,
        },
        index=returns.index,
    )

    return features

