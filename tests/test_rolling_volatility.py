
import numpy as np
import pandas as pd
import pytest

from src.models.rolling_volatility import forecast_rolling_volatility


def test_forecast_matches_known_constant_returns():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series(0.01, index=dates, name="portfolio_return")

    forecast = forecast_rolling_volatility(
        returns,
        window=3,
        horizon=5,
    )

    expected = np.sqrt(5 * 0.01**2)

    assert forecast.iloc[:2].isna().all()
    assert forecast.iloc[2:].notna().all()
    assert forecast.iloc[2] == pytest.approx(expected)


def test_future_return_does_not_change_an_earlier_forecast():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03, 0.04],
        index=dates,
    )

    changed_returns = returns.copy()
    changed_returns.iloc[4] = 1.50

    original_forecast = forecast_rolling_volatility(
        returns, window=3, horizon=5
    )
    changed_forecast = forecast_rolling_volatility(
        changed_returns, window=3, horizon=5
    )

    # Changing a future return must not change an earlier forecast.
    assert original_forecast.iloc[3] == pytest.approx(
        changed_forecast.iloc[3]
    )


def test_invalid_window_is_rejected():
    returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )

    with pytest.raises(ValueError, match="positive integer"):
        forecast_rolling_volatility(returns, window=0)
