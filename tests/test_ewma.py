
import numpy as np
import pandas as pd
import pytest

from src.models.ewma import forecast_ewma_volatility


def test_constant_returns_produce_expected_forecast():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series(0.01, index=dates)

    forecast = forecast_ewma_volatility(
        returns,
        decay=0.94,
        horizon=5,
        min_periods=3,
    )

    expected = np.sqrt(5 * 0.01**2)

    assert forecast.iloc[:2].isna().all()
    assert forecast.iloc[2:].notna().all()
    assert forecast.iloc[2] == pytest.approx(expected)
    assert forecast.iloc[-1] == pytest.approx(expected)


def test_future_return_does_not_change_earlier_forecast():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03, 0.04],
        index=dates,
    )

    changed = returns.copy()
    changed.iloc[4] = 1.50

    original_forecast = forecast_ewma_volatility(
        returns, min_periods=3
    )
    changed_forecast = forecast_ewma_volatility(
        changed, min_periods=3
    )

    assert original_forecast.iloc[3] == pytest.approx(
        changed_forecast.iloc[3]
    )


def test_invalid_decay_is_rejected():
    returns = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )

    with pytest.raises(ValueError, match="decay"):
        forecast_ewma_volatility(returns, decay=1.0)


def test_invalid_returns_are_rejected():
    returns = pd.Series(
        [0.01, np.nan, 0.03],
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )

    with pytest.raises(ValueError, match="non-finite"):
        forecast_ewma_volatility(returns)
