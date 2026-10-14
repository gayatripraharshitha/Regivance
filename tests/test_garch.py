
import numpy as np
import pandas as pd
import pytest

from src.models.garch import forecast_garch_volatility_for_fold


def make_returns():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=260)
    values = rng.normal(loc=0.0002, scale=0.01, size=len(dates))
    return pd.Series(values, index=dates, name="portfolio_return")


def test_garch_forecasts_are_positive_and_aligned():
    returns = make_returns()
    training = returns.iloc[:200]
    forecast_dates = returns.index[220:225]

    forecasts = forecast_garch_volatility_for_fold(
        training_returns=training,
        all_returns=returns,
        forecast_dates=forecast_dates,
        horizon=5,
    )

    assert forecasts.index.equals(forecast_dates)
    assert len(forecasts) == 5
    assert np.isfinite(forecasts.to_numpy()).all()
    assert (forecasts > 0).all()


def test_future_returns_do_not_change_earlier_forecasts():
    returns = make_returns()
    training = returns.iloc[:200]
    forecast_dates = returns.index[220:225]

    original = forecast_garch_volatility_for_fold(
        training_returns=training,
        all_returns=returns,
        forecast_dates=forecast_dates,
        horizon=5,
    )

    changed_returns = returns.copy()
    changed_returns.loc[returns.index[230:]] = 0.5

    changed = forecast_garch_volatility_for_fold(
        training_returns=training,
        all_returns=changed_returns,
        forecast_dates=forecast_dates,
        horizon=5,
    )

    # Alterations after the final forecast date must not affect predictions.
    np.testing.assert_allclose(
        original.to_numpy(),
        changed.to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )


def test_forecast_dates_must_follow_training_dates():
    returns = make_returns()
    training = returns.iloc[:200]

    with pytest.raises(ValueError, match="after the final training date"):
        forecast_garch_volatility_for_fold(
            training_returns=training,
            all_returns=returns,
            forecast_dates=returns.index[199:202],
            horizon=5,
        )


def test_invalid_horizon_is_rejected():
    returns = make_returns()

    with pytest.raises(ValueError, match="horizon"):
        forecast_garch_volatility_for_fold(
            training_returns=returns.iloc[:200],
            all_returns=returns,
            forecast_dates=returns.index[220:225],
            horizon=0,
        )
