import numpy as np
import pandas as pd
import pytest

from src.models.garch import forecast_garch_volatility_for_fold


def make_returns() -> pd.Series:
    """Generate reproducible returns with changing volatility."""
    rng = np.random.default_rng(42)

    volatility = np.where(
        np.arange(600) < 350,
        0.01,
        0.02,
    )

    values = rng.normal(0.0, volatility, size=600)
    dates = pd.date_range("2020-01-01", periods=600, freq="B")

    return pd.Series(values, index=dates, name="portfolio_return")


@pytest.mark.parametrize(
    ("distribution", "asymmetric"),
    [
        ("t", False),  # Student-t GARCH(1,1)
        ("t", True),   # Student-t GJR-GARCH(1,1)
    ],
)
def test_garch_variants_produce_valid_forecasts(
    distribution: str,
    asymmetric: bool,
) -> None:
    returns = make_returns()

    training_returns = returns.iloc[:400]
    forecast_dates = returns.index[450:453]

    forecasts = forecast_garch_volatility_for_fold(
        training_returns=training_returns,
        all_returns=returns,
        forecast_dates=forecast_dates,
        horizon=5,
        distribution=distribution,
        asymmetric=asymmetric,
    )

    assert forecasts.index.equals(forecast_dates)
    assert len(forecasts) == len(forecast_dates)
    assert np.isfinite(forecasts.to_numpy()).all()
    assert (forecasts > 0).all()


def test_invalid_distribution_is_rejected() -> None:
    returns = make_returns()

    with pytest.raises(ValueError, match="distribution"):
        forecast_garch_volatility_for_fold(
            training_returns=returns.iloc[:400],
            all_returns=returns,
            forecast_dates=returns.index[450:453],
            distribution="invalid",  # type: ignore[arg-type]
        )


def test_asymmetric_must_be_boolean() -> None:
    returns = make_returns()

    with pytest.raises(TypeError, match="asymmetric"):
        forecast_garch_volatility_for_fold(
            training_returns=returns.iloc[:400],
            all_returns=returns,
            forecast_dates=returns.index[450:453],
            asymmetric=1,  # type: ignore[arg-type]
        )