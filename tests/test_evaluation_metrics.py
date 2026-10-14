
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_volatility_metrics


def test_perfect_forecasts_have_zero_loss():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    actual = pd.Series([0.01, 0.02, 0.03], index=dates)

    metrics = calculate_volatility_metrics(
        actual=actual,
        forecast=actual.copy(),
    )

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["qlike"] == pytest.approx(0.0)


def test_poorer_forecasts_have_positive_errors():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    actual = pd.Series([0.01, 0.02, 0.03], index=dates)
    forecast = pd.Series([0.02, 0.01, 0.04], index=dates)

    metrics = calculate_volatility_metrics(actual, forecast)

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["qlike"] > 0


def test_mismatched_dates_are_rejected():
    actual = pd.Series(
        [0.01, 0.02],
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )
    forecast = pd.Series(
        [0.01, 0.02],
        index=pd.date_range("2024-01-02", periods=2, freq="B"),
    )

    with pytest.raises(ValueError, match="identical indices"):
        calculate_volatility_metrics(actual, forecast)


def test_nonpositive_forecasts_are_rejected():
    dates = pd.date_range("2024-01-01", periods=2, freq="B")
    actual = pd.Series([0.01, 0.02], index=dates)
    forecast = pd.Series([0.01, 0.0], index=dates)

    with pytest.raises(ValueError, match="must be positive"):
        calculate_volatility_metrics(actual, forecast)
