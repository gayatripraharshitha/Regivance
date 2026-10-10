
import numpy as np
import pandas as pd
import pytest

from src.data.portfolio import build_forecasting_dataset


def test_equal_weighting_and_forward_target():
    dates = pd.date_range("2024-01-01", periods=7, freq="B")

    # Artificial daily returns for two industries.
    industry_returns = pd.DataFrame(
        {
            "Industry_A": [0.02, -0.04, 0.06, -0.08, 0.10, -0.12, 0.14],
            "Industry_B": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )

    result = build_forecasting_dataset(industry_returns, horizon=5)

    # Equal weighting: (Industry A + Industry B) / 2.
    expected_portfolio_return = 0.01
    assert result.loc[dates[0], "portfolio_return"] == pytest.approx(
        expected_portfolio_return
    )

    # The first target must use the NEXT five portfolio returns,
    # not the return on the forecast date.
    portfolio_returns = industry_returns.mean(axis=1)
    expected_target = np.sqrt(
        np.sum(portfolio_returns.iloc[1:6].to_numpy() ** 2)
    )

    assert result.loc[dates[0], "target_volatility_5d"] == pytest.approx(
        expected_target
    )

    # Seven observations provide only two complete five-day targets.
    assert len(result) == 2
    assert result.index[-1] == dates[1]


def test_invalid_horizon_is_rejected():
    dates = pd.date_range("2024-01-01", periods=7, freq="B")
    returns = pd.DataFrame({"Industry_A": np.zeros(7)}, index=dates)

    with pytest.raises(ValueError, match="positive integer"):
        build_forecasting_dataset(returns, horizon=0)
