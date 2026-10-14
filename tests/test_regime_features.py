import numpy as np
import pandas as pd
import pytest

from src.features.rolling_features import build_regime_features


def sample_returns():
    return pd.DataFrame(
        {
            "Industry_A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "Industry_B": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12],
        },
        index=pd.date_range("2024-01-01", periods=6, freq="B"),
    )


def test_features_match_known_values():
    returns = sample_returns()

    features = build_regime_features(returns, window=3)

    # Equal-weight portfolio returns: 0.015, 0.030, 0.045.
    expected_volatility = np.sqrt(
        (0.015**2 + 0.030**2 + 0.045**2) / 3
    )

    assert features.iloc[2]["portfolio_volatility_3d"] == pytest.approx(
        expected_volatility
    )

    # Industry B is exactly twice Industry A.
    assert features.iloc[2][
        "mean_industry_correlation_3d"
    ] == pytest.approx(1.0)

    # Population standard deviation of [0.01, 0.02].
    assert features.iloc[0][
        "cross_sectional_dispersion"
    ] == pytest.approx(0.005)


def test_incomplete_windows_remain_missing():
    features = build_regime_features(sample_returns(), window=3)

    rolling_columns = [
        "portfolio_volatility_3d",
        "mean_industry_correlation_3d",
    ]

    assert features[rolling_columns].iloc[:2].isna().all().all()
    assert features[rolling_columns].iloc[2:].notna().all().all()

    # Same-day dispersion does not need a rolling window.
    assert features["cross_sectional_dispersion"].notna().all()


def test_future_returns_cannot_change_earlier_features():
    returns = sample_returns()
    changed_returns = returns.copy()

    # Change only observations after the fourth date.
    changed_returns.iloc[4:] = [
        [0.50, -0.30],
        [-0.40, 0.60],
    ]

    original = build_regime_features(returns, window=3)
    changed = build_regime_features(changed_returns, window=3)

    pd.testing.assert_frame_equal(
        original.iloc[:4],
        changed.iloc[:4],
    )


def test_constant_industry_produces_undefined_correlation():
    returns = sample_returns()
    returns["Industry_B"] = 0.01

    features = build_regime_features(returns, window=3)

    assert features["mean_industry_correlation_3d"].isna().all()


def test_missing_returns_are_rejected():
    returns = sample_returns()
    returns.iloc[2, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_regime_features(returns, window=3)


def test_unsorted_dates_are_rejected():
    returns = sample_returns().iloc[::-1]

    with pytest.raises(ValueError, match="ascending"):
        build_regime_features(returns, window=3)


@pytest.mark.parametrize("window", [0, 1, True, 2.5])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError, match="window"):
        build_regime_features(sample_returns(), window=window)