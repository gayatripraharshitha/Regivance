import numpy as np
import pandas as pd
import pytest

from scripts.analyze_v2_extensions import inverse_volatility_portfolio, stress_table, var_coverage
from scripts.export_dashboard_data import export


def predictions():
    dates = pd.to_datetime(["2019-01-01", "2020-03-01", "2021-01-01"])
    return pd.DataFrame({"actual": [.02, .04, .03], "fold": [1, 2, 3],
        "regime": ["lower", "lower", "higher"], "GJR-GARCH-Student-t": [.02, .02, .03],
        "Risk-feature + regime LSTM": [.02, .01, .02]}, index=dates)


def test_stress_groups_are_exclusive_and_complete():
    result = stress_table(predictions())
    counts = result.groupby("period").n.first()
    assert counts.to_dict() == {"2020_shock": 1, "high_volatility": 1, "normal": 1}


def test_inverse_volatility_weights_are_lagged():
    dates = pd.date_range("2020-01-01", periods=5)
    returns = pd.DataFrame({"a": [1., 2., 3., 100., 5.], "b": [2., 4., 6., 8., 10.]}, index=dates)
    original = inverse_volatility_portfolio(returns, window=3)
    changed = returns.copy()
    changed.loc[dates[3], "a"] = -100.
    revised = inverse_volatility_portfolio(changed, window=3)
    # Current return changes the portfolio return, but must not change today's weights.
    prior_vol = returns.iloc[:3].std(ddof=0)
    weight_a = (1 / prior_vol.a) / (1 / prior_vol).sum()
    expected_change = (-100. - 100.) * weight_a
    assert revised.loc[dates[3]] - original.loc[dates[3]] == pytest.approx(expected_change)


def test_var_coverage_has_overlapping_and_nonoverlapping_samples():
    dates = pd.date_range("2020-01-01", periods=10)
    frame = pd.concat([predictions().iloc[[0]]] * 5)
    frame.index = dates[:5]
    portfolio = pd.Series(.001, index=dates)
    result = var_coverage(frame, portfolio)
    assert set(result["sample"]) == {"overlapping_all", "nonoverlapping_offset_0"}
    assert set(result["confidence"]) == {.95, .99}


def test_dashboard_export_rejects_incomplete_model_set(tmp_path):
    source = tmp_path / "metrics.csv"
    pd.DataFrame([{"model": "Rolling", "n_test": 2520, "mae": .1, "rmse": .1, "qlike": .1}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="model set"):
        export(source, tmp_path / "output.json")
