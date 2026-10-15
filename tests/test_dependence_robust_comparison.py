from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_dependence_robust_comparison as comparison
from src.evaluation.metrics import (
    calculate_volatility_loss_contributions,
    calculate_volatility_metrics,
)


def make_series() -> tuple[pd.Series, pd.Series, pd.Series]:
    dates = pd.bdate_range("2020-01-01", periods=80)
    actual = pd.Series(np.linspace(0.01, 0.03, len(dates)), index=dates)
    candidate = actual * np.where(np.arange(len(dates)) % 2, 1.01, 0.99)
    comparator = actual * np.where(np.arange(len(dates)) % 2, 1.25, 0.80)
    return actual, candidate, comparator


def test_positive_difference_favours_candidate_with_lower_loss():
    actual, candidate, comparator = make_series()
    comparison.validate_paired_inputs(actual, candidate, comparator)

    candidate_losses = calculate_volatility_loss_contributions(
        actual, candidate
    )
    comparator_losses = calculate_volatility_loss_contributions(
        actual, comparator
    )
    difference = comparator_losses["absolute_error"] - candidate_losses[
        "absolute_error"
    ]
    result = comparison.hac_mean_difference(difference, max_lag=4)

    assert result["mean_loss_difference_comparator_minus_candidate"] > 0
    assert result["ci_95_lower"] > 0


def test_loss_contributions_reproduce_central_metrics():
    actual, candidate, _ = make_series()
    losses = calculate_volatility_loss_contributions(actual, candidate)
    metrics = calculate_volatility_metrics(actual, candidate)

    assert losses["absolute_error"].mean() == pytest.approx(metrics["mae"])
    assert np.sqrt(losses["squared_error"].mean()) == pytest.approx(
        metrics["rmse"]
    )
    assert losses["qlike"].mean() == pytest.approx(metrics["qlike"])


def test_hac_mean_and_confidence_interval_are_finite():
    difference = pd.Series(np.linspace(-0.01, 0.03, 80))

    result = comparison.hac_mean_difference(difference, max_lag=10)

    assert result["mean_loss_difference_comparator_minus_candidate"] == pytest.approx(
        difference.mean()
    )
    assert np.isfinite(list(result.values())).all()
    assert result["ci_95_lower"] < result["ci_95_upper"]


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = comparison.holm_adjust(
        pd.Series([0.01, 0.04, 0.03], index=["first", "second", "third"])
    )

    assert adjusted["first"] == pytest.approx(0.03)
    assert adjusted["third"] == pytest.approx(0.06)
    assert adjusted["second"] == pytest.approx(0.06)


def test_date_misalignment_is_rejected():
    actual, candidate, comparator = make_series()
    comparator.index = comparator.index.shift(1, freq="D")

    with pytest.raises(ValueError, match="identical dates"):
        comparison.validate_paired_inputs(actual, candidate, comparator)


@pytest.mark.parametrize(
    ("difference", "lag", "message"),
    [
        (pd.Series([0.1, 0.2, 0.3]), 2, "Not enough observations"),
        (pd.Series([0.1, np.nan, 0.3, 0.4]), 1, "non-finite"),
    ],
)
def test_hac_rejects_invalid_inputs(difference, lag, message):
    with pytest.raises(ValueError, match=message):
        comparison.hac_mean_difference(difference, lag)


def test_moving_block_bootstrap_is_reproducible_and_finite():
    difference = pd.Series(np.linspace(-0.01, 0.02, 80))

    first = comparison.moving_block_bootstrap_ci(
        difference, block_length=5, repetitions=200, random_state=7
    )
    second = comparison.moving_block_bootstrap_ci(
        difference, block_length=5, repetitions=200, random_state=7
    )

    assert first == second
    assert np.isfinite(list(first.values())).all()
    assert first["bootstrap_ci_95_lower"] < first["bootstrap_ci_95_upper"]


def test_bootstrap_rejects_invalid_block_length():
    with pytest.raises(ValueError, match="block_length"):
        comparison.moving_block_bootstrap_ci(
            pd.Series([0.1, 0.2, 0.3]), block_length=4, repetitions=100
        )
