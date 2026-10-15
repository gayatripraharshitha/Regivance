from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_forecast_failures as failures
from src.evaluation.metrics import calculate_volatility_metrics


def make_predictions() -> pd.DataFrame:
    """Create a valid 120-fold prediction file with ordered fold blocks."""
    n_per_fold = 21
    n_rows = 120 * n_per_fold
    index = np.arange(n_rows)
    data = {
        "date": pd.bdate_range("2016-01-01", periods=n_rows),
        "actual": 0.01 + index * 0.000001,
        "fold": np.repeat(np.arange(1, 121), n_per_fold),
        "regime": np.where(index % 3 == 0, "lower", "middle"),
    }

    for number, model in enumerate(failures.MODELS, start=1):
        data[model] = data["actual"] * (1.0 + number * 0.01)

    return pd.DataFrame(data)


def write_predictions(tmp_path, predictions: pd.DataFrame, monkeypatch) -> None:
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    monkeypatch.setattr(failures, "PREDICTIONS_PATH", path)


def test_load_predictions_accepts_expected_fold_structure(tmp_path, monkeypatch):
    predictions = make_predictions()
    write_predictions(tmp_path, predictions, monkeypatch)

    loaded = failures.load_predictions()

    pd.testing.assert_frame_equal(loaded, predictions)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.drop(columns=["actual"]),
            "missing columns",
        ),
        (
            lambda frame: frame.assign(
                date=lambda x: x["date"].mask(
                    x.index == 1, x.loc[0, "date"]
                )
            ),
            "unique",
        ),
        (
            lambda frame: frame.assign(
                actual=lambda x: x["actual"].mask(x.index == 0, np.nan)
            ),
            "missing or non-numeric",
        ),
        (
            lambda frame: frame.assign(
                **{
                    "Risk-feature + regime LSTM": lambda x: x[
                        "Risk-feature + regime LSTM"
                    ].mask(x.index == 0, 0.0)
                }
            ),
            "must be positive",
        ),
    ],
)
def test_load_predictions_rejects_invalid_inputs(
    tmp_path, monkeypatch, mutate, message
):
    write_predictions(tmp_path, mutate(make_predictions()), monkeypatch)

    with pytest.raises(ValueError, match=message):
        failures.load_predictions()


def test_load_predictions_rejects_nonconsecutive_fold_identifiers(
    tmp_path, monkeypatch
):
    predictions = make_predictions()
    predictions["fold"] += 1
    write_predictions(tmp_path, predictions, monkeypatch)

    with pytest.raises(ValueError, match="consecutively"):
        failures.load_predictions()


def test_load_predictions_rejects_interleaved_fold_blocks(tmp_path, monkeypatch):
    predictions = make_predictions()
    predictions.loc[20, "fold"] = 2
    predictions.loc[21, "fold"] = 1
    write_predictions(tmp_path, predictions, monkeypatch)

    with pytest.raises(ValueError, match="chronological block"):
        failures.load_predictions()


def test_loss_contributions_reproduce_central_metrics():
    predictions = make_predictions()

    results = failures.add_loss_contributions(predictions)

    for model in failures.MODELS:
        expected = calculate_volatility_metrics(
            predictions["actual"], predictions[model]
        )
        assert results[f"qlike__{model}"].mean() == pytest.approx(
            expected["qlike"]
        )
        assert results[f"absolute_error__{model}"].mean() == pytest.approx(
            expected["mae"]
        )
        assert np.sqrt(results[f"squared_error__{model}"].mean()) == pytest.approx(
            expected["rmse"]
        )


def test_concentration_and_fold_summary_are_consistent():
    results = failures.add_loss_contributions(make_predictions())

    concentration = failures.calculate_concentration(results)
    fold_summary = failures.calculate_fold_summary(results)

    assert len(concentration) == len(failures.MODELS)
    assert len(fold_summary) == 120 * len(failures.MODELS)
    assert concentration["n_observations"].eq(len(results)).all()
    assert fold_summary["n_test"].eq(21).all()
    assert concentration.filter(like="loss_share_pct").le(100.0).all().all()
    assert concentration.filter(like="loss_share_pct").ge(0.0).all().all()
