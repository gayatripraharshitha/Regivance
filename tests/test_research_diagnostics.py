from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import generate_research_diagnostics as diagnostics
from src.evaluation.metrics import calculate_volatility_metrics
from src.features.rolling_features import build_regime_features


def make_predictions(n: int = 21) -> pd.DataFrame:
    """Build a small, valid prediction table for diagnostics tests."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    actual = np.linspace(0.005, 0.020, n)

    frame = pd.DataFrame(
        {
            "date": dates,
            "actual": actual,
            "fold": 1,
            "regime": "middle",
        }
    )

    for position, model in enumerate(diagnostics.ALL_MODELS):
        frame[model] = actual * (0.80 + 0.03 * position)

    return frame


def test_valid_predictions_pass(tmp_path, monkeypatch):
    """Valid data should pass all prediction-table integrity checks."""
    predictions = make_predictions()
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "PREDICTIONS_PATH", path)

    checks = []
    result = diagnostics.load_and_validate_predictions(checks)

    assert len(result) == 21
    assert result["date"].is_unique
    assert result["date"].is_monotonic_increasing
    assert len(checks) == 3
    assert all(item["status"] == "PASS" for item in checks)


def test_missing_forecast_column_is_rejected(tmp_path, monkeypatch):
    predictions = make_predictions().drop(
        columns=["Risk-feature LSTM"]
    )
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "PREDICTIONS_PATH", path)

    with pytest.raises(ValueError, match="missing required columns"):
        diagnostics.load_and_validate_predictions([])


def test_duplicate_dates_are_rejected(tmp_path, monkeypatch):
    predictions = make_predictions()
    predictions.loc[1, "date"] = predictions.loc[0, "date"]

    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "PREDICTIONS_PATH", path)

    with pytest.raises(ValueError, match="duplicated"):
        diagnostics.load_and_validate_predictions([])


def test_non_finite_targets_are_rejected(tmp_path, monkeypatch):
    predictions = make_predictions()
    predictions.loc[3, "actual"] = np.nan

    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "PREDICTIONS_PATH", path)

    with pytest.raises(ValueError, match="missing or non-numeric"):
        diagnostics.load_and_validate_predictions([])


def test_non_positive_forecasts_are_rejected(tmp_path, monkeypatch):
    predictions = make_predictions()
    predictions.loc[5, "Risk-feature + regime LSTM"] = 0.0

    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "PREDICTIONS_PATH", path)

    with pytest.raises(ValueError, match="strictly positive|non-positive"):
        diagnostics.load_and_validate_predictions([])


def test_unknown_regime_is_rejected(tmp_path, monkeypatch):
    predictions = make_predictions()
    predictions.loc[0, "regime"] = "unknown"

    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "PREDICTIONS_PATH", path)

    with pytest.raises(ValueError, match="regime labels"):
        diagnostics.load_and_validate_predictions([])


def test_incomplete_fold_is_rejected(tmp_path, monkeypatch):
    predictions = make_predictions().iloc[:-1]

    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "PREDICTIONS_PATH", path)

    with pytest.raises(ValueError, match="21 test predictions"):
        diagnostics.load_and_validate_predictions([])


def test_target_alignment_passes(tmp_path, monkeypatch):
    predictions = make_predictions()

    dataset = pd.DataFrame(
        {"target_volatility_5d": predictions["actual"].to_numpy()},
        index=pd.DatetimeIndex(predictions["date"], name="date"),
    )

    path = tmp_path / "dataset.csv"
    dataset.to_csv(path)

    monkeypatch.setattr(diagnostics, "DATASET_PATH", path)

    checks = []
    diagnostics.validate_target_alignment(predictions, checks)

    assert len(checks) == 1
    assert checks[0]["status"] == "PASS"


def test_target_misalignment_is_rejected(tmp_path, monkeypatch):
    predictions = make_predictions()
    predictions.loc[0, "actual"] += 0.01

    dataset = pd.DataFrame(
        {
            "target_volatility_5d": np.linspace(
                0.005, 0.020, len(predictions)
            )
        },
        index=pd.DatetimeIndex(predictions["date"], name="date"),
    )

    path = tmp_path / "dataset.csv"
    dataset.to_csv(path)

    monkeypatch.setattr(diagnostics, "DATASET_PATH", path)

    with pytest.raises(ValueError, match="do not match"):
        diagnostics.validate_target_alignment(predictions, [])


def make_industry_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=60)

    values = rng.normal(
        loc=0.0,
        scale=0.01,
        size=(len(dates), 4),
    )

    return pd.DataFrame(
        values,
        index=dates,
        columns=["Industry_A", "Industry_B", "Industry_C", "Industry_D"],
    )


def test_feature_warmup_is_preserved(tmp_path, monkeypatch):
    returns = make_industry_returns()

    raw_path = tmp_path / "raw.csv"
    raw_path.touch()

    monkeypatch.setattr(diagnostics, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(
        diagnostics,
        "load_industry_returns",
        lambda _: returns,
    )

    checks = []
    _, summary, features = diagnostics.load_and_validate_risk_features(
        checks
    )

    assert features.index.equals(returns.index)
    assert summary["missing_count"].max() == 20
    assert any(
        item["check"] == "Rolling-feature missingness"
        for item in checks
    )


def test_unexpected_feature_gap_is_rejected(tmp_path, monkeypatch):
    returns = make_industry_returns()
    normal_features = build_regime_features(returns, window=21)

    # Introduce an invalid internal gap after the normal rolling warm-up.
    normal_features.iloc[30, 0] = np.nan

    raw_path = tmp_path / "raw.csv"
    raw_path.touch()

    monkeypatch.setattr(diagnostics, "RAW_DATA_PATH", raw_path)
    monkeypatch.setattr(
        diagnostics,
        "load_industry_returns",
        lambda _: returns,
    )
    monkeypatch.setattr(
        diagnostics,
        "build_regime_features",
        lambda *_args, **_kwargs: normal_features,
    )

    with pytest.raises(
        ValueError,
        match="after its initial warm-up",
    ):
        diagnostics.load_and_validate_risk_features([])


def make_fold_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Recalculate valid fold metrics for the synthetic test data."""
    records = []

    for model in diagnostics.ALL_MODELS:
        records.append(
            {
                "fold": 1,
                "model": model,
                "n_test": len(predictions),
                **calculate_volatility_metrics(
                    predictions["actual"],
                    predictions[model],
                ),
            }
        )

    return pd.DataFrame(records)


def test_saved_metrics_are_reproducible(tmp_path, monkeypatch):
    predictions = make_predictions()
    fold_metrics = make_fold_metrics(predictions)

    path = tmp_path / "fold_metrics.csv"
    fold_metrics.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "FOLD_METRICS_PATH", path)

    checks = []
    result = diagnostics.validate_saved_fold_metrics(
        predictions,
        checks,
    )

    assert len(result) == len(diagnostics.ALL_MODELS)
    assert checks[0]["status"] == "PASS"


def test_incorrect_saved_metric_is_rejected(tmp_path, monkeypatch):
    predictions = make_predictions()
    fold_metrics = make_fold_metrics(predictions)

    fold_metrics.loc[0, "mae"] += 0.01

    path = tmp_path / "fold_metrics.csv"
    fold_metrics.to_csv(path, index=False)

    monkeypatch.setattr(diagnostics, "FOLD_METRICS_PATH", path)

    with pytest.raises(ValueError, match="differs from recalculated"):
        diagnostics.validate_saved_fold_metrics(predictions, [])