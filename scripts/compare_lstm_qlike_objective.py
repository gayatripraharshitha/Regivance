from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_lstm_models import LOOKBACK, make_model_config
from scripts.compare_lstm_risk_features import (
    build_model_inputs,
    scale_returns,
    scale_risk_features,
)
from scripts.compare_lstm_models import build_regime_one_hot, load_inputs
from src.evaluation.metrics import calculate_volatility_metrics
from src.features.rolling_features import build_regime_features
from src.models.lstm import LSTMVolatilityForecaster
from src.models.lstm_data import create_lstm_sequences
from src.research_config import CONFIG

EXISTING_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "lstm_risk_feature_comparison_predictions.csv"
)
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

EXPERIMENT_MODEL = "QLIKE-objective Risk-feature + regime LSTM"
EXISTING_MODEL = "Risk-feature + regime LSTM"
GJR_MODEL = "GJR-GARCH-Student-t"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled QLIKE-objective LSTM comparison."
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        required=True,
        help="Explicit number of chronological folds to run.",
    )
    parser.add_argument("--epochs", type=int, default=CONFIG["models"]["epochs"])
    parser.add_argument("--hidden-size", type=int, default=CONFIG["models"]["hidden_size"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["models"]["batch_size"])
    return parser.parse_args()


def load_existing_predictions() -> pd.DataFrame:
    """Load and validate the saved risk-feature comparison predictions."""
    if not EXISTING_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Existing predictions not found: {EXISTING_PREDICTIONS_PATH}"
        )
    data = pd.read_csv(EXISTING_PREDICTIONS_PATH, parse_dates=["date"])
    required = {"date", "actual", "fold", EXISTING_MODEL, GJR_MODEL}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Existing predictions are missing columns: {sorted(missing)}")
    data = data.set_index("date").sort_index()
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Existing prediction dates must be unique and chronological.")
    for column in ["actual", EXISTING_MODEL, GJR_MODEL]:
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Existing column {column!r} must be finite and positive.")
        data[column] = values
    return data


def evaluate_qlike_model(
    features: pd.DataFrame,
    targets: pd.Series,
    split,
    config,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, LSTMVolatilityForecaster]:
    """Fit with QLIKE and early-stop using validation QLIKE only."""
    x_train, y_train, train_dates = create_lstm_sequences(
        features, targets, split.train, lookback=LOOKBACK
    )
    x_validation, y_validation, validation_dates = create_lstm_sequences(
        features, targets, split.validation, lookback=LOOKBACK
    )
    x_test, y_test, test_dates = create_lstm_sequences(
        features, targets, split.test, lookback=LOOKBACK
    )
    if not train_dates.equals(targets.index[split.train]):
        raise RuntimeError("Training sequence dates are misaligned.")
    if not validation_dates.equals(targets.index[split.validation]):
        raise RuntimeError("Validation sequence dates are misaligned.")
    if not test_dates.equals(targets.index[split.test]):
        raise RuntimeError("Test sequence dates are misaligned.")

    forecaster = LSTMVolatilityForecaster(config).fit(
        x_train, y_train, x_validation, y_validation
    )
    forecasts = forecaster.predict(x_test)
    if not np.isfinite(forecasts).all() or (forecasts <= 0).any():
        raise RuntimeError("QLIKE-objective model produced invalid forecasts.")
    return forecasts, y_test, test_dates, forecaster


def ensure_outputs_absent(paths: list[Path]) -> None:
    """Prevent accidental replacement of an earlier experimental result."""
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing controlled-experiment outputs: "
            f"{existing}"
        )


def main() -> None:
    args = parse_arguments()
    if not 1 <= args.max_folds <= 120:
        raise ValueError("--max-folds must be between 1 and 120.")
    if args.epochs < 1 or args.hidden_size < 1 or args.batch_size < 1:
        raise ValueError("Epochs, hidden size, and batch size must be positive.")

    outputs = {
        "predictions": TABLES_DIR / "lstm_qlike_objective_predictions.csv",
        "metrics": TABLES_DIR / "lstm_qlike_objective_metrics.csv",
        "fold_metrics": TABLES_DIR / "lstm_qlike_objective_fold_metrics.csv",
        "training": TABLES_DIR / "lstm_qlike_objective_training.csv",
        "failures": TABLES_DIR / "lstm_qlike_objective_failures.csv",
    }
    ensure_outputs_absent(list(outputs.values()))
    torch.set_num_threads(1)

    industry_returns, dataset, _, all_splits = load_inputs()
    existing = load_existing_predictions()
    expected_all_dates = pd.DatetimeIndex(
        np.concatenate([dataset.index[split.test].to_numpy() for split in all_splits])
    )
    if not existing.index.equals(expected_all_dates):
        raise ValueError("Existing LSTM predictions do not match configured test dates.")
    if not np.allclose(
        existing["actual"].to_numpy(dtype=float),
        dataset.loc[expected_all_dates, "target_volatility_5d"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Existing LSTM targets differ from the reconstructed dataset.")

    splits = all_splits[: args.max_folds]
    targets = dataset["target_volatility_5d"].astype(float)
    raw_returns = industry_returns.reindex(dataset.index)
    raw_risk_features = build_regime_features(industry_returns, window=21).reindex(
        dataset.index
    )
    if not raw_returns.index.equals(dataset.index) or not raw_risk_features.index.equals(
        dataset.index
    ):
        raise RuntimeError("Reconstructed inputs are not aligned with target dates.")

    frames: list[pd.DataFrame] = []
    training_records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for split in splits:
        test_dates = dataset.index[split.test]
        try:
            train_dates = dataset.index[split.train]
            scaled_returns = scale_returns(raw_returns, train_dates)
            scaled_risk_features = scale_risk_features(raw_risk_features, train_dates)
            regime_one_hot, _ = build_regime_one_hot(raw_risk_features, train_dates)
            model_features = build_model_inputs(
                scaled_returns, scaled_risk_features, regime_one_hot
            )[EXISTING_MODEL]
            config = replace(
                make_model_config(
                    split.fold, args.epochs, args.hidden_size, args.batch_size
                ),
                objective="qlike",
            )
            forecasts, y_test, forecast_dates, forecaster = evaluate_qlike_model(
                model_features, targets, split, config
            )
            if not forecast_dates.equals(test_dates):
                raise RuntimeError("Experimental forecast dates do not match test dates.")
            reference = existing.loc[test_dates]
            if not np.allclose(
                reference["actual"].to_numpy(dtype=float),
                y_test.astype(float),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise RuntimeError("Experimental targets do not match saved targets.")

            frame = reference[["actual", EXISTING_MODEL, GJR_MODEL, "fold", "regime"]].copy()
            frame[EXPERIMENT_MODEL] = forecasts
            frames.append(frame)
            training_records.append(
                {
                    "fold": split.fold,
                    "best_epoch": forecaster.best_epoch_,
                    "epochs_trained": forecaster.n_epochs_trained_,
                    "best_validation_qlike": forecaster.best_validation_loss_,
                    "objective": forecaster.objective_,
                }
            )
            print(
                f"Fold {split.fold:03d}: best epoch {forecaster.best_epoch_}; "
                f"validation QLIKE {forecaster.best_validation_loss_:.6f}"
            )
        except Exception as exc:
            failures.append({"fold": split.fold, "error": repr(exc)})
            break

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    if failures:
        pd.DataFrame(failures).to_csv(outputs["failures"], index=False)
        raise RuntimeError(
            f"QLIKE-objective experiment failed in fold {failures[0]['fold']}; "
            f"details saved to {outputs['failures']}"
        )

    results = pd.concat(frames).sort_index()
    expected_dates = pd.DatetimeIndex(
        np.concatenate([dataset.index[split.test].to_numpy() for split in splits])
    )
    if results.index.has_duplicates or not results.index.equals(expected_dates):
        raise RuntimeError("Experimental result dates are incomplete or misaligned.")

    models = [EXPERIMENT_MODEL, EXISTING_MODEL, GJR_MODEL]
    records = []
    fold_records = []
    for model in models:
        records.append(
            {
                "model": model,
                "n_test": len(results),
                **calculate_volatility_metrics(results["actual"], results[model]),
            }
        )
    for fold, frame in results.groupby("fold", sort=True):
        for model in models:
            fold_records.append(
                {
                    "fold": fold,
                    "model": model,
                    "n_test": len(frame),
                    **calculate_volatility_metrics(frame["actual"], frame[model]),
                }
            )

    results.to_csv(outputs["predictions"], index_label="date")
    pd.DataFrame(records).to_csv(outputs["metrics"], index=False)
    pd.DataFrame(fold_records).to_csv(outputs["fold_metrics"], index=False)
    pd.DataFrame(training_records).to_csv(outputs["training"], index=False)
    if len(splits) < len(all_splits):
        print("\nSMOKE TEST ONLY: these results cover a subset of folds.")
    else:
        print("\nFull development comparison; exploratory, not an untouched final test.")
    print(pd.DataFrame(records).to_string(index=False, float_format=lambda x: f"{x:.8f}"))


if __name__ == "__main__":
    main()
