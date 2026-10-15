
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_statistical_models import (
    DATA_PATH,
    EMBARGO,
    FINAL_TEST_SIZE,
    GARCH_VARIANTS,
    HORIZON,
    STEP_SIZE,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)
from src.data.loader import load_industry_returns
from src.data.portfolio import build_forecasting_dataset
from src.evaluation.metrics import calculate_volatility_metrics
from src.evaluation.splits import generate_walk_forward_splits
from src.features.regimes import MarketRegimeDetector
from src.features.rolling_features import build_regime_features
from src.models.lstm import LSTMConfig, LSTMVolatilityForecaster
from src.models.lstm_data import create_lstm_sequences


RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "30_Industry_Portfolios_Daily.csv"
)

BASELINE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "walk_forward_statistical_models_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"

LOOKBACK = 21
REGIME_NAMES = ("lower", "middle", "higher")

STATISTICAL_MODELS = [
    "Rolling",
    "EWMA",
    *[name for name, _, _ in GARCH_VARIANTS],
]

ALL_MODELS = [
    *STATISTICAL_MODELS,
    "LSTM",
    "Regime-aware LSTM",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare standard and regime-aware LSTMs against statistical "
            "volatility forecasting benchmarks."
        )
    )

    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N folds; useful for a smoke test.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Maximum epochs per LSTM fit. Validation loss controls stopping.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=12,
        help="Number of hidden LSTM units.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size.",
    )

    return parser.parse_args()


def load_inputs():
    """Load source data and validate existing baseline forecasts."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecasting dataset not found: {DATA_PATH}. "
            "Run scripts/build_dataset.py first."
        )

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Statistical predictions not found: {BASELINE_PATH}. "
            "Run scripts/compare_statistical_models.py first."
        )

    industry_returns = load_industry_returns(RAW_DATA_PATH)

    dataset = build_forecasting_dataset(
        industry_returns,
        horizon=HORIZON,
    )

    saved_dataset = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    )
    saved_dataset.index = pd.DatetimeIndex(saved_dataset.index)

    if not saved_dataset.index.equals(dataset.index):
        raise ValueError(
            "The saved dataset index differs from the reconstructed dataset."
        )

    for column in ("portfolio_return", "target_volatility_5d"):
        if not np.allclose(
            saved_dataset[column].to_numpy(dtype=float),
            dataset[column].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Saved dataset column {column!r} differs from source data."
            )

    baselines = pd.read_csv(
        BASELINE_PATH,
        index_col=0,
        parse_dates=True,
    )
    baselines.index = pd.DatetimeIndex(baselines.index)
    baselines = baselines.sort_index()

    missing = set(["actual", *STATISTICAL_MODELS]).difference(
        baselines.columns
    )
    if missing:
        raise ValueError(
            f"Baseline file is missing columns: {sorted(missing)}"
        )

    splits = generate_walk_forward_splits(
        n_samples=len(dataset),
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        final_test_size=FINAL_TEST_SIZE,
        embargo=EMBARGO,
    )

    expected_dates = pd.DatetimeIndex(
        np.concatenate(
            [dataset.index[split.test].to_numpy() for split in splits]
        )
    )

    if not baselines.index.equals(expected_dates):
        raise ValueError(
            "Baseline forecast dates do not match the configured test period."
        )

    if not np.allclose(
        baselines["actual"].to_numpy(dtype=float),
        dataset.loc[expected_dates, "target_volatility_5d"].to_numpy(
            dtype=float
        ),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Baseline targets differ from reconstructed targets.")

    if not np.isfinite(
        baselines[["actual", *STATISTICAL_MODELS]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Baseline forecasts contain invalid values.")

    return industry_returns, dataset, baselines, splits


def build_regime_one_hot(
    regime_features: pd.DataFrame,
    training_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, MarketRegimeDetector]:
    """
    Fit K-means on training features and classify observations with the
    frozen model. Convert arbitrary cluster IDs to volatility-ranked
    one-hot indicators.
    """
    training_features = regime_features.loc[training_dates].dropna()

    if len(training_features) < 3:
        raise ValueError("Too few complete training rows for regime detection.")

    detector = MarketRegimeDetector(
        n_regimes=3,
        random_state=42,
    ).fit(training_features)

    centers = detector.cluster_centers()

    volatility_column = "portfolio_volatility_21d"
    if volatility_column not in centers.columns:
        raise ValueError(f"Missing feature: {volatility_column}")

    ordered_ids = centers[volatility_column].sort_values(
        kind="stable"
    ).index.tolist()

    if len(ordered_ids) != len(REGIME_NAMES):
        raise ValueError("Unexpected number of fitted regimes.")

    id_to_regime = dict(zip(ordered_ids, REGIME_NAMES))

    columns = [f"regime_{name}" for name in REGIME_NAMES]

    # Unavailable warm-up or invalid rows remain NaN; they are not
    # silently turned into a real regime.
    one_hot = pd.DataFrame(
        np.nan,
        index=regime_features.index,
        columns=columns,
        dtype=float,
    )

    valid_mask = regime_features.notna().all(axis=1)
    complete_features = regime_features.loc[valid_mask]

    labels = detector.predict(complete_features)
    ranked_labels = labels.map(id_to_regime)

    for regime in REGIME_NAMES:
        one_hot.loc[complete_features.index, f"regime_{regime}"] = (
            ranked_labels.eq(regime).astype(float).to_numpy()
        )

    return one_hot, detector


def make_model_config(
    fold_number: int,
    epochs: int,
    hidden_size: int,
    batch_size: int,
) -> LSTMConfig:
    return LSTMConfig(
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=min(4, epochs),
        random_state=42 + fold_number,
    )


def evaluate_model(
    model_features: pd.DataFrame,
    targets: pd.Series,
    split,
    config: LSTMConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train and validation-select an LSTM, then predict the test window."""
    x_train, y_train, train_dates = create_lstm_sequences(
        model_features,
        targets,
        target_positions=split.train,
        lookback=LOOKBACK,
    )

    x_validation, y_validation, validation_dates = create_lstm_sequences(
        model_features,
        targets,
        target_positions=split.validation,
        lookback=LOOKBACK,
    )

    x_test, y_test, test_dates = create_lstm_sequences(
        model_features,
        targets,
        target_positions=split.test,
        lookback=LOOKBACK,
    )

    expected_train_dates = targets.index[split.train]
    expected_validation_dates = targets.index[split.validation]
    expected_test_dates = targets.index[split.test]

    if not train_dates.equals(expected_train_dates):
        raise RuntimeError("Training sequence dates are misaligned.")

    if not validation_dates.equals(expected_validation_dates):
        raise RuntimeError("Validation sequence dates are misaligned.")

    if not test_dates.equals(expected_test_dates):
        raise RuntimeError("Test sequence dates are misaligned.")

    forecaster = LSTMVolatilityForecaster(config)

    forecaster.fit(
        training_sequences=x_train,
        training_targets=y_train,
        validation_sequences=x_validation,
        validation_targets=y_validation,
    )

    forecasts = forecaster.predict(x_test)

    print(
        f"    best epoch: {forecaster.best_epoch_}, "
        f"validation loss: {forecaster.best_validation_loss_:.6f}"
    )

    return forecasts, y_test, test_dates


def main() -> None:
    args = parse_arguments()

    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("--max-folds must be a positive integer.")

    if args.epochs < 1:
        raise ValueError("--epochs must be a positive integer.")

    if args.hidden_size < 1:
        raise ValueError("--hidden-size must be a positive integer.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")

    # This project currently targets CPU-compatible execution.
    torch.set_num_threads(1)

    industry_returns, dataset, baselines, all_splits = load_inputs()

    splits = (
        all_splits[:args.max_folds]
        if args.max_folds is not None
        else all_splits
    )

    portfolio_targets = dataset["target_volatility_5d"].astype(float)

    regime_features = build_regime_features(
        industry_returns,
        window=21,
    ).reindex(dataset.index)

    raw_inputs = industry_returns.reindex(dataset.index)

    test_frames = []

    print(f"Device: CPU")
    print(f"Dataset observations: {len(dataset):,}")
    print(f"Folds selected: {len(splits)} / {len(all_splits)}")
    print(f"Lookback: {LOOKBACK} observations")
    print(f"Max epochs per model: {args.epochs}")
    print(f"Hidden size: {args.hidden_size}")
    print("\nTraining the two LSTM specifications...")

    for split in splits:
        train_dates = dataset.index[split.train]
        validation_dates = dataset.index[split.validation]
        test_dates = dataset.index[split.test]

        print(
            f"\nFold {split.fold:03d}/{len(splits):03d}: "
            f"{test_dates[0].date()} to {test_dates[-1].date()}"
        )

        # Standardize industry returns using training observations only.
        input_scaler = StandardScaler()
        input_scaler.fit(raw_inputs.loc[train_dates])

        scaled_returns = pd.DataFrame(
            input_scaler.transform(raw_inputs),
            index=dataset.index,
            columns=raw_inputs.columns,
        )

        # Fit regime scaling and clustering using this fold's training
        # regime features only. Predictions for validation and test use
        # the same fitted detector without refitting it.
        regime_one_hot, detector = build_regime_one_hot(
            regime_features,
            train_dates,
        )

        standard_features = scaled_returns

        regime_aware_features = pd.concat(
            [scaled_returns, regime_one_hot],
            axis=1,
        )

        config = make_model_config(
            fold_number=split.fold,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
        )

        print("  Fitting standard LSTM...")
        standard_forecasts, y_test, forecast_dates = evaluate_model(
            standard_features,
            portfolio_targets,
            split,
            config,
        )

        print("  Fitting regime-aware LSTM...")
        regime_forecasts, regime_y_test, regime_dates = evaluate_model(
            regime_aware_features,
            portfolio_targets,
            split,
            config,
        )

        if not forecast_dates.equals(regime_dates):
            raise RuntimeError("LSTM variants use different test dates.")

        if not np.array_equal(y_test, regime_y_test):
            raise RuntimeError("LSTM variants use different targets.")

        if not forecast_dates.equals(test_dates):
            raise RuntimeError("LSTM predictions do not match fold test dates.")

        # Verify that all three input indicators are available on test dates.
        test_regimes = regime_one_hot.loc[test_dates]

        if test_regimes.isna().any().any():
            raise ValueError(
                f"Fold {split.fold} has unavailable test regime indicators."
            )

        # Preserve the already-computed statistical forecasts so all
        # models are compared on precisely the same target dates.
        fold_frame = baselines.loc[
            test_dates,
            ["actual", *STATISTICAL_MODELS],
        ].copy()

        if not np.allclose(
            fold_frame["actual"].to_numpy(dtype=float),
            y_test.astype(float),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                f"Fold {split.fold}: target values are inconsistent."
            )

        fold_frame["LSTM"] = standard_forecasts
        fold_frame["Regime-aware LSTM"] = regime_forecasts
        fold_frame["fold"] = split.fold

        # Retain the regime label for subsequent descriptive analysis.
        fold_frame["regime"] = (
            test_regimes.to_numpy().argmax(axis=1)
        )

        fold_frame["regime"] = fold_frame["regime"].map(
            {0: "lower", 1: "middle", 2: "higher"}
        )

        test_frames.append(fold_frame)

    results = pd.concat(test_frames).sort_index()

    if results.index.has_duplicates:
        raise ValueError("Test folds produced duplicate dates.")

    expected_results_dates = pd.DatetimeIndex(
        np.concatenate(
            [dataset.index[split.test].to_numpy() for split in splits]
        )
    )

    if not results.index.equals(expected_results_dates):
        raise ValueError("Result dates differ from selected fold dates.")

    for column in ["actual", *ALL_MODELS]:
        if not np.isfinite(results[column].to_numpy(dtype=float)).all():
            raise ValueError(
                f"Results contain invalid values in column {column!r}."
            )

    overall_records = []

    for model_name in ALL_MODELS:
        overall_records.append(
            {
                "model": model_name,
                "n_test": len(results),
                **calculate_volatility_metrics(
                    results["actual"],
                    results[model_name],
                ),
            }
        )

    overall_metrics = pd.DataFrame(overall_records)

    fold_records = []

    for fold_number, fold_frame in results.groupby("fold", sort=True):
        for model_name in ALL_MODELS:
            fold_records.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "n_test": len(fold_frame),
                    **calculate_volatility_metrics(
                        fold_frame["actual"],
                        fold_frame[model_name],
                    ),
                }
            )

    fold_metrics = pd.DataFrame(fold_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "lstm_model_comparison_predictions.csv"
    metrics_path = OUTPUT_DIR / "lstm_model_comparison_metrics.csv"
    fold_metrics_path = OUTPUT_DIR / "lstm_model_comparison_fold_metrics.csv"

    results.to_csv(predictions_path, index_label="date")
    overall_metrics.to_csv(metrics_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)

    print("\nPOOLED OUT-OF-SAMPLE RESULTS")
    print("===========================")
    print(
        f"Predictions per model: {len(results)}"
    )
    print(
        f"Evaluation period: {results.index[0].date()} to "
        f"{results.index[-1].date()}"
    )
    print()
    print(
        overall_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    print("\nSaved files:")
    print(f"Predictions: {predictions_path.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Fold metrics: {fold_metrics_path.relative_to(PROJECT_ROOT)}")

    if len(splits) < len(all_splits):
        print(
            "\nSMOKE TEST ONLY: these results cover a subset of folds. "
            "Do not interpret them as final model-performance estimates."
        )
    else:
        print(
            "\nResearch note: these folds have already informed development. "
            "Use a separate untouched period for the final evaluation."
        )


if __name__ == "__main__":
    main()
