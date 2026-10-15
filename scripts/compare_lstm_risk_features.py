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

from scripts.compare_lstm_models import (
    LOOKBACK,
    OUTPUT_DIR,
    STATISTICAL_MODELS,
    build_regime_one_hot,
    evaluate_model,
    load_inputs,
    make_model_config,
)
from src.evaluation.metrics import calculate_volatility_metrics
from src.features.rolling_features import build_regime_features


MODEL_NAMES = [
    *STATISTICAL_MODELS,
    "LSTM",
    "Regime-aware LSTM",
    "Risk-feature LSTM",
    "Risk-feature + regime LSTM",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LSTM volatility forecasts with continuous risk "
            "features and regime indicators."
        )
    )

    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N folds for a smoke test.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Maximum training epochs per LSTM.",
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


def scale_returns(
    raw_returns: pd.DataFrame,
    training_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Scale returns using training observations only."""
    scaler = StandardScaler()
    scaler.fit(raw_returns.loc[training_dates])

    scaled = pd.DataFrame(
        scaler.transform(raw_returns),
        index=raw_returns.index,
        columns=raw_returns.columns,
    )

    if not np.isfinite(scaled.to_numpy(dtype=float)).all():
        raise ValueError("Scaled return features contain invalid values.")

    return scaled


def scale_risk_features(
    risk_features: pd.DataFrame,
    training_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Standardize risk features using training observations only.

    Initial rolling-window NaNs are preserved. Any missing values after
    the first complete feature row cause an explicit error.
    """
    if risk_features.empty:
        raise ValueError("Risk features must not be empty.")

    complete_mask = risk_features.notna().all(axis=1)

    if not complete_mask.any():
        raise ValueError("No complete risk-feature rows are available.")

    first_valid_position = int(
        np.flatnonzero(complete_mask.to_numpy())[0]
    )

    # Only the initial rolling-window warm-up may be incomplete.
    if not complete_mask.iloc[first_valid_position:].all():
        bad_dates = risk_features.index[
            first_valid_position:
        ][~complete_mask.iloc[first_valid_position:].to_numpy()]

        raise ValueError(
            "Risk features contain missing values after their initial "
            f"warm-up. First affected date: {bad_dates[0]}"
        )

    training_features = risk_features.loc[training_dates]

    if training_features.isna().any().any():
        raise ValueError(
            "Training dates contain incomplete risk features."
        )

    if not np.isfinite(
        training_features.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Training risk features contain invalid numeric values."
        )

    scaler = StandardScaler()
    scaler.fit(training_features)

    scaled = pd.DataFrame(
        np.nan,
        index=risk_features.index,
        columns=risk_features.columns,
        dtype=float,
    )

    valid_dates = risk_features.index[complete_mask]

    scaled.loc[valid_dates] = scaler.transform(
        risk_features.loc[valid_dates]
    )

    return scaled


def build_model_inputs(
    scaled_returns: pd.DataFrame,
    scaled_risk_features: pd.DataFrame,
    regime_one_hot: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create four controlled input specifications."""
    return {
        "LSTM": scaled_returns,
        "Regime-aware LSTM": pd.concat(
            [scaled_returns, regime_one_hot],
            axis=1,
        ),
        "Risk-feature LSTM": pd.concat(
            [scaled_returns, scaled_risk_features],
            axis=1,
        ),
        "Risk-feature + regime LSTM": pd.concat(
            [
                scaled_returns,
                scaled_risk_features,
                regime_one_hot,
            ],
            axis=1,
        ),
    }


def validate_model_dates(
    model_name: str,
    forecast_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    forecasts: np.ndarray,
    expected_targets: np.ndarray,
    model_targets: np.ndarray,
) -> None:
    if not forecast_dates.equals(test_dates):
        raise RuntimeError(
            f"{model_name}: forecast dates do not match the test dates."
        )

    if not np.array_equal(expected_targets, model_targets):
        raise RuntimeError(
            f"{model_name}: target values differ from the other models."
        )

    if not np.isfinite(np.asarray(forecasts, dtype=float)).all():
        raise ValueError(
            f"{model_name}: predictions contain invalid values."
        )

    if (np.asarray(forecasts, dtype=float) <= 0).any():
        raise ValueError(
            f"{model_name}: predictions must be positive."
        )


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

    # Keep execution compatible with the project's CPU setup.
    torch.set_num_threads(1)

    industry_returns, dataset, baselines, all_splits = load_inputs()

    splits = (
        all_splits[:args.max_folds]
        if args.max_folds is not None
        else all_splits
    )

    targets = dataset["target_volatility_5d"].astype(float)

    # These features use information available through each date.
    raw_risk_features = build_regime_features(
        industry_returns,
        window=21,
    ).reindex(dataset.index)

    raw_returns = industry_returns.reindex(dataset.index)

    # Validate that the reconstructed feature dates align.
    if not raw_returns.index.equals(dataset.index):
        raise RuntimeError("Return feature dates are misaligned.")

    if not raw_risk_features.index.equals(dataset.index):
        raise RuntimeError("Risk feature dates are misaligned.")

    test_frames = []

    print("Device: CPU")
    print(f"Dataset observations: {len(dataset):,}")
    print(f"Folds selected: {len(splits)} / {len(all_splits)}")
    print(f"Lookback: {LOOKBACK} observations")
    print(f"Maximum epochs: {args.epochs}")
    print(f"Hidden size: {args.hidden_size}")
    print("\nFour LSTM input specifications will be compared.")

    for split in splits:
        train_dates = dataset.index[split.train]
        test_dates = dataset.index[split.test]

        print(
            f"\nFold {split.fold:03d}/{len(splits):03d}: "
            f"{test_dates[0].date()} to {test_dates[-1].date()}"
        )

        # Fit both scalers only on this fold's training observations.
        scaled_returns = scale_returns(
            raw_returns,
            train_dates,
        )

        scaled_risk_features = scale_risk_features(
            raw_risk_features,
            train_dates,
        )

        # Fit clustering and regime feature scaling on training data only.
        # Validation/test dates use the frozen detector.
        regime_one_hot, _ = build_regime_one_hot(
            raw_risk_features,
            train_dates,
        )

        test_regimes = regime_one_hot.loc[test_dates]

        if test_regimes.isna().any().any():
            raise ValueError(
                f"Fold {split.fold} has unavailable test regime indicators."
            )

        feature_sets = build_model_inputs(
            scaled_returns,
            scaled_risk_features,
            regime_one_hot,
        )

        config = make_model_config(
            fold_number=split.fold,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
        )

        predictions: dict[str, np.ndarray] = {}
        common_targets: np.ndarray | None = None

        for model_name, model_features in feature_sets.items():
            print(f"  Fitting {model_name}...")

            forecasts, model_targets, forecast_dates = evaluate_model(
                model_features,
                targets,
                split,
                config,
            )

            if common_targets is None:
                common_targets = model_targets.copy()

            validate_model_dates(
                model_name=model_name,
                forecast_dates=forecast_dates,
                test_dates=test_dates,
                forecasts=forecasts,
                expected_targets=common_targets,
                model_targets=model_targets,
            )

            predictions[model_name] = forecasts

        if common_targets is None:
            raise RuntimeError("No LSTM predictions were generated.")

        # Start with the previously generated statistical forecasts.
        fold_frame = baselines.loc[
            test_dates,
            ["actual", *STATISTICAL_MODELS],
        ].copy()

        if not np.allclose(
            fold_frame["actual"].to_numpy(dtype=float),
            common_targets.astype(float),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                f"Fold {split.fold}: LSTM and statistical targets differ."
            )

        for model_name, forecasts in predictions.items():
            fold_frame[model_name] = forecasts

        fold_frame["fold"] = split.fold

        # Assign interpretable labels based on the existing one-hot columns.
        regime_ids = test_regimes.to_numpy(dtype=float).argmax(axis=1)

        fold_frame["regime"] = pd.Series(
            regime_ids,
            index=test_dates,
        ).map(
            {
                0: "lower",
                1: "middle",
                2: "higher",
            }
        )

        test_frames.append(fold_frame)

    if not test_frames:
        raise RuntimeError("No test folds were evaluated.")

    results = pd.concat(test_frames).sort_index()

    if results.index.has_duplicates:
        raise ValueError("Test folds produced duplicate dates.")

    expected_dates = pd.DatetimeIndex(
        np.concatenate(
            [
                dataset.index[split.test].to_numpy()
                for split in splits
            ]
        )
    )

    if not results.index.equals(expected_dates):
        raise ValueError(
            "Result dates differ from the configured test dates."
        )

    for column in ["actual", *MODEL_NAMES]:
        values = results[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Results contain invalid values in {column!r}."
            )

    overall_records = []

    for model_name in MODEL_NAMES:
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
        for model_name in MODEL_NAMES:
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

    predictions_path = (
        OUTPUT_DIR / "lstm_risk_feature_comparison_predictions.csv"
    )
    metrics_path = (
        OUTPUT_DIR / "lstm_risk_feature_comparison_metrics.csv"
    )
    fold_metrics_path = (
        OUTPUT_DIR / "lstm_risk_feature_comparison_fold_metrics.csv"
    )

    # These filenames are separate from the original LSTM experiment.
    results.to_csv(predictions_path, index_label="date")
    overall_metrics.to_csv(metrics_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)

    print("\nPOOLED OUT-OF-SAMPLE RESULTS")
    print("===========================")
    print(f"Predictions per model: {len(results)}")
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
    print(predictions_path.relative_to(PROJECT_ROOT))
    print(metrics_path.relative_to(PROJECT_ROOT))
    print(fold_metrics_path.relative_to(PROJECT_ROOT))

    if len(splits) < len(all_splits):
        print(
            "\nSMOKE TEST ONLY: the results use a subset of folds. "
            "Do not interpret them as final performance estimates."
        )
    else:
        print(
            "\nResearch note: these folds have informed model development. "
            "Use a separate untouched period for final evaluation."
        )


if __name__ == "__main__":
    main()