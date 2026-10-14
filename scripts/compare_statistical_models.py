
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Make the project root importable when the script runs from PowerShell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_volatility_metrics
from src.evaluation.splits import generate_walk_forward_splits
from src.models.ewma import forecast_ewma_volatility
from src.models.garch import forecast_garch_volatility_for_fold
from src.models.rolling_volatility import forecast_rolling_volatility


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "portfolio_forecasting_dataset.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_SIZE = 1260
VALIDATION_SIZE = 252
TEST_SIZE = 21
STEP_SIZE = 21
FINAL_TEST_SIZE = 2520
EMBARGO = 5
HORIZON = 5

# Each tuple contains: model name, innovation distribution, asymmetric flag.
GARCH_VARIANTS = [
    ("GARCH-normal", "normal", False),
    ("GARCH-Student-t", "t", False),
    ("GJR-GARCH-Student-t", "t", True),
]


def load_dataset() -> pd.DataFrame:
    """Load and validate the portfolio forecasting dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run scripts/build_dataset.py first."
        )

    data = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    )

    data.index = pd.DatetimeIndex(data.index)
    data = data.sort_index()

    required_columns = {
        "portfolio_return",
        "target_volatility_5d",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    if data.index.has_duplicates:
        raise ValueError("Dataset contains duplicate dates.")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Dataset dates must be chronological.")

    for column in required_columns:
        values = data[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Column {column!r} contains missing or non-finite values."
            )

    if (data["target_volatility_5d"] <= 0).any():
        raise ValueError("Target volatility must be positive.")

    return data


def main() -> None:
    data = load_dataset()

    portfolio_returns = data["portfolio_return"].astype(float)
    actual_volatility = data["target_volatility_5d"].astype(float)

    # Existing fixed-parameter baselines.
    rolling_forecast = forecast_rolling_volatility(
        portfolio_returns,
        window=21,
        horizon=HORIZON,
    ).rename("Rolling")

    ewma_forecast = forecast_ewma_volatility(
        portfolio_returns,
        decay=0.94,
        horizon=HORIZON,
        min_periods=21,
    ).rename("EWMA")

    # Generate chronological walk-forward evaluation folds.
    splits = generate_walk_forward_splits(
        n_samples=len(data),
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        final_test_size=FINAL_TEST_SIZE,
        embargo=EMBARGO,
    )

    # Allocate a separate out-of-sample prediction series for each
    # GARCH specification.
    garch_forecasts = {
        name: pd.Series(
            np.nan,
            index=data.index,
            name=name,
            dtype=float,
        )
        for name, _, _ in GARCH_VARIANTS
    }

    print(f"Dataset observations: {len(data):,}")
    print(f"Walk-forward test folds: {len(splits)}")
    print(f"Test observations per fold: {TEST_SIZE}")
    print(f"Expected test observations per model: {len(splits) * TEST_SIZE}")
    print(f"Embargo: {EMBARGO} observations")
    print(f"GARCH specifications: {len(GARCH_VARIANTS)}")
    print("\nFitting GARCH variants separately within each fold...")

    # Fit each model on that fold's training data only.
    # Later observed returns update the conditional variance state,
    # while the fitted parameters remain fixed within the test block.
    for fold_number, split in enumerate(splits, start=1):
        training_dates = data.index[split.train]
        test_dates = pd.DatetimeIndex(data.index[split.test])

        training_returns = portfolio_returns.loc[training_dates]

        for model_name, distribution, asymmetric in GARCH_VARIANTS:
            try:
                fold_forecast = forecast_garch_volatility_for_fold(
                    training_returns=training_returns,
                    all_returns=portfolio_returns,
                    forecast_dates=test_dates,
                    horizon=HORIZON,
                    distribution=distribution,
                    asymmetric=asymmetric,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{model_name} failed on fold "
                    f"{fold_number}/{len(splits)} "
                    f"({test_dates[0].date()} to {test_dates[-1].date()})."
                ) from exc

            fold_forecast = fold_forecast.reindex(test_dates)
            values = fold_forecast.to_numpy(dtype=float)

            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(
                    f"{model_name} produced invalid forecasts "
                    f"on fold {fold_number}."
                )

            garch_forecasts[model_name].loc[test_dates] = fold_forecast

        print(
            f"Fold {fold_number:03d}/{len(splits):03d}: "
            f"{test_dates[0].date()} to {test_dates[-1].date()}"
        )

    # Extract exactly the same test dates for every model.
    test_indices = np.concatenate(
        [
            np.arange(split.test.start, split.test.stop)
            for split in splits
        ]
    )

    test_dates = pd.DatetimeIndex(data.index[test_indices])

    model_names = [
        "Rolling",
        "EWMA",
        *[name for name, _, _ in GARCH_VARIANTS],
    ]

    predictions = pd.DataFrame(
        {
            "actual": actual_volatility.reindex(test_dates),
            "Rolling": rolling_forecast.reindex(test_dates),
            "EWMA": ewma_forecast.reindex(test_dates),
            **{
                name: series.reindex(test_dates)
                for name, series in garch_forecasts.items()
            },
        },
        index=test_dates,
    )

    predictions.index.name = "date"

    if predictions.isna().any().any():
        missing_counts = predictions.isna().sum()
        raise ValueError(
            "Missing values in test predictions:\n"
            f"{missing_counts[missing_counts > 0]}"
        )

    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Predictions contain non-finite values.")

    # Calculate pooled out-of-sample metrics over identical dates.
    overall_records = []

    for model_name in model_names:
        scores = calculate_volatility_metrics(
            predictions["actual"],
            predictions[model_name],
        )

        overall_records.append(
            {
                "model": model_name,
                "n_test": len(predictions),
                **scores,
            }
        )

    overall_metrics = pd.DataFrame(overall_records)

    # Calculate metrics separately for every 21-observation test fold.
    fold_records = []

    for fold_number, split in enumerate(splits, start=1):
        fold_dates = pd.DatetimeIndex(data.index[split.test])
        fold_data = predictions.loc[fold_dates]

        for model_name in model_names:
            scores = calculate_volatility_metrics(
                fold_data["actual"],
                fold_data[model_name],
            )

            fold_records.append(
                {
                    "fold": fold_number,
                    "start_date": fold_dates[0].date(),
                    "end_date": fold_dates[-1].date(),
                    "model": model_name,
                    "n_test": len(fold_data),
                    **scores,
                }
            )

    fold_metrics = pd.DataFrame(fold_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        OUTPUT_DIR / "walk_forward_statistical_models_predictions.csv"
    )

    metrics_path = (
        OUTPUT_DIR / "statistical_model_comparison_metrics.csv"
    )

    fold_metrics_path = (
        OUTPUT_DIR / "statistical_model_comparison_fold_metrics.csv"
    )

    predictions.to_csv(predictions_path)
    overall_metrics.to_csv(metrics_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)

    # Print pooled results.
    print("\nPOOLED OUT-OF-SAMPLE RESULTS")
    print("===========================")
    print(
        f"Evaluation period: {test_dates[0].date()} to "
        f"{test_dates[-1].date()}"
    )
    print(f"Total predictions per model: {len(predictions)}")
    print()

    print(
        overall_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    # Count the number of folds each model wins for each metric.
    print("\nFOLD WINS (lower is better)")
    print("===========================")

    for metric in ("mae", "rmse", "qlike"):
        pivot = fold_metrics.pivot(
            index="fold",
            columns="model",
            values=metric,
        )

        winners = pivot.idxmin(axis=1).value_counts()
        winners = winners.reindex(model_names, fill_value=0)

        print(f"\n{metric.upper()}:")

        for model_name, count in winners.items():
            print(f"  {model_name}: {count}/{len(splits)}")

    print("\nSaved files:")
    print(f"Predictions: {predictions_path.relative_to(PROJECT_ROOT)}")
    print(f"Overall metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Fold metrics: {fold_metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
