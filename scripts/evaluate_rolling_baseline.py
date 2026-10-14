
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_industry_returns
from src.data.portfolio import build_forecasting_dataset
from src.models.rolling_volatility import forecast_rolling_volatility
from src.evaluation.splits import generate_walk_forward_splits
from src.evaluation.metrics import calculate_volatility_metrics


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "30_Industry_Portfolios_Daily.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

HORIZON = 5
ROLLING_WINDOW = 21
FINAL_TEST_SIZE = 252
TEST_BLOCK_SIZE = 21
EMBARGO = 5


def main():
    # 1. Load the industry returns and construct the forecasting dataset.
    industry_returns = load_industry_returns(RAW_PATH)

    dataset = build_forecasting_dataset(
        industry_returns,
        horizon=HORIZON,
    )

    # 2. Generate baseline forecasts using only historical returns.
    portfolio_returns = industry_returns.mean(axis=1).rename(
        "portfolio_return"
    )

    forecast_name = "rolling_baseline_forecast"

    forecasts = forecast_rolling_volatility(
        portfolio_returns,
        window=ROLLING_WINDOW,
        horizon=HORIZON,
    ).rename(forecast_name)

    data = dataset.join(forecasts)

    # 3. Define the final evaluation period using chronological folds.
    splits = generate_walk_forward_splits(
        n_samples=len(data),
        train_size=1260,
        validation_size=252,
        test_size=TEST_BLOCK_SIZE,
        step_size=TEST_BLOCK_SIZE,
        final_test_size=FINAL_TEST_SIZE,
        embargo=EMBARGO,
    )

    prediction_frames = []
    fold_metric_rows = []

    # 4. Score each test block independently.
    for split in splits:
        fold_data = data.iloc[split.test].copy()

        if fold_data[forecast_name].isna().any():
            raise ValueError(
                f"Missing forecasts in test fold {split.fold}."
            )

        metrics = calculate_volatility_metrics(
            actual=fold_data["target_volatility_5d"],
            forecast=fold_data[forecast_name],
        )

        fold_data["fold"] = split.fold
        prediction_frames.append(fold_data)

        fold_metric_rows.append(
            {
                "fold": split.fold,
                "train_start": data.index[split.train.start],
                "train_end": data.index[split.train.stop - 1],
                "validation_start": data.index[split.validation.start],
                "validation_end": data.index[split.validation.stop - 1],
                "test_start": fold_data.index.min(),
                "test_end": fold_data.index.max(),
                "n_test": len(fold_data),
                **metrics,
            }
        )

    # 5. Combine all test blocks and calculate pooled out-of-sample metrics.
    predictions = pd.concat(prediction_frames).sort_index()

    if predictions.index.has_duplicates:
        raise ValueError("Test predictions contain duplicate dates.")

    if len(predictions) != FINAL_TEST_SIZE:
        raise ValueError(
            f"Expected {FINAL_TEST_SIZE} test predictions, "
            f"received {len(predictions)}."
        )

    overall_metrics = calculate_volatility_metrics(
        actual=predictions["target_volatility_5d"],
        forecast=predictions[forecast_name],
    )

    fold_metrics = pd.DataFrame(fold_metric_rows)

    # 6. Save the predictions and fold-level metrics.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = (
        OUTPUT_DIR / "walk_forward_rolling_baseline_predictions.csv"
    )
    metrics_path = (
        OUTPUT_DIR / "walk_forward_rolling_baseline_metrics.csv"
    )

    predictions.to_csv(predictions_path, index_label="date")
    fold_metrics.to_csv(metrics_path, index=False)

    print("=== REGIVANCE WALK-FORWARD BASELINE ===")
    print(f"Total usable targets: {len(data):,}")
    print(f"Number of test folds: {len(splits)}")
    print(f"Test observations per fold: {TEST_BLOCK_SIZE}")
    print(f"Embargo: {EMBARGO} observations")
    print(f"Total out-of-sample predictions: {len(predictions)}")
    print(
        f"Evaluation period: {predictions.index.min().date()} "
        f"to {predictions.index.max().date()}"
    )

    print("\nPooled out-of-sample metrics:")
    for name, value in overall_metrics.items():
        print(f"{name.upper()}: {value:.8f}")

    print("\nMetrics by test fold:")
    print(
        fold_metrics[["fold", "test_start", "test_end",
                      "n_test", "mae", "rmse", "qlike"]].to_string(
            index=False
        )
    )

    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Fold metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
