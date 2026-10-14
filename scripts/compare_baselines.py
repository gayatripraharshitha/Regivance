
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_industry_returns
from src.data.portfolio import build_forecasting_dataset
from src.models.rolling_volatility import forecast_rolling_volatility
from src.models.ewma import forecast_ewma_volatility
from src.evaluation.splits import generate_walk_forward_splits
from src.evaluation.metrics import calculate_volatility_metrics


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "30_Industry_Portfolios_Daily.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

HORIZON = 5
ROLLING_WINDOW = 21
EWMA_DECAY = 0.94
FINAL_TEST_SIZE = 252
TEST_BLOCK_SIZE = 21
EMBARGO = 5

ROLLING_COLUMN = "rolling_21d_forecast"
EWMA_COLUMN = "ewma_094_forecast"


def main():
    # 1. Load industry returns and construct the portfolio target.
    industry_returns = load_industry_returns(RAW_PATH)

    dataset = build_forecasting_dataset(
        industry_returns,
        horizon=HORIZON,
    )

    portfolio_returns = industry_returns.mean(axis=1).rename(
        "portfolio_return"
    )

    # 2. Generate both forecasts using information available on each date.
    rolling_forecast = forecast_rolling_volatility(
        portfolio_returns,
        window=ROLLING_WINDOW,
        horizon=HORIZON,
    ).rename(ROLLING_COLUMN)

    ewma_forecast = forecast_ewma_volatility(
        portfolio_returns,
        decay=EWMA_DECAY,
        horizon=HORIZON,
        min_periods=ROLLING_WINDOW,
    ).rename(EWMA_COLUMN)

    # All columns are aligned by forecast date.
    data = dataset.join([rolling_forecast, ewma_forecast])

    # 3. Create identical chronological test dates for both models.
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

    # 4. Score each model on every test block.
    for split in splits:
        fold_data = data.iloc[split.test].copy()

        required_columns = [
            "target_volatility_5d",
            ROLLING_COLUMN,
            EWMA_COLUMN,
        ]

        if fold_data[required_columns].isna().any().any():
            raise ValueError(
                f"Missing targets or forecasts in fold {split.fold}."
            )

        fold_data["fold"] = split.fold
        prediction_frames.append(fold_data)

        for model_name, forecast_column in [
            ("Rolling volatility", ROLLING_COLUMN),
            ("EWMA", EWMA_COLUMN),
        ]:
            scores = calculate_volatility_metrics(
                actual=fold_data["target_volatility_5d"],
                forecast=fold_data[forecast_column],
            )

            fold_metric_rows.append(
                {
                    "fold": split.fold,
                    "model": model_name,
                    "test_start": fold_data.index.min(),
                    "test_end": fold_data.index.max(),
                    "n_test": len(fold_data),
                    **scores,
                }
            )

    # 5. Combine test blocks and verify identical, unique dates.
    predictions = pd.concat(prediction_frames).sort_index()

    if predictions.index.has_duplicates:
        raise ValueError("Evaluation predictions contain duplicate dates.")

    if len(predictions) != FINAL_TEST_SIZE:
        raise ValueError(
            f"Expected {FINAL_TEST_SIZE} predictions per model; "
            f"received {len(predictions)} dates."
        )

    # 6. Calculate pooled metrics over exactly the same dates.
    overall_rows = []

    for model_name, forecast_column in [
        ("Rolling volatility", ROLLING_COLUMN),
        ("EWMA", EWMA_COLUMN),
    ]:
        scores = calculate_volatility_metrics(
            actual=predictions["target_volatility_5d"],
            forecast=predictions[forecast_column],
        )

        overall_rows.append(
            {
                "model": model_name,
                "n_test": len(predictions),
                **scores,
            }
        )

    overall_metrics = pd.DataFrame(overall_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    # 7. Save reproducible comparison outputs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "baseline_comparison_predictions.csv"
    overall_path = OUTPUT_DIR / "baseline_comparison_metrics.csv"
    fold_path = OUTPUT_DIR / "baseline_comparison_fold_metrics.csv"

    predictions.to_csv(predictions_path, index_label="date")
    overall_metrics.to_csv(overall_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)

    print("=== REGIVANCE BASELINE COMPARISON ===")
    print(f"Forecast horizon: {HORIZON} trading observations")
    print(f"Rolling window: {ROLLING_WINDOW}")
    print(f"EWMA decay: {EWMA_DECAY}")
    print(f"Test folds: {len(splits)}")
    print(f"Dates scored per model: {len(predictions)}")
    print(
        f"Evaluation period: {predictions.index.min().date()} "
        f"to {predictions.index.max().date()}"
    )

    print("\nPooled out-of-sample metrics (lower is better):")
    print(
        overall_metrics.to_string(
            index=False,
            formatters={
                "mae": "{:.8f}".format,
                "rmse": "{:.8f}".format,
                "qlike": "{:.8f}".format,
            },
        )
    )

    print("\nMetrics by model and test fold:")
    print(
        fold_metrics.to_string(
            index=False,
            formatters={
                "mae": "{:.6f}".format,
                "rmse": "{:.6f}".format,
                "qlike": "{:.6f}".format,
            },
        )
    )

    print("\nSaved outputs:")
    print(predictions_path)
    print(overall_path)
    print(fold_path)


if __name__ == "__main__":
    main()
