
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_industry_returns
from src.data.portfolio import build_forecasting_dataset
from src.models.rolling_volatility import forecast_rolling_volatility


RAW_PATH = PROJECT_ROOT / "data" / "raw" / "30_Industry_Portfolios_Daily.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rolling_volatility_baseline.csv"
)

WINDOW = 21
HORIZON = 5


def main():
    # Load historical returns for all 30 industry portfolios.
    industry_returns = load_industry_returns(RAW_PATH)

    # Build the equal-weighted portfolio and its future target.
    dataset = build_forecasting_dataset(
        industry_returns,
        horizon=HORIZON,
    )

    # Generate a forecast using only returns available on each date.
    portfolio_returns = industry_returns.mean(axis=1).rename(
        "portfolio_return"
    )

    forecasts = forecast_rolling_volatility(
        portfolio_returns,
        window=WINDOW,
        horizon=HORIZON,
    )

    forecast_name = f"forecast_volatility_{HORIZON}d_rolling{WINDOW}d"

    # Align forecasts and realized targets by forecast date.
    results = dataset.join(forecasts.rename(forecast_name))

    # Early dates lack a complete 21-day history.
    results = results.dropna(
        subset=["target_volatility_5d", forecast_name]
    ).copy()

    forecast_values = results[forecast_name].to_numpy()
    target_values = results["target_volatility_5d"].to_numpy()

    # Initial descriptive scores; these are not our final test results.
    errors = forecast_values - target_values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    # QLIKE-style loss using realized variance and forecast variance.
    realized_variance = target_values**2
    forecast_variance = forecast_values**2
    ratio = realized_variance / forecast_variance
    qlike = np.mean(ratio - np.log(ratio) - 1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index_label="date")

    print("=== REGIVANCE: ROLLING-VOLATILITY BASELINE ===")
    print(f"Rolling window: {WINDOW} trading observations")
    print(f"Forecast horizon: {HORIZON} trading observations")
    print(f"Scored observations: {len(results):,}")
    print(f"First scored date: {results.index.min().date()}")
    print(f"Last scored date: {results.index.max().date()}")
    print("\nPreliminary scores (decimal-return units):")
    print(f"MAE:   {mae:.8f}")
    print(f"RMSE:  {rmse:.8f}")
    print(f"QLIKE: {qlike:.8f}")
    print(f"\nForecasts saved to: {OUTPUT_PATH}")
    print("\nFirst five scored observations:")
    print(results.head())


if __name__ == "__main__":
    main()
