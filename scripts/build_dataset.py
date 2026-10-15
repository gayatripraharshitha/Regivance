
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_industry_returns
from src.data.portfolio import build_forecasting_dataset
from src.research_config import CONFIG


RAW_PATH = PROJECT_ROOT / CONFIG["data"]["raw_path"]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "portfolio_forecasting_dataset.csv"
)


def main():
    # Load the 30 historical industry return series.
    industry_returns = load_industry_returns(RAW_PATH)

    # Construct portfolio returns and the five-day forward target.
    dataset = build_forecasting_dataset(
        industry_returns,
        horizon=5,
    )

    # Save the processed dataset.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index_label="date")

    print("=== REGIVANCE PORTFOLIO DATASET ===")
    print(f"Industry observations: {len(industry_returns):,}")
    print(f"Industry columns: {len(industry_returns.columns)}")
    print(f"Forecasting observations: {len(dataset):,}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"First target date: {dataset.index.min().date()}")
    print(f"Last target date: {dataset.index.max().date()}")
    print("\nFirst five rows:")
    print(dataset.head())
    print("\nSummary statistics:")
    print(dataset.describe())


if __name__ == "__main__":
    main()
