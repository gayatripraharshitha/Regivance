
from pathlib import Path
import sys

import pandas as pd

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_industry_returns
from src.research_config import CONFIG


DATA_PATH = PROJECT_ROOT / CONFIG["data"]["raw_path"]


def main():
    returns = load_industry_returns(DATA_PATH)

    print("=== REGIVANCE DATA VALIDATION ===")
    print(f"Rows: {len(returns):,}")
    print(f"Industry columns: {len(returns.columns)}")
    print(
        f"Date range: {returns.index.min().date()} to "
        f"{returns.index.max().date()}"
    )
    print(f"Duplicate dates: {returns.index.duplicated().sum()}")
    print(f"Missing values: {returns.isna().sum().sum()}")
    print(
        "Non-numeric columns:",
        list(returns.select_dtypes(exclude="number").columns),
    )
    print(
        "All returns finite:",
        bool(
            returns.apply(
                lambda column: column.map(
                    lambda value: pd.notna(value) and abs(value) != float("inf")
                )
            ).all().all()
        ),
    )

    print("\nReturn summary (decimal units):")
    print(returns.describe().loc[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
