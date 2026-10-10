from io import StringIO
from pathlib import Path

import pandas as pd


def load_industry_returns(path: str | Path) -> pd.DataFrame:
    """
    Load daily value-weighted returns for the 30 Kenneth French industries.

    Returns:
        A DataFrame indexed by date, with decimal daily returns.
    """
    source_path = Path(path)

    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    # Read the source as text because it contains notes and multiple tables.
    lines = source_path.read_text(encoding="utf-8-sig").splitlines()

    marker = "Average Value Weighted Returns -- Daily"

    try:
        marker_index = next(
            index for index, line in enumerate(lines) if marker in line
        )
    except StopIteration:
        raise ValueError("Could not find the value-weighted daily returns section.")

    # Find the real CSV header, which begins with a blank date-column name.
    header_index = marker_index + 1

    while header_index < len(lines):
        if lines[header_index].lstrip().startswith(","):
            break
        header_index += 1

    if header_index == len(lines):
        raise ValueError("Could not find the CSV header.")

    # Collect only rows belonging to this table.
    csv_lines = [lines[header_index]]

    for line in lines[header_index + 1:]:
        if not line.strip():
            break

        first_value = line.split(",", 1)[0].strip()

        if not first_value.isdigit():
            break

        csv_lines.append(line)

    if len(csv_lines) == 1:
        raise ValueError("No return observations were found.")

    # Parse only the extracted CSV block.
    raw_returns = pd.read_csv(
        StringIO("\n".join(csv_lines)),
        na_values=[-99.99, -999],
    )

    # The first column is the date column.
    date_column = raw_returns.columns[0]

    dates = pd.to_datetime(
        raw_returns.pop(date_column).astype(str).str.strip(),
        format="%Y%m%d",
    )

    # Convert source percentages to decimal returns.
    returns = raw_returns.apply(pd.to_numeric, errors="raise") / 100

    # Clean industry names and attach dates as the index.
    returns.columns = returns.columns.str.strip()
    returns.index = pd.DatetimeIndex(dates, name="date")

    return returns