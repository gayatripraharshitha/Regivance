"""Load and transform Kenneth French 30-industry daily returns."""

from io import StringIO
from pathlib import Path

import pandas as pd

from src.data.validation import validate_returns_frame


SOURCE_MISSING_VALUES = (-99.99, -999.0)


def _value_weighted_csv_block(lines: list[str]) -> str:
    """Return the CSV block beneath the value-weighted daily heading."""
    marker = "Average Value Weighted Returns -- Daily"
    try:
        marker_index = next(i for i, line in enumerate(lines) if marker in line)
    except StopIteration as error:
        raise ValueError(f"Could not find '{marker}' in the source file.") from error

    header_index = marker_index + 1
    while header_index < len(lines) and not lines[header_index].lstrip().startswith(","
    ):
        header_index += 1
    if header_index == len(lines):
        raise ValueError("Could not find the value-weighted CSV header.")

    block = [lines[header_index]]
    for line in lines[header_index + 1 :]:
        if not line.strip():
            break
        if not line.split(",", 1)[0].strip().isdigit():
            break
        block.append(line)
    if len(block) == 1:
        raise ValueError("The value-weighted section has no observations.")
    return "".join(block)


def load_industry_returns(path: str | Path) -> pd.DataFrame:
    """Load the value-weighted daily 30-industry table as decimal returns.

    Kenneth French files contain descriptive text and two return tables. This
    parser selects only the first, value-weighted daily table. Source values are
    percentages and are converted to decimal simple returns.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Raw source file not found: {source_path}")
    lines = source_path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    raw = pd.read_csv(StringIO(_value_weighted_csv_block(lines)), na_values=SOURCE_MISSING_VALUES)
    date_column = raw.columns[0]
    dates = pd.to_datetime(raw.pop(date_column).astype(str).str.strip(), format="%Y%m%d")
    returns = raw.apply(pd.to_numeric, errors="raise") / 100.0
    returns.columns = returns.columns.str.strip()
    returns.index = pd.DatetimeIndex(dates, name="date")
    validate_returns_frame(returns)
    return returns


def equal_weight_portfolio_return(industry_returns: pd.DataFrame) -> pd.Series:
    """Construct a daily return series with one equal weight per industry."""
    validate_returns_frame(industry_returns)
    return industry_returns.mean(axis=1).rename("portfolio_return")


def forward_realized_volatility(returns: pd.Series, horizon_days: int) -> pd.Series:
    """Compute sqrt(mean(r²)) over days t+1 through t+h for every origin t."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns must have a DatetimeIndex.")
    squared = returns.astype(float).pow(2)
    future_mean_square = squared.shift(-1).rolling(horizon_days).mean().shift(-(horizon_days - 1))
    return future_mean_square.pow(0.5).rename(f"forward_realized_volatility_{horizon_days}d")


def build_modeling_frame(
    industry_returns: pd.DataFrame, horizon_days: int
) -> pd.DataFrame:
    """Combine raw industry returns, portfolio return, and future-only target."""
    portfolio = equal_weight_portfolio_return(industry_returns)
    target = forward_realized_volatility(portfolio, horizon_days)
    return pd.concat([industry_returns, portfolio, target], axis=1)
