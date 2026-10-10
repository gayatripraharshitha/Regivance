"""Source-specific loading functions will live here."""

from pathlib import Path

import pandas as pd


def load_industry_returns(path: str | Path) -> pd.DataFrame:
    """Load standardized industry returns.

    The Kenneth French parser is deliberately left for the first vertical slice,
    because its header/footer format must be validated against the retrieved file.
    """
    raise NotImplementedError("Implement after archiving and inspecting the source file.")
