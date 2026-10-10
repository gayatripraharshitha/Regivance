from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    equal_weight_portfolio_return,
    build_modeling_frame,
    forward_realized_volatility,
    load_industry_returns,
)


FIXTURE = Path("tests/fixtures/kenneth_french_30_industry_sample.csv")


def test_loader_selects_value_weighted_section_and_converts_percentages() -> None:
    returns = load_industry_returns(FIXTURE)
    assert returns.columns.tolist() == ["Food", "Beer"]
    assert returns.loc[pd.Timestamp("2020-01-02"), "Food"] == pytest.approx(0.01)
    assert returns.loc[pd.Timestamp("2020-01-03"), "Beer"] == pytest.approx(0.03)


def test_equal_weight_portfolio_return() -> None:
    returns = load_industry_returns(FIXTURE)
    portfolio = equal_weight_portfolio_return(returns)
    assert portfolio.iloc[0] == pytest.approx(-0.005)


def test_forward_realized_volatility_uses_only_future_returns() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=dates)
    target = forward_realized_volatility(returns, horizon_days=2)
    assert target.iloc[0] == pytest.approx(((0.02**2 + 0.03**2) / 2) ** 0.5)
    assert pd.isna(target.iloc[-2])


def test_modeling_frame_contains_portfolio_and_target() -> None:
    frame = build_modeling_frame(load_industry_returns(FIXTURE), horizon_days=1)
    assert {"Food", "Beer", "portfolio_return", "forward_realized_volatility_1d"} == set(frame.columns)
