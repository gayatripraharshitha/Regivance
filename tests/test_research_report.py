import numpy as np
import pandas as pd
import pytest

from scripts.build_research_report import combine_predictions, summarize, QLIKE_MODEL


def sample():
    dates = pd.bdate_range("2020-01-01", periods=5)
    names = ["actual", "Rolling", "EWMA", "GARCH-normal", "GARCH-Student-t",
             "GJR-GARCH-Student-t", "LSTM", "Regime-aware LSTM", "Risk-feature LSTM", "Risk-feature + regime LSTM"]
    base = pd.DataFrame(0.02, columns=names, index=dates).assign(fold=1, regime="lower")
    switching = base.assign(regime_switching_forecast=0.02)
    qlike = base.copy()
    qlike[QLIKE_MODEL] = 0.02
    return base, switching, qlike


def test_report_contains_all_eleven_models():
    result = summarize(combine_predictions(*sample()))
    assert len(result) == 11
    assert (result[["mae", "rmse", "qlike"]] == 0).all().all()


@pytest.mark.parametrize("change", ["dates", "actual", "fold", "regime", "missing", "nonfinite"])
def test_report_rejects_unmatched_or_invalid_inputs(change):
    base, switching, qlike = sample()
    if change == "dates":
        qlike = qlike.iloc[:-1]
    elif change == "missing":
        base = base.drop(columns=["LSTM"])
    elif change == "nonfinite":
        qlike.iloc[0, qlike.columns.get_loc(QLIKE_MODEL)] = np.nan
    else:
        qlike.iloc[0, qlike.columns.get_loc(change)] = {"actual": 0.5, "fold": 2, "regime": "higher"}[change]
    with pytest.raises(ValueError):
        combine_predictions(base, switching, qlike)
