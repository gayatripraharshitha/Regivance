"""Small fixed-subset experiment measuring neural initialization sensitivity."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_lstm_models import build_regime_one_hot, evaluate_model, load_inputs, make_model_config
from scripts.compare_lstm_risk_features import build_model_inputs, scale_returns, scale_risk_features
from src.evaluation.metrics import calculate_volatility_metrics
from src.features.rolling_features import build_regime_features
from src.research_config import CONFIG

PREDICTIONS = ROOT / "reports/tables/lstm_risk_feature_comparison_predictions.csv"
OUTPUT = ROOT / "reports/tables/v2_lstm_seed_sensitivity.csv"
REPORT = ROOT / "reports/v2_lstm_seed_sensitivity.md"
MODEL = "Risk-feature + regime LSTM"
REFERENCE = "GJR-GARCH-Student-t"
SEED_BASES = (7, 42, 101, 202, 999)
# Prespecified broad historical coverage plus the 2020 shock fold (43).
FOLD_NUMBERS = (1, 21, 41, 43, 61, 81, 101, 120)


def main():
    torch.set_num_threads(1)
    returns, dataset, _, splits = load_inputs()
    selected = [split for split in splits if split.fold in FOLD_NUMBERS]
    if tuple(split.fold for split in selected) != FOLD_NUMBERS:
        raise ValueError("Configured seed-sensitivity folds are unavailable.")
    saved = pd.read_csv(PREDICTIONS, parse_dates=["date"]).set_index("date")
    targets = dataset.target_volatility_5d.astype(float)
    risk = build_regime_features(returns, window=CONFIG["features"]["window"]).reindex(dataset.index)
    industry = returns.reindex(dataset.index)
    records = []
    for seed_base in SEED_BASES:
        frames = []
        for split in selected:
            train_dates = dataset.index[split.train]
            inputs = build_model_inputs(scale_returns(industry, train_dates),
                scale_risk_features(risk, train_dates), build_regime_one_hot(risk, train_dates)[0])
            config = replace(make_model_config(split.fold, CONFIG["models"]["epochs"],
                CONFIG["models"]["hidden_size"], CONFIG["models"]["batch_size"]),
                random_state=seed_base + split.fold)
            forecast, actual, dates = evaluate_model(inputs[MODEL], targets, split, config)
            frames.append(pd.DataFrame({"actual": actual, "forecast": forecast}, index=dates))
        result = pd.concat(frames)
        expected = saved.loc[result.index]
        if not np.allclose(result.actual, expected.actual, rtol=1e-6, atol=1e-8):
            raise ValueError("Seed experiment target dates do not match saved development targets.")
        records.append({"seed_base": seed_base, "n_folds": len(selected), "n_test": len(result),
            **calculate_volatility_metrics(result.actual, result.forecast)})
    summary = pd.DataFrame(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    dates = pd.DatetimeIndex(np.concatenate([dataset.index[s.test].to_numpy() for s in selected]))
    reference = calculate_volatility_metrics(saved.loc[dates, "actual"], saved.loc[dates, REFERENCE])
    original = calculate_volatility_metrics(saved.loc[dates, "actual"], saved.loc[dates, MODEL])
    stats = summary[["mae", "rmse", "qlike"]].agg(["mean", "std", "min", "max"])
    REPORT.write_text(f"""# V2 LSTM initialization sensitivity

## Design

The risk-feature-plus-regime LSTM is retrained with five prespecified seed bases ({', '.join(map(str, SEED_BASES))}) on eight fixed folds ({', '.join(map(str, FOLD_NUMBERS))}). Each run uses seed base plus fold number and retains the frozen 21-observation lookback, hidden size 12, maximum 12 epochs and validation early stopping. The subset contains {len(dates)} test origins and deliberately includes Fold 43, the February–March 2020 stress fold.

This compact experiment measures initialization sensitivity. It is exploratory, uses a historical subset and is not a substitute for full repeated-seed evaluation or prospective evidence.

## Per-seed results

| Seed base | Test origins | MAE | RMSE | QLIKE |
| ---: | ---: | ---: | ---: | ---: |
{chr(10).join(f'| {int(r.seed_base)} | {int(r.n_test)} | {r.mae:.6f} | {r.rmse:.6f} | {r.qlike:.6f} |' for r in summary.itertuples())}

## Across-seed dispersion

| Statistic | MAE | RMSE | QLIKE |
| --- | ---: | ---: | ---: |
{chr(10).join(f'| {name} | {row.mae:.6f} | {row.rmse:.6f} | {row.qlike:.6f} |' for name, row in stats.iterrows())}

On the identical subset, the originally saved LSTM has QLIKE {original['qlike']:.6f}; Student-t GJR-GARCH has QLIKE {reference['qlike']:.6f}. Seed dispersion quantifies one source of neural uncertainty, while model comparison remains secondary because these folds were selected for diagnostic coverage.
""", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
