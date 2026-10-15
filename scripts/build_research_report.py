"""Consolidate verified development forecasts, robustness tables and a technical report."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research_config import CONFIG
from src.evaluation.metrics import calculate_volatility_metrics, calculate_volatility_loss_contributions
from scripts.analyze_dependence_robust_comparison import hac_mean_difference, holm_adjust, moving_block_bootstrap_ci

TABLES = ROOT / "reports/tables"
QLIKE_MODEL = "QLIKE-objective Risk-feature + regime LSTM"
REFERENCE = "GJR-GARCH-Student-t"


def combine_predictions(base, switching, qlike):
    """Reject mismatched experiment dates/targets instead of silently intersecting."""
    required = {"actual", "fold", "regime", REFERENCE, "LSTM", "Regime-aware LSTM",
                "Risk-feature LSTM", "Risk-feature + regime LSTM", "Rolling", "EWMA",
                "GARCH-normal", "GARCH-Student-t"}
    if not required.issubset(base.columns):
        raise ValueError("Incomplete base comparison schema.")
    if base.empty or base.index.has_duplicates or not base.index.is_monotonic_increasing or base.index.hasnans:
        raise ValueError("Forecast dates must be nonempty, valid, unique and chronological.")
    result = base.copy()
    for other, source, name in ((switching, "regime_switching_forecast", "Regime-switching"),
                                 (qlike, QLIKE_MODEL, QLIKE_MODEL)):
        if not other.index.equals(base.index):
            raise ValueError("Experiment dates differ.")
        for column in ("actual", REFERENCE):
            if not np.allclose(other[column], base[column], rtol=1e-6, atol=1e-10):
                raise ValueError(f"Experiment {column} differs.")
        for column in ("fold", "regime"):
            if not other[column].equals(base[column]):
                raise ValueError(f"Experiment {column} differs.")
        result[name] = other[source]
    numeric = result.drop(columns=["fold", "regime"]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Targets and predictions must be finite and positive.")
    return result


def summarize(data):
    models = data.columns.drop(["actual", "fold", "regime"])
    return pd.DataFrame([{"model": model, "n_test": len(data),
        **calculate_volatility_metrics(data.actual, data[model])} for model in models])


def markdown_table(frame):
    # Avoid adding a tabulate dependency just for Markdown rendering.
    rows = [[f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v)
             for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows)])


def main():
    paths = [TABLES / name for name in ("lstm_risk_feature_comparison_predictions.csv",
        "regime_switching_predictions.csv", "lstm_qlike_objective_predictions.csv")]
    data = combine_predictions(*(pd.read_csv(p, index_col="date", parse_dates=True) for p in paths))
    if len(data) != CONFIG["evaluation"]["development_size"]:
        raise ValueError("Refusing to present a smoke test as a full development run.")
    if not data.groupby("fold").size().eq(CONFIG["evaluation"]["test_size"]).all():
        raise ValueError("Incomplete folds.")
    overall = summarize(data).sort_values("qlike")
    overall.to_csv(TABLES / "research_all_model_metrics.csv", index=False)
    data.to_csv(TABLES / "research_all_model_predictions.csv", index_label="date")
    groups = []
    for kind, grouped in (("year", data.groupby(data.index.year)), ("regime", data.groupby("regime"))):
        for label, rows in grouped:
            groups.append(summarize(rows).assign(group_type=kind, group=str(label)))
    for offset in range(5):
        groups.append(summarize(data.iloc[offset::5]).assign(group_type="nonoverlapping_offset", group=str(offset)))
    # These are descriptive stress checks chosen after seeing development results.
    shock = (data.index >= "2020-02-01") & (data.index <= "2020-04-30")
    for label, mask in (("2020_Feb_Apr", shock), ("excluding_2020_Feb_Apr", ~shock)):
        if mask.any():
            groups.append(summarize(data.loc[mask]).assign(group_type="shock_sensitivity", group=label))
    pd.concat(groups, ignore_index=True).to_csv(TABLES / "research_robustness_metrics.csv", index=False)
    reference_loss = calculate_volatility_loss_contributions(data.actual, data[REFERENCE])["qlike"]
    tests = []
    for model in overall.model:
        if model == REFERENCE:
            continue
        difference = reference_loss - calculate_volatility_loss_contributions(data.actual, data[model])["qlike"]
        bootstrap = moving_block_bootstrap_ci(difference, block_length=21)
        for lag in (4, 10, 20):
            tests.append({"model": model, "lag": lag, **hac_mean_difference(difference, lag), **bootstrap})
    tests = pd.DataFrame(tests)
    tests["holm_p"] = tests.groupby("lag")["p_value_two_sided"].transform(holm_adjust)
    tests.to_csv(TABLES / "research_qlike_comparisons.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(overall.model, overall.qlike, color=["#217a60" if m == REFERENCE else "#587ca3" for m in overall.model])
    ax.invert_yaxis()
    ax.set(xlabel="Mean QLIKE (lower is better)", title="Regivance: development comparison, not final holdout")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/all_models_qlike.png", dpi=180)
    plt.close(fig)
    source = ROOT / CONFIG["data"]["raw_path"]
    lineage = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, *paths]}
    (TABLES / "research_report_inputs.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    score = overall.set_index("model")
    qlike_change = 100 * (1 - score.loc[QLIKE_MODEL, "qlike"] / score.loc["Risk-feature + regime LSTM", "qlike"])
    report = f"""# Regivance — completed development research report

## Executive conclusion

On {len(data):,} chronological development origins ({data.index[0].date()} to {data.index[-1].date()}),
{overall.iloc[0]['model']} has the lowest pooled QLIKE. This is an out-of-sample forecasting
comparison within an already inspected development period, **not an untouched confirmatory test**.
The current evidence does not establish a general advantage from market regimes or LSTMs.
Changing the risk-plus-regime LSTM objective to QLIKE reduces its QLIKE by {qlike_change:.1f}%
versus its log-MSE counterpart, but does not beat GJR-GARCH. No model is a trading recommendation.

## Data and forecast definition

Input: Kenneth R. French Data Library, daily 30 Industry Portfolios, value-weighted source table.
Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Archive: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_Daily_CSV.zip
The local vintage ends 2026-08-31. Retrieval date was not recorded and must not be invented.
Source SHA-256: `{lineage[str(source.relative_to(ROOT))]}`.
Loader converts percentages to decimal returns. Source observations, including historical
Saturdays, are preserved. The source-confirmed Services return of 111.85% on 1931-01-29
is retained; it is not automatically treated as a parser error or winsorized.

Each day, the portfolio assigns 1/30 weight to each industry (daily rebalancing).
At the close of t, target volatility is `sqrt(sum(r[t+j]**2, j=1..5))`.
It is neither annualized nor a centered sample standard deviation. The final five
observations cannot have complete targets. MAE/RMSE are in decimal five-observation
volatility units. QLIKE uses variance ratio `z = actual**2 / forecast**2` and loss
`z - log(z) - 1`; lower is better for all three metrics.

## Chronology and models

120 rolling folds: 1,260 training origins, 252 validation origins, 21 test origins.
Five-observation gaps separate training/validation and validation/test targets. Models
use information through each origin only. GARCH parameters are fitted on training data;
newly observed returns update its variance state without refitting within a test block.
Scalers and K-means are fitted on training features only. LSTM early stopping and regime
model choices use validation data, not test scores. Subsequent folds may use earlier
test observations once they have become historical: this is a sequential protocol.

Five statistical models: rolling 21-observation RMS volatility, EWMA (0.94), normal
GARCH(1,1), Student-t GARCH(1,1), and Student-t GJR-GARCH. Regime-switching selects among
these via validation QLIKE, falling back to the global validation winner when a regime
has fewer than 20 validation observations.

Three regime features: trailing 21-observation portfolio volatility, mean pairwise
industry correlation, and same-day cross-industry dispersion. Training-standardized
K-means uses three clusters, seed 42. Clusters are ranked lower/middle/higher by their
training volatility centers; they are relative ranks, not permanent economic labels.

Four LSTM inputs: 30 industry returns; returns plus one-hot regimes; returns plus
three risk features; returns plus risk features and regimes. Lookback 21, one recurrent
layer, hidden size 12, batch size 128, Adam learning rate 0.001 and weight decay 0.00001,
maximum 12 epochs, patience 4, CPU execution, seed 42 + fold. Primary neural fits use
training-standardized log-volatility MSE; the additional controlled model uses QLIKE.
These are small fixed research baselines, not an exhaustive hyperparameter search.

## Reproduced pooled scores

{markdown_table(overall)}

![Pooled QLIKE comparison](figures/all_models_qlike.png)

## Uncertainty and robustness

`tables/research_qlike_comparisons.csv` pairs every alternative against GJR-GARCH:
Newey-West HAC lags 4/10/20, Holm-adjusted two-sided p-values across the 10 alternatives
separately for each lag, and moving-block bootstrap 95% intervals (length 21, 2,000
resamples, seed 42). Positive reference-minus-candidate loss differences favor candidates.
Existing `dependence_robust_comparison.md` also isolates incremental regime contributions
with matched neural inputs, three loss functions and bootstrap block lengths 5/21.

`tables/research_robustness_metrics.csv` includes year/regime subsets, all five offsets of
nonoverlapping five-observation targets, and February–April 2020 shock sensitivity.
Nonoverlapping targets reduce mechanical overlap but do not imply independent errors.
These subsets were selected after development inspection and are descriptive, not new
confirmatory tests. The bootstrap conditions on saved predictions; it does not account
for all model-selection or training-seed uncertainty. See `forecast_failure_attribution.md`
and `preprocessing_and_diagnostics.md` for concentration, calibration and timing checks.

## Prospective evaluation and limitations

All available dates have informed development. No historical split is relabeled untouched.
The frozen prospective protocol begins strictly after 2026-10-15 and requires the first
252 forecast origins plus five subsequent target observations: 257 new observations.
`scripts/final_evaluation.py` checks code/config fingerprints and the original history,
then evaluates the same 11 models with train-only transformations and validation choices.
It refuses incomplete periods, protocol drift and output replacement. QLIKE against
GJR-GARCH is primary; other losses and regimes are secondary. Later folds may refit
using earlier final-period observations, as prescribed before scoring begins.

The run cannot supply future market observations today. Confirmatory conclusions remain
pending. Further limitations: a single portfolio/data vintage, overlapping targets,
historical source revisions, relative cluster instability, unoptimized small networks,
one seed per fold (not a multi-seed generalization claim), and no transaction-cost,
portfolio-allocation or economic-utility experiment. No causal claim about regimes is made.

## Reproduction and deliverables

Run `python scripts/run_research.py` after dependency installation and data placement.
Use `--resume` only with unchanged code/config/data/environment; successful stage artifact
hashes are checked. Previous controlled QLIKE outputs are archived, never silently erased.
The run manifest and environment snapshot are under `reports/runs/development/`.
This report's input hashes are in `tables/research_report_inputs.json`.
See README for tests, exact-version environment and prospective commands.
The implementation and development report are complete; prospective evidence remains pending.
"""
    (ROOT / "reports/research_report.md").write_text(report, encoding="utf-8")
    print(overall.to_string(index=False))
    print("Report and robustness artifacts generated; final holdout remains prospective.")


if __name__ == "__main__":
    main()
