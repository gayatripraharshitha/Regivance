# Regivance — completed development research report

## Executive conclusion

On 2,520 chronological development origins (2016-08-15 to 2026-08-24),
GJR-GARCH-Student-t has the lowest pooled QLIKE. This is an out-of-sample forecasting
comparison within an already inspected development period, **not an untouched confirmatory test**.
The current evidence does not establish a general advantage from market regimes or LSTMs.
Changing the risk-plus-regime LSTM objective to QLIKE reduces its QLIKE by 7.9%
versus its log-MSE counterpart, but does not beat GJR-GARCH. No model is a trading recommendation.

## Data and forecast definition

Input: Kenneth R. French Data Library, daily 30 Industry Portfolios, value-weighted source table.
Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Archive: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_Daily_CSV.zip
The local vintage ends 2026-08-31. Retrieval date was not recorded and must not be invented.
Source SHA-256: `dd2ca61d3cfe66c279e347a91535e12520907ceb98808ca42fff05c0657dc924`.
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

| model | n_test | mae | rmse | qlike |
| --- | --- | --- | --- | --- |
| GJR-GARCH-Student-t | 2520 | 0.007468 | 0.011129 | 0.419191 |
| GARCH-Student-t | 2520 | 0.007828 | 0.011590 | 0.423978 |
| GARCH-normal | 2520 | 0.007779 | 0.011505 | 0.429227 |
| Regime-switching | 2520 | 0.007537 | 0.011246 | 0.437712 |
| EWMA | 2520 | 0.008098 | 0.012693 | 0.465240 |
| Rolling | 2520 | 0.008193 | 0.013208 | 0.514782 |
| QLIKE-objective Risk-feature + regime LSTM | 2520 | 0.008328 | 0.014348 | 0.641875 |
| Risk-feature + regime LSTM | 2520 | 0.007499 | 0.014195 | 0.697074 |
| Risk-feature LSTM | 2520 | 0.007652 | 0.014686 | 0.803322 |
| LSTM | 2520 | 0.008189 | 0.015012 | 0.896497 |
| Regime-aware LSTM | 2520 | 0.007932 | 0.015043 | 0.915365 |

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
