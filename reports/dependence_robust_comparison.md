# Dependence-robust forecast comparison

## Scope and status

- Forecast origins: 2,520, from 2016-08-15 to 2026-08-24.
- Comparisons use identical chronological dates and actual targets.
- This is exploratory: the evaluation period has informed model development.
- Positive mean differences are `comparator loss − candidate loss`; positive values favour the candidate.

## Method

Per-origin absolute error, squared error, and variance-based QLIKE are paired by date. The reported intervals use an intercept-only OLS mean with Bartlett Newey-West/HAC covariance and finite-sample correction. Lags 4, 10, and 20 are reported; lag 4 is the prespecified overlap-motivated primary lag. Holm correction is applied across the 12 comparison-by-loss hypotheses separately at each HAC lag. Moving-block bootstrap percentile intervals (2,000 deterministic repetitions; chronological blocks of 5 and 21) are sensitivity analyses.

## Primary HAC results (lag 4)

| Candidate | Comparator | Loss | Mean difference | 95% HAC CI | Unadjusted p | Holm p |
|---|---|---|---:|---:|---:|---:|
| Risk-feature + regime LSTM | Risk-feature LSTM | MAE | 0.00015227404 | [-3.3708729e-06, 0.00030791896] | 0.05517 | 0.331 |
| Risk-feature + regime LSTM | Risk-feature LSTM | Squared error | 1.4184397e-05 | [3.9557988e-06, 2.4412995e-05] | 0.006588 | 0.05929 |
| Risk-feature + regime LSTM | Risk-feature LSTM | QLIKE | 0.1062482 | [0.014038357, 0.19845804] | 0.02394 | 0.1915 |
| Regime-aware LSTM | LSTM | MAE | 0.00025733663 | [0.00010293999, 0.00041173327] | 0.001097 | 0.01316 |
| Regime-aware LSTM | LSTM | Squared error | -9.4511556e-07 | [-7.9755174e-06, 6.0852863e-06] | 0.7921 | 1 |
| Regime-aware LSTM | LSTM | QLIKE | -0.018867461 | [-0.089381084, 0.051646163] | 0.5998 | 1 |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | MAE | -3.1770292e-05 | [-0.00063731965, 0.00057377907] | 0.9181 | 1 |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | Squared error | -7.762647e-05 | [-0.00015732697, 2.0740322e-06] | 0.05626 | 0.331 |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | QLIKE | -0.27788236 | [-0.44751268, -0.10825203] | 0.001333 | 0.01467 |
| Risk-feature LSTM | GJR-GARCH-Student-t | MAE | -0.00018404433 | [-0.00083513107, 0.0004670424] | 0.5794 | 1 |
| Risk-feature LSTM | GJR-GARCH-Student-t | Squared error | -9.1810867e-05 | [-0.00017990141, -3.7203204e-06] | 0.04108 | 0.2876 |
| Risk-feature LSTM | GJR-GARCH-Student-t | QLIKE | -0.38413056 | [-0.62975111, -0.13851] | 0.002187 | 0.02187 |

## HAC lag sensitivity

| Candidate | Comparator | Loss | Lags with CI excluding zero | Lags with Holm p < 0.05 |
|---|---|---|---|---|
| Risk-feature + regime LSTM | Risk-feature LSTM | MAE | none | none |
| Risk-feature + regime LSTM | Risk-feature LSTM | Squared error | 4, 10 | none |
| Risk-feature + regime LSTM | Risk-feature LSTM | QLIKE | 4 | none |
| Regime-aware LSTM | LSTM | MAE | 4, 10, 20 | 4 |
| Regime-aware LSTM | LSTM | Squared error | none | none |
| Regime-aware LSTM | LSTM | QLIKE | none | none |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | MAE | none | none |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | Squared error | none | none |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | QLIKE | 4, 10, 20 | 4 |
| Risk-feature LSTM | GJR-GARCH-Student-t | MAE | none | none |
| Risk-feature LSTM | GJR-GARCH-Student-t | Squared error | 4 | none |
| Risk-feature LSTM | GJR-GARCH-Student-t | QLIKE | 4, 10 | 4 |

## Interpretation

The lag-sensitivity table shows that some lag-4 findings do not remain after broader HAC dependence allowances or Holm adjustment. Therefore the analysis does not support a general claim that regime indicators improve LSTM forecasting. The QLIKE disadvantage of both risk-feature LSTM variants against GJR-GARCH is directionally persistent, but its Holm-adjusted conclusion is also lag-sensitive. A non-significant result is not evidence of equivalence, and a statistically distinguishable loss difference does not establish economic usefulness. Fold-win counts are not used as independent statistical evidence. Overlapping targets induce serial dependence, and all intervals and p-values remain exploratory because this period informed development.

## Outputs

- `reports/tables/dependence_robust_paired_loss_tests.csv`
- `reports/tables/dependence_robust_moving_block_bootstrap.csv`
- `reports/figures/dependence_robust_hac_effects.png`
