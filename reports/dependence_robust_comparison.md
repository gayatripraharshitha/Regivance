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
| Risk-feature + regime LSTM | Risk-feature LSTM | MAE | 0.00015227403 | [-3.3708784e-06, 0.00030791894] | 0.05517 | 0.331 |
| Risk-feature + regime LSTM | Risk-feature LSTM | Squared error | 1.4184395e-05 | [3.9557975e-06, 2.4412992e-05] | 0.006588 | 0.05929 |
| Risk-feature + regime LSTM | Risk-feature LSTM | QLIKE | 0.10624817 | [0.014038344, 0.198458] | 0.02394 | 0.1915 |
| Regime-aware LSTM | LSTM | MAE | 0.00025733661 | [0.00010293997, 0.00041173325] | 0.001097 | 0.01316 |
| Regime-aware LSTM | LSTM | Squared error | -9.4511623e-07 | [-7.9755184e-06, 6.0852859e-06] | 0.7921 | 1 |
| Regime-aware LSTM | LSTM | QLIKE | -0.01886747 | [-0.089381096, 0.051646156] | 0.5998 | 1 |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | MAE | -3.1767085e-05 | [-0.0006373168, 0.00057378263] | 0.9181 | 1 |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | Squared error | -7.762638e-05 | [-0.00015732689, 2.0741273e-06] | 0.05626 | 0.331 |
| Risk-feature + regime LSTM | GJR-GARCH-Student-t | QLIKE | -0.27788239 | [-0.44751276, -0.10825202] | 0.001333 | 0.01467 |
| Risk-feature LSTM | GJR-GARCH-Student-t | MAE | -0.00018404111 | [-0.00083512813, 0.0004670459] | 0.5794 | 1 |
| Risk-feature LSTM | GJR-GARCH-Student-t | Squared error | -9.1810774e-05 | [-0.00017990132, -3.7202252e-06] | 0.04109 | 0.2876 |
| Risk-feature LSTM | GJR-GARCH-Student-t | QLIKE | -0.38413056 | [-0.62975113, -0.13850999] | 0.002187 | 0.02187 |

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
