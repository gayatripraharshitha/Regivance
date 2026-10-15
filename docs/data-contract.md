# Data contract

## Source and units

`load_industry_returns` reads the Kenneth French daily 30 Industry Portfolios CSV's value-weighted section, not its equal-weighted section. Source values are percentages; its returned DataFrame contains decimal simple returns. Source missing-value sentinels (-99.99 and -999) become missing values and downstream portfolio/feature construction rejects missing or nonfinite inputs.

## Timing and validation

Dates must be valid, unique and chronological. Valid historical Saturday observations are preserved: windows are measured in source observations, not calendar days. The loader parses; portfolio/feature builders enforce the relevant data constraints. No missing-value imputation or return clipping is performed.

The portfolio is the equal-weight mean of the 30 industry returns each day, implying daily rebalancing across industries. The source's 111.85% Services observation on 1931-01-29 is retained as 1.1185, not arbitrarily capped.

## Forecasting target

At close t, target volatility is `sqrt(sum(r[t+j]**2 for j=1..5))`. It intentionally uses future data as a supervised label; it must never be an input feature. The final five incomplete labels are dropped. The target is neither annualized nor demeaned.

A five-observation embargo separates training/validation and validation/test labels. All learned preprocessing, clustering and model parameters use the permitted training window; validation is reserved for early stopping/model choice. Inputs at forecast t may include returns through t, not t+1.

## Risk features

- Portfolio volatility: trailing 21-observation root mean squared portfolio return.
- Correlation: mean of the 435 distinct off-diagonal industry-pair correlations in the same trailing window.
- Dispersion: same-day population standard deviation across industry returns (`ddof=0`).

The first 20 rolling-feature rows remain missing. Undefined correlations from constant series also remain missing, not invented. Clustering/scaling fit training-only complete features; evaluation dates are never shifted to conceal warm-up issues.

## Provenance and prospective checks

Keep the unmodified raw vintage separately. Development manifests include raw/config/code/artifact hashes and dependency versions. The prospective protocol checks a canonical parsed-history hash through the exposure boundary, refuses revised history or protocol drift, and requires all five future target observations before scoring any final period.
