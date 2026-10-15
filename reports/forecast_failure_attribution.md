# Forecast failure attribution

## Scope

- Predictions: 2,520
- Walk-forward folds: 120
- Evaluation dates: 2016-08-15 to 2026-08-24
- Focus model: Risk-feature + regime LSTM
- Reference model: GJR-GARCH-Student-t

The evaluation period has already informed model development. These are exploratory diagnostics, not a final untouched test.

## Pooled QLIKE

| Model | Mean QLIKE |
|---|---:|
| GJR-GARCH-Student-t | 0.41919135 |
| Risk-feature + regime LSTM | 0.69707371 |

## Loss concentration

| Model | Worst 1% share | Worst 5% share | Worst 10% share |
|---|---:|---:|---:|
| GJR-GARCH-Student-t | 18.04% | 37.61% | 51.32% |
| Risk-feature + regime LSTM | 32.95% | 56.64% | 67.92% |

## Fold 43 diagnostic

Fold 43 spans February 18 to March 17, 2020. It contains 21 forecast origins.

- Mean daily QLIKE for the focus model: 17.263926
- Median daily QLIKE for the focus model: 16.330819
- Maximum daily QLIKE for the focus model: 41.509931
- The ten largest daily QLIKE contributions total 281.396389 out of 362.542456, accounting for 77.62% of Fold 43's total QLIKE.

## Interpretation limitations

The five-observation targets use overlapping future return windows. Consequently, adjacent forecast errors are dependent, and one market shock may contribute to several successive large losses. The loss-concentration results are descriptive; they do not establish statistical significance or identify causality.

This analysis does not change or clip forecasts. It attributes loss using the predictions that were already generated.

## Generated outputs

- `reports/tables/forecast_failure_daily_losses.csv`
- `reports/tables/forecast_failure_concentration.csv`
- `reports/tables/forecast_failure_fold_summary.csv`
- `reports/tables/forecast_failure_top_50_dates.csv`
- `reports/tables/forecast_failure_fold43_top10.csv`
- `reports/figures/forecast_failure_daily_qlike.png`
- `reports/figures/forecast_failure_loss_concentration.png`
- `reports/figures/forecast_failure_worst_dates.png`
