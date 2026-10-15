# Data preprocessing and model diagnostics

## Evaluation data

- Prediction rows: 2,520
- Folds: 120
- Evaluation dates: 2016-08-15 to 2026-08-24
- Results are development results, not an untouched final holdout.

## Preprocessing approach

- Industry returns are loaded through the project's data loader.
- The forecasting target is checked against `target_volatility_5d` in the saved processed dataset.
- Rolling risk features use a trailing 21-observation window.
- The features are trailing portfolio volatility, mean industry correlation, and cross-sectional dispersion.
- The rolling-window warm-up remains missing; no generic backfill, interpolation, or blanket imputation is performed.
- In the LSTM experiment, continuous feature scalers are fitted on each fold's training dates only.
- Regime detection is fitted on training features, then applied to later observations using the frozen detector.
- Input sequences use a 21-observation lookback; labels remain aligned to the target date.

## Error handling and integrity checks

Inputs are checked for required files and columns, valid and unique dates, chronological ordering, numeric finite values, positive volatility targets and forecasts, valid regime labels, fold completeness, target alignment, and reproducibility of saved fold metrics.

Invalid or unexpected missing values cause an explicit error instead of being silently replaced.

| Check | Status | Details |
|---|---|---|
| Prediction date integrity | PASS | Dates are valid, unique and chronological. |
| Numeric prediction validity | PASS | Targets and every model forecast are finite and positive. |
| Fold and regime integrity | PASS | 120 folds; 21 predictions per fold; regime labels are valid. |
| Target alignment | PASS | All saved actual targets match the processed dataset by date. |
| Raw input and feature construction | PASS | The project loader and 21-observation risk-feature builder accepted the data; feature dates match return dates. |
| Rolling-feature missingness | PASS | Missing values, if any, are confined to each feature's initial rolling-window warm-up; no silent imputation occurred. |
| Fold-metric reproducibility | PASS | Every saved fold/model MAE, RMSE and QLIKE matches the score recalculated from the prediction rows. |

## Pooled metrics

| Metric | Best model in this evaluation | Score |
|---|---|---:|
| MAE | GJR-GARCH-Student-t | 0.00746768 |
| RMSE | GJR-GARCH-Student-t | 0.01112928 |
| QLIKE | GJR-GARCH-Student-t | 0.41919133 |

These metrics summarise this evaluation only. They do not establish statistical significance or performance on a future untouched holdout.

## Generated figures

- `model_comparison_mae.png`
- `model_comparison_rmse.png`
- `model_comparison_qlike.png`
- `foldwise_qlike_difference.png`
- `forecasts_during_2020_shock.png`
- `forecast_calibration_scatter.png`
- `preprocessing_risk_features_2020.png`
- `preprocessing_feature_missingness.png`

## Interpretation note

The 2020 period is shown as a diagnostic example, not as a separate tuning target. Fold-wise results should be read alongside pooled metrics because a small number of extreme errors can dominate a pooled loss.
