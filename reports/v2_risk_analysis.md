# V2 risk relevance and generalization analysis

## Scope

These are exploratory extensions on the inspected development period. They do not replace the frozen prospective evaluation. The analyses expose stress weaknesses, study rapid-rise underprediction, demonstrate a simple conditional risk application, and test one additional portfolio construction.

## Normal, high-volatility and 2020 stress periods

The three groups are mutually exclusive. The 2020 shock is February–April 2020; high volatility means the fold-relative higher regime outside that interval; normal combines lower/middle regimes outside the shock.

| period | model | n | mae | rmse | qlike | underprediction_pct |
| --- | --- | --- | --- | --- | --- | --- |
| normal | GJR-GARCH-Student-t | 2242 | 0.0068 | 0.0093 | 0.4030 | 31.8912 |
| normal | Risk-feature + regime LSTM | 2242 | 0.0062 | 0.0092 | 0.4869 | 49.3756 |
| high_volatility | GJR-GARCH-Student-t | 216 | 0.0091 | 0.0129 | 0.4034 | 39.8148 |
| high_volatility | Risk-feature + regime LSTM | 216 | 0.0087 | 0.0128 | 0.7486 | 71.2963 |
| 2020_shock | GJR-GARCH-Student-t | 62 | 0.0255 | 0.0368 | 1.0605 | 43.5484 |
| 2020_shock | Risk-feature + regime LSTM | 62 | 0.0505 | 0.0674 | 8.1162 | 88.7097 |

## Rapid volatility increases

The rise ratio divides the forward target by trailing five-observation realized volatility known at the origin. Buckets are development-sample quantiles: bottom 50%, 50–80%, 80–95%, and top 5%. Underprediction means forecast volatility is below realized target volatility.

| rise_bucket | model | n | median_target_to_trailing_vol | underprediction_rate | median_forecast_to_actual | qlike |
| --- | --- | --- | --- | --- | --- | --- |
| stable_or_falling | GJR-GARCH-Student-t | 1260 | 0.6950 | 0.0675 | 1.5468 | 0.4151 |
| stable_or_falling | Risk-feature + regime LSTM | 1260 | 0.6950 | 0.3008 | 1.2153 | 0.4317 |
| moderate_rise | GJR-GARCH-Student-t | 756 | 1.2049 | 0.4444 | 1.0311 | 0.1224 |
| moderate_rise | Risk-feature + regime LSTM | 756 | 1.2049 | 0.6825 | 0.8584 | 0.4159 |
| large_rise | GJR-GARCH-Student-t | 378 | 1.8685 | 0.7778 | 0.8060 | 0.5063 |
| large_rise | Risk-feature + regime LSTM | 378 | 1.8685 | 0.8175 | 0.7334 | 1.3589 |
| extreme_rise | GJR-GARCH-Student-t | 126 | 3.1261 | 0.8968 | 0.5457 | 1.9791 |
| extreme_rise | Risk-feature + regime LSTM | 126 | 3.1261 | 0.8889 | 0.5742 | 3.0528 |

## Five-observation conditional VaR demonstration

The application compares the next five simple returns summed together with a zero-mean Normal VaR threshold formed from each model's five-observation volatility forecast. This is a calibration demonstration, not a trading strategy. Summed simple returns are an approximation to a compounded holding-period return. Overlapping observations are dependent; offset 0 reports every fifth origin. Exact binomial p-values do not correct serial dependence and are secondary diagnostics.

| model | confidence | sample | n | breaches | expected_pct | observed_pct | exact_binomial_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GJR-GARCH-Student-t | 0.9500 | overlapping_all | 2520 | 125 | 5.0000 | 4.9603 | 0.9636 |
| GJR-GARCH-Student-t | 0.9500 | nonoverlapping_offset_0 | 504 | 25 | 5.0000 | 4.9603 | 1.0000 |
| GJR-GARCH-Student-t | 0.9900 | overlapping_all | 2520 | 63 | 1.0000 | 2.5000 | 0.0000 |
| GJR-GARCH-Student-t | 0.9900 | nonoverlapping_offset_0 | 504 | 14 | 1.0000 | 2.7778 | 0.0007 |
| Risk-feature + regime LSTM | 0.9500 | overlapping_all | 2520 | 177 | 5.0000 | 7.0238 | 0.0000 |
| Risk-feature + regime LSTM | 0.9500 | nonoverlapping_offset_0 | 504 | 38 | 5.0000 | 7.5397 | 0.0136 |
| Risk-feature + regime LSTM | 0.9900 | overlapping_all | 2520 | 95 | 1.0000 | 3.7698 | 0.0000 |
| Risk-feature + regime LSTM | 0.9900 | nonoverlapping_offset_0 | 504 | 23 | 1.0000 | 4.5635 | 0.0000 |

## Alternative portfolio sensitivity

The additional portfolio weights industries inversely to their trailing 63-observation volatility. Weights are lagged one observation, normalized daily, and use no transaction costs. This changes both the portfolio and target. Three established models are rerun over 2,520 origins; this is a sensitivity check rather than a complete second research study.

| portfolio | model | n | mae | rmse | qlike |
| --- | --- | --- | --- | --- | --- |
| lagged_63d_inverse_industry_volatility | Rolling | 2520 | 0.0077 | 0.0127 | 0.5455 |
| lagged_63d_inverse_industry_volatility | EWMA | 2520 | 0.0077 | 0.0122 | 0.4937 |
| lagged_63d_inverse_industry_volatility | GJR-GARCH-Student-t | 2520 | 0.0071 | 0.0107 | 0.4457 |

## Interpretation

Stress and rapid-rise results should be judged by the generated values rather than pooled averages alone. The VaR exercise adds a risk-management interpretation, while its distributional and overlapping-window assumptions prevent a production risk claim. The inverse-volatility result probes portfolio specificity but does not establish generalization across asset classes or datasets.
