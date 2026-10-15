export const modelResults = [
  { model: "GJR-GARCH Student-t", family: "Statistical", mae: 0.007468, rmse: 0.011129, qlike: 0.419191 },
  { model: "GARCH Student-t", family: "Statistical", mae: 0.007828, rmse: 0.011590, qlike: 0.423978 },
  { model: "GARCH Normal", family: "Statistical", mae: 0.007779, rmse: 0.011505, qlike: 0.429227 },
  { model: "Regime switching", family: "Regime", mae: 0.007537, rmse: 0.011246, qlike: 0.437712 },
  { model: "EWMA", family: "Baseline", mae: 0.008098, rmse: 0.012693, qlike: 0.465240 },
  { model: "Rolling", family: "Baseline", mae: 0.008193, rmse: 0.013208, qlike: 0.514782 },
  { model: "QLIKE risk + regime LSTM", family: "Neural", mae: 0.008328, rmse: 0.014348, qlike: 0.641875 },
  { model: "Risk + regime LSTM", family: "Neural", mae: 0.007499, rmse: 0.014195, qlike: 0.697074 },
  { model: "Risk-feature LSTM", family: "Neural", mae: 0.007652, rmse: 0.014686, qlike: 0.803322 },
  { model: "LSTM", family: "Neural", mae: 0.008189, rmse: 0.015012, qlike: 0.896497 },
  { model: "Regime-aware LSTM", family: "Neural", mae: 0.007932, rmse: 0.015043, qlike: 0.915365 },
];

export const regimeFeatures = [
  { title: "Portfolio volatility", formula: "21-observation RMS", body: "Measures the recent amplitude of equal-industry portfolio returns." },
  { title: "Industry correlation", formula: "Mean of 435 pairs", body: "Shows whether industries are moving together or following separate paths." },
  { title: "Cross-sectional dispersion", formula: "Same-day population σ", body: "Captures how widely the 30 industry returns differ at a forecast origin." },
];

export const workflow = [
  ["01", "Parse", "Read the value-weighted 30-industry source table and convert percent returns to decimals."],
  ["02", "Construct", "Build the daily equal-industry portfolio and five-observation forward volatility target."],
  ["03", "Discover", "Fit training-only scaling and K-means; rank clusters by training-center volatility."],
  ["04", "Forecast", "Compare statistical, regime-switching and LSTM models on identical future dates."],
  ["05", "Evaluate", "Measure MAE, RMSE and QLIKE with dependence-aware uncertainty checks."],
];
