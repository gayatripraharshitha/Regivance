import generated from "./research.generated.json";

export const modelResults = generated.models;
export const stressPeriods = generated.stressPeriods;

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
