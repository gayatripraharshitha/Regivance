"""V2 stress, underprediction, VaR, and alternative-portfolio analyses."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_industry_returns
from src.data.portfolio import build_forecasting_dataset
from src.evaluation.metrics import calculate_volatility_metrics
from src.evaluation.splits import generate_walk_forward_splits
from src.models.ewma import forecast_ewma_volatility
from src.models.garch import forecast_garch_volatility_for_fold
from src.models.rolling_volatility import forecast_rolling_volatility
from src.research_config import CONFIG

PREDICTIONS = ROOT / "reports/tables/lstm_risk_feature_comparison_predictions.csv"
TABLES = ROOT / "reports/tables"
REPORT = ROOT / "reports/v2_risk_analysis.md"
REFERENCE = "GJR-GARCH-Student-t"
FOCUS = "Risk-feature + regime LSTM"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs():
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["date"]).set_index("date")
    required = {"actual", "fold", "regime", REFERENCE, FOCUS}
    require(required.issubset(predictions), f"Missing prediction columns: {sorted(required - set(predictions))}")
    require(len(predictions) == CONFIG["evaluation"]["development_size"], "Expected the full development evaluation.")
    require(predictions.index.is_unique and predictions.index.is_monotonic_increasing, "Prediction dates must be unique and chronological.")
    returns = load_industry_returns(ROOT / CONFIG["data"]["raw_path"])
    portfolio = returns.mean(axis=1)
    return predictions, returns, portfolio


def stress_table(predictions):
    shock = predictions.index.to_series().between("2020-02-01", "2020-04-30")
    groups = pd.Series("normal", index=predictions.index)
    groups.loc[predictions["regime"].eq("higher")] = "high_volatility"
    groups.loc[shock.to_numpy()] = "2020_shock"
    records = []
    for period in ("normal", "high_volatility", "2020_shock"):
        rows = predictions.loc[groups.eq(period)]
        for model in (REFERENCE, FOCUS):
            forecast = rows[model]
            records.append({"period": period, "model": model, "n": len(rows),
                "underprediction_rate": float((forecast < rows.actual).mean()),
                "mean_forecast_to_actual": float((forecast / rows.actual).mean()),
                **calculate_volatility_metrics(rows.actual, forecast)})
    return pd.DataFrame(records)


def underprediction_table(predictions, portfolio):
    # Information through t only: RMS of the five portfolio returns ending at t.
    trailing = portfolio.pow(2).rolling(5, min_periods=5).sum().pow(.5).reindex(predictions.index)
    rise = predictions.actual / trailing
    require(np.isfinite(rise).all() and (rise > 0).all(), "Invalid rapid-rise ratios.")
    labels = pd.qcut(rise, [0, .5, .8, .95, 1], labels=["stable_or_falling", "moderate_rise", "large_rise", "extreme_rise"])
    records = []
    for bucket in labels.cat.categories:
        rows = predictions.loc[labels.eq(bucket)]
        for model in (REFERENCE, FOCUS):
            ratio = rows[model] / rows.actual
            records.append({"rise_bucket": bucket, "model": model, "n": len(rows),
                "median_target_to_trailing_vol": float(rise.loc[rows.index].median()),
                "underprediction_rate": float((ratio < 1).mean()),
                "median_forecast_to_actual": float(ratio.median()),
                **calculate_volatility_metrics(rows.actual, rows[model])})
    return pd.DataFrame(records)


def var_coverage(predictions, portfolio):
    future_return = sum(portfolio.shift(-step) for step in range(1, 6)).reindex(predictions.index)
    require(np.isfinite(future_return).all(), "Incomplete future returns for VaR evaluation.")
    records = []
    for model in (REFERENCE, FOCUS):
        for confidence in (.95, .99):
            threshold = -norm.ppf(confidence) * predictions[model]
            breaches = future_return < threshold
            expected = 1 - confidence
            for sample, positions in (("overlapping_all", np.arange(len(predictions))),
                                      ("nonoverlapping_offset_0", np.arange(0, len(predictions), 5))):
                observed = breaches.iloc[positions]
                test = binomtest(int(observed.sum()), len(observed), expected)
                records.append({"model": model, "confidence": confidence, "sample": sample,
                    "n": len(observed), "breaches": int(observed.sum()),
                    "expected_breach_rate": expected, "observed_breach_rate": float(observed.mean()),
                    "exact_binomial_p_value": float(test.pvalue)})
    return pd.DataFrame(records)


def inverse_volatility_portfolio(returns, window=63):
    # Lagged weights ensure today's return is not used to set today's weights.
    volatility = returns.rolling(window, min_periods=window).std(ddof=0).shift(1)
    inverse = 1 / volatility.replace(0, np.nan)
    weights = inverse.div(inverse.sum(axis=1), axis=0)
    portfolio = (returns * weights).sum(axis=1, min_count=returns.shape[1]).dropna()
    require(np.isfinite(portfolio).all(), "Alternative portfolio contains invalid returns.")
    return portfolio


def portfolio_sensitivity(returns):
    alternative = inverse_volatility_portfolio(returns)
    synthetic = pd.DataFrame({"portfolio": alternative}, index=alternative.index)
    dataset = build_forecasting_dataset(synthetic, horizon=5)
    portfolio, target = dataset.portfolio_return, dataset.target_volatility_5d
    ev = CONFIG["evaluation"]
    splits = generate_walk_forward_splits(len(dataset), ev["train_size"], ev["validation_size"],
        ev["test_size"], ev["step_size"], ev["development_size"], ev["embargo"])
    frames = []
    for split in splits:
        dates = dataset.index[split.test]
        frame = pd.DataFrame(index=dates)
        frame["Rolling"] = forecast_rolling_volatility(portfolio, window=21, horizon=5).loc[dates]
        frame["EWMA"] = forecast_ewma_volatility(portfolio, decay=.94, horizon=5, min_periods=21).loc[dates]
        frame[REFERENCE] = forecast_garch_volatility_for_fold(
            portfolio.iloc[split.train], portfolio, dates, horizon=5, distribution="t", asymmetric=True)
        frame["actual"] = target.loc[dates]
        frames.append(frame)
    forecasts = pd.concat(frames)
    require(len(forecasts) == 2520 and np.isfinite(forecasts.to_numpy()).all(), "Incomplete alternative-portfolio forecasts.")
    return pd.DataFrame([{"portfolio": "lagged_63d_inverse_industry_volatility", "model": model,
        "n": len(forecasts), **calculate_volatility_metrics(forecasts.actual, forecasts[model])}
        for model in ("Rolling", "EWMA", REFERENCE)])


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    predictions, returns, portfolio = load_inputs()
    stress = stress_table(predictions)
    underprediction = underprediction_table(predictions, portfolio)
    var = var_coverage(predictions, portfolio)
    alternative = portfolio_sensitivity(returns)
    TABLES.mkdir(parents=True, exist_ok=True)
    stress.to_csv(TABLES / "v2_stress_period_metrics.csv", index=False)
    underprediction.to_csv(TABLES / "v2_rapid_rise_underprediction.csv", index=False)
    var.to_csv(TABLES / "v2_var_coverage.csv", index=False)
    alternative.to_csv(TABLES / "v2_alternative_portfolio_metrics.csv", index=False)
    stress_display = stress.assign(underprediction_pct=100*stress.underprediction_rate)
    var_display = var.assign(observed_pct=100*var.observed_breach_rate, expected_pct=100*var.expected_breach_rate)
    report = f"""# V2 risk relevance and generalization analysis

## Scope

These are exploratory extensions on the inspected development period. They do not replace the frozen prospective evaluation. The analyses expose stress weaknesses, study rapid-rise underprediction, demonstrate a simple conditional risk application, and test one additional portfolio construction.

## Normal, high-volatility and 2020 stress periods

The three groups are mutually exclusive. The 2020 shock is February–April 2020; high volatility means the fold-relative higher regime outside that interval; normal combines lower/middle regimes outside the shock.

{table(stress_display, ['period','model','n','mae','rmse','qlike','underprediction_pct'])}

## Rapid volatility increases

The rise ratio divides the forward target by trailing five-observation realized volatility known at the origin. Buckets are development-sample quantiles: bottom 50%, 50–80%, 80–95%, and top 5%. Underprediction means forecast volatility is below realized target volatility.

{table(underprediction, ['rise_bucket','model','n','median_target_to_trailing_vol','underprediction_rate','median_forecast_to_actual','qlike'])}

## Five-observation conditional VaR demonstration

The application compares the next five simple returns summed together with a zero-mean Normal VaR threshold formed from each model's five-observation volatility forecast. This is a calibration demonstration, not a trading strategy. Summed simple returns are an approximation to a compounded holding-period return. Overlapping observations are dependent; offset 0 reports every fifth origin. Exact binomial p-values do not correct serial dependence and are secondary diagnostics.

{table(var_display, ['model','confidence','sample','n','breaches','expected_pct','observed_pct','exact_binomial_p_value'])}

## Alternative portfolio sensitivity

The additional portfolio weights industries inversely to their trailing 63-observation volatility. Weights are lagged one observation, normalized daily, and use no transaction costs. This changes both the portfolio and target. Three established models are rerun over 2,520 origins; this is a sensitivity check rather than a complete second research study.

{table(alternative, ['portfolio','model','n','mae','rmse','qlike'])}

## Interpretation

Stress and rapid-rise results should be judged by the generated values rather than pooled averages alone. The VaR exercise adds a risk-management interpretation, while its distributional and overlapping-window assumptions prevent a production risk claim. The inverse-volatility result probes portfolio specificity but does not establish generalization across asset classes or datasets.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(stress.to_string(index=False))
    print("V2 risk analyses written.")


if __name__ == "__main__":
    main()
