from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    calculate_volatility_loss_contributions,
    calculate_volatility_metrics,
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "lstm_risk_feature_comparison_predictions.csv"
)
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "dependence_robust_comparison.md"

COMPARISONS = [
    ("Risk-feature + regime LSTM", "Risk-feature LSTM"),
    ("Regime-aware LSTM", "LSTM"),
    ("Risk-feature + regime LSTM", "GJR-GARCH-Student-t"),
    ("Risk-feature LSTM", "GJR-GARCH-Student-t"),
]
LOSS_COLUMNS = {
    "MAE": "absolute_error",
    "Squared error": "squared_error",
    "QLIKE": "qlike",
}
HAC_LAGS = (4, 10, 20)
BOOTSTRAP_BLOCK_LENGTHS = (5, 21)
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_RANDOM_STATE = 42


def require(condition: bool, message: str) -> None:
    """Raise an actionable error for a violated analysis invariant."""
    if not condition:
        raise ValueError(message)


def validate_paired_inputs(
    actual: pd.Series,
    candidate: pd.Series,
    comparator: pd.Series,
) -> None:
    """Validate identical dates and valid volatility quantities."""
    if not isinstance(actual, pd.Series):
        raise TypeError("actual must be a pandas Series.")
    if not isinstance(candidate, pd.Series) or not isinstance(comparator, pd.Series):
        raise TypeError("Forecasts must be pandas Series.")
    if actual.empty:
        raise ValueError("Paired comparison requires at least one observation.")
    if not actual.index.equals(candidate.index) or not actual.index.equals(
        comparator.index
    ):
        raise ValueError("Actual and both forecasts must have identical dates.")
    if actual.index.has_duplicates or not actual.index.is_monotonic_increasing:
        raise ValueError("Comparison dates must be unique and chronological.")

    for name, values in {
        "actual": actual,
        "candidate": candidate,
        "comparator": comparator,
    }.items():
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric <= 0).any():
            raise ValueError(f"{name} must be finite and strictly positive.")


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load the combined saved forecasts and validate the analysis schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    data = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    required = {"actual", "fold", *(name for pair in COMPARISONS for name in pair)}
    missing = required.difference(data.columns)
    require(not missing, f"Prediction file is missing columns: {sorted(missing)}")
    require(
        data.index.is_unique and data.index.is_monotonic_increasing,
        "Prediction dates must be unique and chronological.",
    )
    require(data.index.notna().all(), "Prediction dates must be valid.")

    numeric_columns = ["actual", *(name for pair in COMPARISONS for name in pair)]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        values = data[column].to_numpy(dtype=float)
        require(
            np.isfinite(values).all() and (values > 0).all(),
            f"Column {column!r} must be finite and strictly positive.",
        )

    data["fold"] = pd.to_numeric(data["fold"], errors="coerce")
    require(data["fold"].notna().all(), "Fold identifiers must be numeric.")
    require(
        np.equal(data["fold"], np.floor(data["fold"])).all(),
        "Fold identifiers must be integers.",
    )
    data["fold"] = data["fold"].astype(int)
    require(set(data["fold"]) == set(range(1, 121)), "Expected folds 1 through 120.")
    require(data.groupby("fold").size().eq(21).all(), "Expected 21 rows per fold.")
    require(
        np.array_equal(
            data.loc[data["fold"].ne(data["fold"].shift()), "fold"].to_numpy(),
            np.arange(1, 121),
        ),
        "Each fold must be one chronological block in order.",
    )
    return data


def hac_mean_difference(loss_difference: pd.Series, max_lag: int) -> dict[str, float]:
    """Estimate a mean loss difference with a Bartlett Newey-West HAC CI."""
    if not isinstance(max_lag, int) or isinstance(max_lag, bool) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer.")
    values = loss_difference.to_numpy(dtype=float)
    if len(values) <= max_lag + 1:
        raise ValueError("Not enough observations for the requested HAC lag.")
    if not np.isfinite(values).all():
        raise ValueError("Loss differences contain non-finite values.")

    fitted = sm.OLS(values, np.ones((len(values), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lag, "use_correction": True},
        use_t=True,
    )
    lower, upper = fitted.conf_int(alpha=0.05)[0]
    return {
        "mean_loss_difference_comparator_minus_candidate": float(values.mean()),
        "hac_standard_error": float(fitted.bse[0]),
        "test_statistic": float(fitted.tvalues[0]),
        "p_value_two_sided": float(fitted.pvalues[0]),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm family-wise-error adjusted p-values in original order."""
    if p_values.empty or not np.isfinite(p_values.to_numpy(dtype=float)).all():
        raise ValueError("Holm correction requires finite non-empty p-values.")
    if ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("P-values must lie between zero and one.")

    ordered = p_values.sort_values(kind="stable")
    adjusted = np.maximum.accumulate(
        ordered.to_numpy(dtype=float) * np.arange(len(ordered), 0, -1)
    ).clip(0.0, 1.0)
    return pd.Series(adjusted, index=ordered.index).reindex(p_values.index)


def moving_block_bootstrap_ci(
    loss_difference: pd.Series,
    block_length: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> dict[str, float]:
    """Deterministic percentile CI from chronological moving blocks."""
    values = loss_difference.to_numpy(dtype=float)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("Loss differences must be finite and non-empty.")
    if not isinstance(block_length, int) or block_length < 1 or block_length > len(values):
        raise ValueError("block_length must be between one and sample length.")
    if not isinstance(repetitions, int) or repetitions < 100:
        raise ValueError("repetitions must be an integer of at least 100.")

    n_blocks = int(np.ceil(len(values) / block_length))
    starts = np.random.default_rng(random_state).integers(
        0, len(values) - block_length + 1, size=(repetitions, n_blocks)
    )
    indices = (starts[:, :, None] + np.arange(block_length)).reshape(
        repetitions, -1
    )
    means = values[indices][:, : len(values)].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "bootstrap_mean": float(means.mean()),
        "bootstrap_ci_95_lower": float(lower),
        "bootstrap_ci_95_upper": float(upper),
    }


def build_results(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate paired HAC and moving-block-bootstrap summaries."""
    records: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []

    for candidate_name, comparator_name in COMPARISONS:
        actual = data["actual"]
        candidate = data[candidate_name]
        comparator = data[comparator_name]
        validate_paired_inputs(actual, candidate, comparator)
        candidate_metrics = calculate_volatility_metrics(actual, candidate)
        comparator_metrics = calculate_volatility_metrics(actual, comparator)
        candidate_losses = calculate_volatility_loss_contributions(actual, candidate)
        comparator_losses = calculate_volatility_loss_contributions(actual, comparator)

        for loss_name, loss_column in LOSS_COLUMNS.items():
            # Positive differences mean comparator loss exceeds candidate loss,
            # so positive values favour the named candidate.
            difference = comparator_losses[loss_column] - candidate_losses[loss_column]
            base = {
                "candidate": candidate_name,
                "comparator": comparator_name,
                "loss": loss_name,
                "n_observations": len(data),
                "candidate_mae": candidate_metrics["mae"],
                "candidate_rmse": candidate_metrics["rmse"],
                "candidate_qlike": candidate_metrics["qlike"],
                "comparator_mae": comparator_metrics["mae"],
                "comparator_rmse": comparator_metrics["rmse"],
                "comparator_qlike": comparator_metrics["qlike"],
                "candidate_mean_squared_error": float(candidate_losses["squared_error"].mean()),
                "comparator_mean_squared_error": float(comparator_losses["squared_error"].mean()),
            }
            for lag in HAC_LAGS:
                records.append(
                    {**base, "hac_max_lag": lag, **hac_mean_difference(difference, lag)}
                )
            for block_length in BOOTSTRAP_BLOCK_LENGTHS:
                bootstrap_records.append(
                    {
                        **base,
                        "block_length": block_length,
                        "repetitions": BOOTSTRAP_REPETITIONS,
                        **moving_block_bootstrap_ci(difference, block_length),
                    }
                )

    results = pd.DataFrame(records)
    results["p_value_holm"] = np.nan
    for lag in HAC_LAGS:
        mask = results["hac_max_lag"].eq(lag)
        results.loc[mask, "p_value_holm"] = holm_adjust(
            results.loc[mask, "p_value_two_sided"]
        ).to_numpy()
    results["ci_excludes_zero"] = (
        (results["ci_95_lower"] > 0) | (results["ci_95_upper"] < 0)
    )
    results["holm_reject_05"] = results["p_value_holm"] < 0.05
    return results, pd.DataFrame(bootstrap_records)


def plot_hac_lag_four(results: pd.DataFrame) -> None:
    """Plot primary-lag effects and HAC confidence intervals by loss."""
    primary = results.loc[results["hac_max_lag"].eq(4)].copy()
    fig, axes = plt.subplots(1, len(LOSS_COLUMNS), figsize=(16, 5))
    for axis, loss_name in zip(axes, LOSS_COLUMNS):
        subset = primary.loc[primary["loss"].eq(loss_name)].reset_index(drop=True)
        y = np.arange(len(subset))
        means = subset["mean_loss_difference_comparator_minus_candidate"].to_numpy()
        lower = subset["ci_95_lower"].to_numpy()
        upper = subset["ci_95_upper"].to_numpy()
        axis.errorbar(means, y, xerr=[means - lower, upper - means], fmt="o", capsize=3)
        axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
        labels = [f"{row.candidate}\nvs {row.comparator}" for row in subset.itertuples()]
        axis.set_yticks(y, labels)
        axis.set_title(loss_name)
        axis.set_xlabel("Comparator loss − candidate loss")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "Exploratory paired loss differences: HAC lag 4, 95% CI\n"
        "Positive values favour the candidate"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "dependence_robust_hac_effects.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_report(data: pd.DataFrame, results: pd.DataFrame) -> None:
    """Write a concise report that preserves the exploratory interpretation."""
    primary = results.loc[results["hac_max_lag"].eq(4)]
    lines = [
        "# Dependence-robust forecast comparison",
        "",
        "## Scope and status",
        "",
        f"- Forecast origins: {len(data):,}, from {data.index.min().date()} to {data.index.max().date()}.",
        "- Comparisons use identical chronological dates and actual targets.",
        "- This is exploratory: the evaluation period has informed model development.",
        "- Positive mean differences are `comparator loss − candidate loss`; positive values favour the candidate.",
        "",
        "## Method",
        "",
        "Per-origin absolute error, squared error, and variance-based QLIKE are paired by date. "
        "The reported intervals use an intercept-only OLS mean with Bartlett Newey-West/HAC covariance and finite-sample correction. "
        "Lags 4, 10, and 20 are reported; lag 4 is the prespecified overlap-motivated primary lag. "
        "Holm correction is applied across the 12 comparison-by-loss hypotheses separately at each HAC lag. "
        "Moving-block bootstrap percentile intervals (2,000 deterministic repetitions; chronological blocks of 5 and 21) are sensitivity analyses.",
        "",
        "## Primary HAC results (lag 4)",
        "",
        "| Candidate | Comparator | Loss | Mean difference | 95% HAC CI | Unadjusted p | Holm p |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"| {row.candidate} | {row.comparator} | {row.loss} | "
            f"{row.mean_loss_difference_comparator_minus_candidate:.8g} | "
            f"[{row.ci_95_lower:.8g}, {row.ci_95_upper:.8g}] | "
            f"{row.p_value_two_sided:.4g} | {row.p_value_holm:.4g} |"
        )
    lines.extend(
        [
            "",
            "## HAC lag sensitivity",
            "",
            "| Candidate | Comparator | Loss | Lags with CI excluding zero | Lags with Holm p < 0.05 |",
            "|---|---|---|---|---|",
        ]
    )
    for (candidate, comparator, loss), group in results.groupby(
        ["candidate", "comparator", "loss"], sort=False
    ):
        ci_lags = ", ".join(
            str(value)
            for value in group.loc[group["ci_excludes_zero"], "hac_max_lag"]
        ) or "none"
        holm_lags = ", ".join(
            str(value)
            for value in group.loc[group["holm_reject_05"], "hac_max_lag"]
        ) or "none"
        lines.append(
            f"| {candidate} | {comparator} | {loss} | {ci_lags} | {holm_lags} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The lag-sensitivity table shows that some lag-4 findings do not remain after broader HAC dependence allowances or Holm adjustment. Therefore the analysis does not support a general claim that regime indicators improve LSTM forecasting. The QLIKE disadvantage of both risk-feature LSTM variants against GJR-GARCH is directionally persistent, but its Holm-adjusted conclusion is also lag-sensitive. A non-significant result is not evidence of equivalence, and a statistically distinguishable loss difference does not establish economic usefulness. Fold-win counts are not used as independent statistical evidence. Overlapping targets induce serial dependence, and all intervals and p-values remain exploratory because this period informed development.",
            "",
            "## Outputs",
            "",
            "- `reports/tables/dependence_robust_paired_loss_tests.csv`",
            "- `reports/tables/dependence_robust_moving_block_bootstrap.csv`",
            "- `reports/figures/dependence_robust_hac_effects.png`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_predictions()
    results, bootstrap = build_results(data)
    results.to_csv(
        TABLES_DIR / "dependence_robust_paired_loss_tests.csv", index=False
    )
    bootstrap.to_csv(
        TABLES_DIR / "dependence_robust_moving_block_bootstrap.csv", index=False
    )
    plot_hac_lag_four(results)
    write_report(data, results)
    print("Exploratory dependence-robust comparison completed.")


if __name__ == "__main__":
    main()
