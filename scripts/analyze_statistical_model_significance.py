
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "walk_forward_statistical_models_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "gjr_garch_hac_significance.csv"
)

CANDIDATE = "GJR-GARCH-Student-t"
COMPARATORS = ["GARCH-normal", "EWMA"]

# Five-day forward targets overlap for up to four observation lags.
# Lag 20 is included as a sensitivity check for longer dependence.
HAC_LAGS = [4, 20]


def calculate_losses(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate observation-level losses for paired model comparison."""
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("Actual values and forecasts must be finite.")

    if np.any(actual <= 0) or np.any(forecast <= 0):
        raise ValueError("Actual volatility and forecasts must be positive.")

    error = actual - forecast
    ratio = (actual**2) / (forecast**2)

    return {
        "MAE": np.abs(error),
        # MSE is tested because it is the per-observation loss underlying RMSE.
        "Squared error": error**2,
        "QLIKE": ratio - np.log(ratio) - 1.0,
    }


def hac_mean_difference(
    loss_difference: np.ndarray,
    max_lag: int,
) -> dict[str, float]:
    """
    Test whether the mean paired loss difference differs from zero.

    Negative values mean the candidate model has lower loss than
    its comparator. HAC covariance provides autocorrelation-robust
    standard errors.
    """
    if len(loss_difference) <= max_lag + 1:
        raise ValueError("Not enough observations for the requested HAC lag.")

    if not np.isfinite(loss_difference).all():
        raise ValueError("Loss differences contain non-finite values.")

    design = np.ones((len(loss_difference), 1))

    fitted = sm.OLS(loss_difference, design).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": max_lag,
            "use_correction": True,
        },
        use_t=True,
    )

    confidence_interval = fitted.conf_int(alpha=0.05)[0]

    return {
        "mean_loss_difference_candidate_minus_comparator": float(
            np.mean(loss_difference)
        ),
        "hac_standard_error": float(fitted.bse[0]),
        "test_statistic": float(fitted.tvalues[0]),
        "p_value_two_sided": float(fitted.pvalues[0]),
        "ci_95_lower": float(confidence_interval[0]),
        "ci_95_upper": float(confidence_interval[1]),
    }


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}\n"
            "Run scripts/compare_statistical_models.py first."
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    required = ["actual", CANDIDATE, *COMPARATORS]
    missing = set(required).difference(predictions.columns)

    if missing:
        raise ValueError(
            f"Predictions file is missing columns: {sorted(missing)}"
        )

    values = predictions[required].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Predictions contain missing or non-finite values.")

    actual = predictions["actual"].to_numpy(dtype=float)
    candidate_forecast = predictions[CANDIDATE].to_numpy(dtype=float)
    candidate_losses = calculate_losses(actual, candidate_forecast)

    records: list[dict[str, object]] = []

    for comparator in COMPARATORS:
        comparator_forecast = predictions[comparator].to_numpy(dtype=float)
        comparator_losses = calculate_losses(actual, comparator_forecast)

        for loss_name in candidate_losses:
            # Paired comparison: both models are evaluated on the same dates.
            difference = (
                candidate_losses[loss_name]
                - comparator_losses[loss_name]
            )

            for lag in HAC_LAGS:
                results = hac_mean_difference(difference, lag)

                records.append(
                    {
                        "candidate": CANDIDATE,
                        "comparator": comparator,
                        "loss": loss_name,
                        "hac_max_lag": lag,
                        "n_observations": len(difference),
                        **results,
                    }
                )

    result_table = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(OUTPUT_PATH, index=False)

    print("PAIRED HAC-ROBUST LOSS COMPARISONS")
    print("==================================")
    print(f"Candidate: {CANDIDATE}")
    print(f"Evaluation observations: {len(predictions)}")
    print("Negative mean differences favour the candidate.")
    print("P-values are two-sided; lag 20 is a dependence sensitivity check.\n")

    columns = [
        "comparator",
        "loss",
        "hac_max_lag",
        "mean_loss_difference_candidate_minus_comparator",
        "hac_standard_error",
        "p_value_two_sided",
        "ci_95_lower",
        "ci_95_upper",
    ]

    print(
        result_table[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    print(f"\nSaved results: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
