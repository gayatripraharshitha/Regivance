from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_volatility_metrics


PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "lstm_risk_feature_comparison_predictions.csv"
)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORT_PATH = PROJECT_ROOT / "reports" / "forecast_failure_attribution.md"

STATISTICAL_MODELS = [
    "Rolling",
    "EWMA",
    "GARCH-normal",
    "GARCH-Student-t",
    "GJR-GARCH-Student-t",
]

NEURAL_MODELS = [
    "LSTM",
    "Regime-aware LSTM",
    "Risk-feature LSTM",
    "Risk-feature + regime LSTM",
]

MODELS = STATISTICAL_MODELS + NEURAL_MODELS

REFERENCE_MODEL = "GJR-GARCH-Student-t"
FOCUS_MODEL = "Risk-feature + regime LSTM"
RISK_FEATURE_MODEL = "Risk-feature LSTM"


def require(condition: bool, message: str) -> None:
    """Raise an explicit error when an analysis assumption is violated."""
    if not condition:
        raise ValueError(message)


def load_predictions() -> pd.DataFrame:
    """Load the existing experiment and validate all required inputs."""
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}\n"
            "Run scripts/compare_lstm_risk_features.py first."
        )

    data = pd.read_csv(PREDICTIONS_PATH)

    required = {
        "date",
        "actual",
        "fold",
        "regime",
        *MODELS,
    }

    missing = required.difference(data.columns)
    require(
        not missing,
        f"Prediction file is missing columns: {sorted(missing)}",
    )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    require(
        not data["date"].isna().any(),
        "Prediction dates contain missing or invalid values.",
    )
    require(
        data["date"].is_unique,
        "Prediction dates must be unique.",
    )
    require(
        data["date"].is_monotonic_increasing,
        "Prediction dates must be chronological.",
    )

    numeric_columns = ["actual", *MODELS, "fold"]

    for column in numeric_columns:
        converted = pd.to_numeric(data[column], errors="coerce")

        require(
            not converted.isna().any(),
            f"Column {column!r} contains missing or non-numeric values.",
        )
        require(
            np.isfinite(converted.to_numpy(dtype=float)).all(),
            f"Column {column!r} contains non-finite values.",
        )

        data[column] = converted

    require(
        (data["actual"] > 0).all(),
        "Actual volatility targets must be positive.",
    )

    for model in MODELS:
        require(
            (data[model] > 0).all(),
            f"Forecasts from {model!r} must be positive for QLIKE.",
        )

    require(
        np.equal(data["fold"], np.floor(data["fold"])).all(),
        "Fold identifiers must be integers.",
    )
    data["fold"] = data["fold"].astype(int)

    require(
        data["fold"].nunique() == 120,
        "Expected 120 walk-forward folds. Check the prediction file.",
    )

    fold_sizes = data.groupby("fold").size()
    require(
        fold_sizes.eq(21).all(),
        "Expected exactly 21 test observations per fold.",
    )

    expected_folds = set(range(1, 121))
    observed_folds = set(data["fold"])
    require(
        observed_folds == expected_folds,
        "Fold identifiers must run consecutively from 1 through 120.",
    )

    # Each walk-forward test set must be one chronological block.  Merely
    # checking per-fold counts would allow interleaved labels, which would
    # make a fold-level attribution (including the Fold 43 diagnostic)
    # scientifically meaningless.
    fold_blocks = data.loc[
        data["fold"].ne(data["fold"].shift()), "fold"
    ].to_numpy()
    require(
        np.array_equal(fold_blocks, np.arange(1, 121)),
        "Each fold must be one chronological block in order from 1 through 120.",
    )

    return data


def add_loss_contributions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate per-observation MAE, squared error and QLIKE.

    QLIKE uses the same variance ratio as the project metrics:
        ratio = actual^2 / forecast^2
        loss = ratio - log(ratio) - 1

    Actual and forecast are volatilities, hence the squared ratio.
    """
    result = data.copy()
    actual = result["actual"].to_numpy(dtype=float)

    for model in MODELS:
        forecast = result[model].to_numpy(dtype=float)
        ratio = np.square(actual / forecast)

        require(
            np.isfinite(ratio).all() and (ratio > 0).all(),
            f"Invalid variance ratios calculated for {model!r}.",
        )

        loss = ratio - np.log(ratio) - 1.0

        require(
            np.isfinite(loss).all(),
            f"QLIKE contributions for {model!r} are non-finite.",
        )

        # Numerical round-off can create tiny negative values near zero.
        # This is a tolerance for floating-point precision, not a forecast
        # adjustment or clipping procedure.
        require(
            (loss >= -1e-10).all(),
            f"Materially negative QLIKE contribution for {model!r}.",
        )

        loss = np.where(loss < 0.0, 0.0, loss)

        result[f"qlike__{model}"] = loss
        result[f"absolute_error__{model}"] = np.abs(actual - forecast)
        result[f"squared_error__{model}"] = np.square(actual - forecast)

        pooled = calculate_volatility_metrics(
            result["actual"],
            result[model],
        )

        require(
            np.isclose(
                loss.mean(),
                pooled["qlike"],
                rtol=1e-8,
                atol=1e-10,
            ),
            f"Per-observation QLIKE does not reproduce the pooled "
            f"metric for {model!r}.",
        )

    return result


def calculate_concentration(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Measure how much pooled QLIKE is contributed by worst observations."""
    records = []
    fractions = (0.01, 0.05, 0.10)

    for model in MODELS:
        column = f"qlike__{model}"
        losses = data[column].sort_values(ascending=False).to_numpy()
        total = float(losses.sum())

        require(
            np.isfinite(total) and total > 0,
            f"Total QLIKE for {model!r} must be positive and finite.",
        )

        record = {
            "model": model,
            "n_observations": len(losses),
            "mean_qlike": float(losses.mean()),
            "median_qlike": float(np.median(losses)),
            "maximum_daily_qlike": float(losses[0]),
        }

        for fraction in fractions:
            count = max(1, int(np.ceil(fraction * len(losses))))
            top_sum = float(losses[:count].sum())

            label = f"top_{int(fraction * 100)}pct"

            record[f"{label}_count"] = count
            record[f"{label}_loss_share_pct"] = (
                100.0 * top_sum / total
            )

        records.append(record)

    return pd.DataFrame(records)


def calculate_fold_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each model's losses within each test fold."""
    records = []

    for fold_number, group in data.groupby("fold", sort=True):
        for model in MODELS:
            qlike = group[f"qlike__{model}"].to_numpy(dtype=float)
            absolute_error = group[
                f"absolute_error__{model}"
            ].to_numpy(dtype=float)
            squared_error = group[
                f"squared_error__{model}"
            ].to_numpy(dtype=float)

            worst_count = min(10, len(group))
            worst_loss_sum = float(
                np.sort(qlike)[-worst_count:].sum()
            )
            total_loss = float(qlike.sum())

            records.append(
                {
                    "fold": int(fold_number),
                    "model": model,
                    "n_test": len(group),
                    "mean_qlike": float(qlike.mean()),
                    "median_daily_qlike": float(np.median(qlike)),
                    "maximum_daily_qlike": float(qlike.max()),
                    "top_10_observations_loss_share_pct": (
                        100.0 * worst_loss_sum / total_loss
                        if total_loss > 0
                        else np.nan
                    ),
                    "mae": float(absolute_error.mean()),
                    "rmse": float(np.sqrt(squared_error.mean())),
                }
            )

    return pd.DataFrame(records)


def save_top_dates(data: pd.DataFrame) -> None:
    """Export the global worst dates and the ten worst dates in Fold 43."""
    qlike_column = f"qlike__{FOCUS_MODEL}"

    display_columns = [
        "date",
        "fold",
        "regime",
        "actual",
        REFERENCE_MODEL,
        RISK_FEATURE_MODEL,
        FOCUS_MODEL,
        f"qlike__{REFERENCE_MODEL}",
        f"qlike__{FOCUS_MODEL}",
        f"absolute_error__{FOCUS_MODEL}",
    ]

    global_worst = (
        data.nlargest(50, qlike_column)[display_columns]
        .sort_values(qlike_column, ascending=False)
    )

    global_worst.to_csv(
        TABLES_DIR / "forecast_failure_top_50_dates.csv",
        index=False,
    )

    fold_43 = data.loc[data["fold"].eq(43)]

    require(
        len(fold_43) == 21,
        "Fold 43 should contain 21 observations.",
    )

    fold_43_top = (
        fold_43.nlargest(10, qlike_column)[display_columns]
        .sort_values(qlike_column, ascending=False)
    )

    fold_43_top.to_csv(
        TABLES_DIR / "forecast_failure_fold43_top10.csv",
        index=False,
    )

    top_ten_sum = float(fold_43_top[qlike_column].sum())
    fold_total = float(fold_43[qlike_column].sum())
    share = 100.0 * top_ten_sum / fold_total

    print("\nFOLD 43 LOSS CONCENTRATION")
    print("=========================")
    print(f"Top 10 daily losses: {top_ten_sum:.6f}")
    print(f"Total Fold 43 loss:  {fold_total:.6f}")
    print(f"Top 10 share:        {share:.2f}%")

    print("\nTop dates:")
    print(
        fold_43_top[
            ["date", "actual", FOCUS_MODEL, qlike_column]
        ].to_string(
            index=False,
            formatters={
                "actual": "{:.6f}".format,
                FOCUS_MODEL: "{:.6f}".format,
                qlike_column: "{:.6f}".format,
            },
        )
    )


def plot_daily_losses(
    data: pd.DataFrame,
) -> None:
    """Plot daily QLIKE contributions for the two main comparison models."""
    fig, ax = plt.subplots(figsize=(13, 6))

    for model in [REFERENCE_MODEL, FOCUS_MODEL]:
        ax.plot(
            data["date"],
            data[f"qlike__{model}"].clip(lower=1e-10),
            label=model,
            linewidth=1.0,
            alpha=0.8,
        )

    # Highlight the known COVID-19 stress window as context.
    ax.axvspan(
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-04-30"),
        alpha=0.12,
        label="Feb–Apr 2020 reference period",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Forecast origin date")
    ax.set_ylabel("Daily QLIKE contribution (log scale)")
    ax.set_title(
        "Daily forecast-loss contributions\n"
        "Lower values are better; target windows overlap"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "forecast_failure_daily_qlike.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_loss_concentration(
    data: pd.DataFrame,
) -> None:
    """
    Plot cumulative QLIKE share against the fraction of observations,
    ordering each model's observations from worst loss to best.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for model in [REFERENCE_MODEL, FOCUS_MODEL]:
        losses = data[f"qlike__{model}"].sort_values(
            ascending=False
        ).to_numpy()

        cumulative_share = np.cumsum(losses) / losses.sum() * 100.0
        observation_share = (
            np.arange(1, len(losses) + 1) / len(losses) * 100.0
        )

        ax.plot(
            observation_share,
            cumulative_share,
            label=model,
            linewidth=2,
        )

    ax.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        linewidth=1,
        label="Uniform loss contribution",
    )

    ax.set_xlabel("Worst observations included (%)")
    ax.set_ylabel("Cumulative share of total QLIKE (%)")
    ax.set_title(
        "Concentration of forecast loss\n"
        "Curves farther above the diagonal indicate more concentrated loss"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "forecast_failure_loss_concentration.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_worst_dates(
    data: pd.DataFrame,
) -> None:
    """Visualize the worst observations for the focus LSTM variant."""
    qlike_column = f"qlike__{FOCUS_MODEL}"

    worst = data.nlargest(15, qlike_column).copy()
    worst["date_label"] = worst["date"].dt.strftime("%Y-%m-%d")
    worst = worst.sort_values(qlike_column, ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(
        worst["date_label"],
        worst[qlike_column],
    )

    ax.set_xlabel("Daily QLIKE contribution")
    ax.set_ylabel("Forecast origin date")
    ax.set_title(
        "15 worst daily QLIKE contributions\n"
        "Risk-feature + regime LSTM"
    )

    for bar, value in zip(bars, worst[qlike_column]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.2f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "forecast_failure_worst_dates.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_report(
    data: pd.DataFrame,
    concentration: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> None:
    focus = concentration.loc[
        concentration["model"].eq(FOCUS_MODEL)
    ].iloc[0]

    baseline = concentration.loc[
        concentration["model"].eq(REFERENCE_MODEL)
    ].iloc[0]

    fold_43 = data.loc[data["fold"].eq(43)]
    fold_43_losses = fold_43[f"qlike__{FOCUS_MODEL}"]

    qlike_focus = float(
        data[f"qlike__{FOCUS_MODEL}"].mean()
    )
    qlike_baseline = float(
        data[f"qlike__{REFERENCE_MODEL}"].mean()
    )

    report = [
        "# Forecast failure attribution",
        "",
        "## Scope",
        "",
        f"- Predictions: {len(data):,}",
        f"- Walk-forward folds: {data['fold'].nunique()}",
        f"- Evaluation dates: {data['date'].min().date()} to "
        f"{data['date'].max().date()}",
        f"- Focus model: {FOCUS_MODEL}",
        f"- Reference model: {REFERENCE_MODEL}",
        "",
        "The evaluation period has already informed model development. "
        "These are exploratory diagnostics, not a final untouched test.",
        "",
        "## Pooled QLIKE",
        "",
        "| Model | Mean QLIKE |",
        "|---|---:|",
        f"| {REFERENCE_MODEL} | {qlike_baseline:.8f} |",
        f"| {FOCUS_MODEL} | {qlike_focus:.8f} |",
        "",
        "## Loss concentration",
        "",
        "| Model | Worst 1% share | Worst 5% share | Worst 10% share |",
        "|---|---:|---:|---:|",
        (
            f"| {REFERENCE_MODEL} | "
            f"{baseline['top_1pct_loss_share_pct']:.2f}% | "
            f"{baseline['top_5pct_loss_share_pct']:.2f}% | "
            f"{baseline['top_10pct_loss_share_pct']:.2f}% |"
        ),
        (
            f"| {FOCUS_MODEL} | "
            f"{focus['top_1pct_loss_share_pct']:.2f}% | "
            f"{focus['top_5pct_loss_share_pct']:.2f}% | "
            f"{focus['top_10pct_loss_share_pct']:.2f}% |"
        ),
        "",
        "## Fold 43 diagnostic",
        "",
        "Fold 43 spans February 18 to March 17, 2020. It contains "
        f"{len(fold_43)} forecast origins.",
        "",
        f"- Mean daily QLIKE for the focus model: "
        f"{fold_43_losses.mean():.6f}",
        f"- Median daily QLIKE for the focus model: "
        f"{fold_43_losses.median():.6f}",
        f"- Maximum daily QLIKE for the focus model: "
        f"{fold_43_losses.max():.6f}",
        "- The top-ten loss share is calculated from the 10 largest "
        "daily QLIKE contributions divided by the fold's total QLIKE.",
        "",
        "## Interpretation limitations",
        "",
        "The five-observation targets use overlapping future return "
        "windows. Consequently, adjacent forecast errors are dependent, "
        "and one market shock may contribute to several successive "
        "large losses. The loss-concentration results are descriptive; "
        "they do not establish statistical significance or identify "
        "causality.",
        "",
        "This analysis does not change or clip forecasts. It attributes "
        "loss using the predictions that were already generated.",
        "",
        "## Generated outputs",
        "",
        "- `reports/tables/forecast_failure_daily_losses.csv`",
        "- `reports/tables/forecast_failure_concentration.csv`",
        "- `reports/tables/forecast_failure_fold_summary.csv`",
        "- `reports/tables/forecast_failure_top_50_dates.csv`",
        "- `reports/tables/forecast_failure_fold43_top10.csv`",
        "- `reports/figures/forecast_failure_daily_qlike.png`",
        "- `reports/figures/forecast_failure_loss_concentration.png`",
        "- `reports/figures/forecast_failure_worst_dates.png`",
        "",
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and validating existing forecasts...")
    predictions = load_predictions()

    print("Calculating observation-level loss contributions...")
    results = add_loss_contributions(predictions)

    results.to_csv(
        TABLES_DIR / "forecast_failure_daily_losses.csv",
        index=False,
    )

    concentration = calculate_concentration(results)
    concentration.to_csv(
        TABLES_DIR / "forecast_failure_concentration.csv",
        index=False,
    )

    fold_summary = calculate_fold_summary(results)
    fold_summary.to_csv(
        TABLES_DIR / "forecast_failure_fold_summary.csv",
        index=False,
    )

    save_top_dates(results)
    plot_daily_losses(results)
    plot_loss_concentration(results)
    plot_worst_dates(results)

    write_report(results, concentration, fold_summary)

    print("\nPOOLED LOSS-CONCENTRATION SUMMARY")
    print("=================================")
    print(
        concentration[
            [
                "model",
                "mean_qlike",
                "top_1pct_loss_share_pct",
                "top_5pct_loss_share_pct",
                "top_10pct_loss_share_pct",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nAll forecast failure analyses completed.")
    print("Report: reports/forecast_failure_attribution.md")


if __name__ == "__main__":
    main()
