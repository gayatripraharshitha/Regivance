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

from src.data.loader import load_industry_returns
from src.evaluation.metrics import calculate_volatility_metrics
from src.features.rolling_features import build_regime_features
from src.research_config import CONFIG


RAW_DATA_PATH = PROJECT_ROOT / CONFIG["data"]["raw_path"]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "portfolio_forecasting_dataset.csv"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "lstm_risk_feature_comparison_predictions.csv"
)
FOLD_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "lstm_risk_feature_comparison_fold_metrics.csv"
)

TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

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

ALL_MODELS = STATISTICAL_MODELS + NEURAL_MODELS
METRICS = ["mae", "rmse", "qlike"]

REGIME_LABELS = {"lower", "middle", "higher"}


def require(condition: bool, message: str) -> None:
    """Stop immediately with an actionable message if a check fails."""
    if not condition:
        raise ValueError(message)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Check the project directory and run the relevant "
            "data-preparation or model-evaluation script first."
        )


def record_check(
    checks: list[dict[str, str]],
    name: str,
    details: str,
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS",
            "details": details,
        }
    )
    print(f"[PASS] {name}: {details}")


def require_numeric_finite(
    frame: pd.DataFrame,
    columns: list[str],
    description: str,
) -> None:
    for column in columns:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce")

        # Invalid text must not silently become a valid observation.
        if converted.isna().any():
            invalid_count = int(converted.isna().sum())
            raise ValueError(
                f"{description}: column {column!r} contains "
                f"{invalid_count} missing or non-numeric values."
            )

        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{description}: column {column!r} contains "
                "infinite or otherwise non-finite values."
            )

        frame[column] = converted


def load_and_validate_predictions(
    checks: list[dict[str, str]],
) -> pd.DataFrame:
    require_file(PREDICTIONS_PATH)

    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "date",
        "actual",
        "fold",
        "regime",
        *ALL_MODELS,
    }

    missing_columns = required_columns.difference(predictions.columns)

    require(
        not missing_columns,
        f"Predictions are missing required columns: "
        f"{sorted(missing_columns)}",
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"],
        errors="coerce",
    )

    require(
        not predictions["date"].isna().any(),
        "Prediction dates contain missing or invalid dates.",
    )

    require(
        not predictions["date"].duplicated().any(),
        "Prediction dates are duplicated.",
    )

    require(
        predictions["date"].is_monotonic_increasing,
        "Prediction dates are not in chronological order.",
    )

    record_check(
        checks,
        "Prediction date integrity",
        "Dates are valid, unique and chronological.",
    )

    require_numeric_finite(
        predictions,
        ["actual", *ALL_MODELS],
        "Prediction values",
    )

    require(
        (predictions["actual"] > 0).all(),
        "Actual volatility targets must be strictly positive.",
    )

    for model_name in ALL_MODELS:
        if (predictions[model_name] <= 0).any():
            raise ValueError(
                f"{model_name!r} contains non-positive volatility "
                "forecasts. Do not hide this with clipping in the "
                "diagnostics script."
            )

    record_check(
        checks,
        "Numeric prediction validity",
        "Targets and every model forecast are finite and positive.",
    )

    fold_values = pd.to_numeric(
        predictions["fold"],
        errors="coerce",
    )

    require(
        not fold_values.isna().any(),
        "Fold numbers contain missing or invalid values.",
    )

    require(
        np.equal(
            fold_values.to_numpy(dtype=float),
            np.floor(fold_values.to_numpy(dtype=float)),
        ).all(),
        "Fold numbers must be integers.",
    )

    predictions["fold"] = fold_values.astype(int)

    require(
        predictions["fold"].min() >= 1,
        "Fold numbers must be positive.",
    )

    regimes = set(predictions["regime"].dropna().unique())

    require(
        predictions["regime"].notna().all()
        and regimes.issubset(REGIME_LABELS),
        f"Unexpected or missing regime labels: {sorted(regimes)}",
    )

    fold_counts = predictions.groupby("fold").size()

    require(
        fold_counts.eq(21).all(),
        "Expected 21 test predictions in every fold; observed counts "
        f"are {fold_counts.value_counts().to_dict()}.",
    )

    record_check(
        checks,
        "Fold and regime integrity",
        f"{predictions['fold'].nunique()} folds; "
        "21 predictions per fold; regime labels are valid.",
    )

    return predictions


def validate_target_alignment(
    predictions: pd.DataFrame,
    checks: list[dict[str, str]],
) -> None:
    require_file(DATASET_PATH)

    dataset = pd.read_csv(
        DATASET_PATH,
        index_col=0,
        parse_dates=True,
    )

    dataset.index = pd.DatetimeIndex(dataset.index)

    require(
        "target_volatility_5d" in dataset.columns,
        "Processed dataset lacks target_volatility_5d.",
    )

    require(
        dataset.index.is_unique,
        "Processed dataset contains duplicated dates.",
    )

    require(
        dataset.index.is_monotonic_increasing,
        "Processed dataset dates are not chronological.",
    )

    prediction_dates = pd.DatetimeIndex(predictions["date"])

    expected = dataset["target_volatility_5d"].reindex(
        prediction_dates
    )

    require(
        not expected.isna().any(),
        "Some prediction dates are absent from the processed target "
        "dataset.",
    )

    if not np.allclose(
        predictions["actual"].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Forecast actual values do not match the processed "
            "five-observation volatility targets. Check date alignment "
            "and target construction."
        )

    record_check(
        checks,
        "Target alignment",
        "All saved actual targets match the processed dataset by date.",
    )


def load_and_validate_risk_features(
    checks: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_file(RAW_DATA_PATH)

    # The project loader and feature builder perform their own
    # input validation. Their exceptions are intentionally not swallowed.
    industry_returns = load_industry_returns(RAW_DATA_PATH)

    risk_features = build_regime_features(
        industry_returns,
        window=21,
    )

    require(
        not industry_returns.empty,
        "Raw industry-return dataset is empty.",
    )

    require(
        industry_returns.index.equals(risk_features.index),
        "Risk-feature dates differ from raw return dates.",
    )

    require(
        np.isfinite(
            industry_returns.to_numpy(dtype=float)
        ).all(),
        "Raw industry returns contain non-finite values.",
    )

    record_check(
        checks,
        "Raw input and feature construction",
        "The project loader and 21-observation risk-feature builder "
        "accepted the data; feature dates match return dates.",
    )

    # Missing rolling features during initial warm-up are expected.
    # Missing values occurring later are not silently imputed.
    for column in risk_features.columns:
        values = risk_features[column]
        valid_positions = np.flatnonzero(values.notna().to_numpy())

        require(
            len(valid_positions) > 0,
            f"Risk feature {column!r} has no valid observations.",
        )

        first_valid = int(valid_positions[0])

        later_missing = values.iloc[first_valid:].isna()

        if later_missing.any():
            first_bad_date = values.index[
                first_valid + int(np.flatnonzero(later_missing.to_numpy())[0])
            ]

            raise ValueError(
                f"Risk feature {column!r} contains a missing value "
                f"after its initial warm-up: {first_bad_date.date()}. "
                "Investigate the source data or correlation calculation; "
                "do not automatically backfill or interpolate it."
            )

    feature_summary = pd.DataFrame(
        {
            "feature": risk_features.columns,
            "observations": len(risk_features),
            "missing_count": [
                int(risk_features[column].isna().sum())
                for column in risk_features.columns
            ],
            "missing_percent": [
                float(100 * risk_features[column].isna().mean())
                for column in risk_features.columns
            ],
            "minimum": [
                float(risk_features[column].min())
                for column in risk_features.columns
            ],
            "median": [
                float(risk_features[column].median())
                for column in risk_features.columns
            ],
            "maximum": [
                float(risk_features[column].max())
                for column in risk_features.columns
            ],
        }
    )

    record_check(
        checks,
        "Rolling-feature missingness",
        "Missing values, if any, are confined to each feature's "
        "initial rolling-window warm-up; no silent imputation occurred.",
    )

    return industry_returns, feature_summary, risk_features


def validate_saved_fold_metrics(
    predictions: pd.DataFrame,
    checks: list[dict[str, str]],
) -> pd.DataFrame:
    require_file(FOLD_METRICS_PATH)

    fold_metrics = pd.read_csv(FOLD_METRICS_PATH)

    required_columns = {
        "fold",
        "model",
        "n_test",
        *METRICS,
    }

    missing_columns = required_columns.difference(fold_metrics.columns)

    require(
        not missing_columns,
        "Fold-metrics file lacks columns: "
        f"{sorted(missing_columns)}",
    )

    fold_values = pd.to_numeric(
        fold_metrics["fold"],
        errors="coerce",
    )

    require(
        not fold_values.isna().any(),
        "Fold-metrics file contains invalid fold numbers.",
    )

    fold_metrics["fold"] = fold_values.astype(int)

    require_numeric_finite(
        fold_metrics,
        ["n_test", *METRICS],
        "Saved fold metrics",
    )

    require(
        (fold_metrics["n_test"] > 0).all(),
        "Fold metrics contain non-positive sample counts.",
    )

    require(
        (fold_metrics[METRICS] >= 0).all().all(),
        "Fold metrics contain negative error scores.",
    )

    require(
        not fold_metrics.duplicated(["fold", "model"]).any(),
        "Fold-metrics file contains duplicated fold/model pairs.",
    )

    expected_pairs = {
        (int(fold), model)
        for fold in predictions["fold"].unique()
        for model in ALL_MODELS
    }

    observed_pairs = set(
        zip(fold_metrics["fold"], fold_metrics["model"])
    )

    require(
        expected_pairs == observed_pairs,
        "The saved fold-metrics file does not contain exactly one "
        "record for every expected fold/model pair.",
    )

    saved_lookup = fold_metrics.set_index(["fold", "model"])

    for fold_number, fold_frame in predictions.groupby(
        "fold",
        sort=True,
    ):
        for model_name in ALL_MODELS:
            saved = saved_lookup.loc[(fold_number, model_name)]

            require(
                int(saved["n_test"]) == len(fold_frame),
                f"Fold {fold_number}, {model_name}: saved sample count "
                "does not match the prediction file.",
            )

            recalculated = calculate_volatility_metrics(
                fold_frame["actual"],
                fold_frame[model_name],
            )

            for metric in METRICS:
                if not np.isclose(
                    float(saved[metric]),
                    float(recalculated[metric]),
                    rtol=1e-6,
                    atol=1e-8,
                ):
                    raise ValueError(
                        f"Fold {fold_number}, model {model_name}, "
                        f"metric {metric}: saved score "
                        f"{saved[metric]:.10f} differs from recalculated "
                        f"score {recalculated[metric]:.10f}."
                    )

    record_check(
        checks,
        "Fold-metric reproducibility",
        "Every saved fold/model MAE, RMSE and QLIKE matches the "
        "score recalculated from the prediction rows.",
    )

    return fold_metrics


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {path.relative_to(PROJECT_ROOT)}")


def plot_overall_metrics(
    predictions: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    records = []

    for model in ALL_MODELS:
        records.append(
            {
                "model": model,
                "n_test": len(predictions),
                **calculate_volatility_metrics(
                    predictions["actual"],
                    predictions[model],
                ),
            }
        )

    overall = pd.DataFrame(records)

    overall.to_csv(
        tables_dir / "diagnostic_overall_metrics.csv",
        index=False,
    )

    for metric in METRICS:
        ordered = overall.sort_values(metric, ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(
            ordered["model"],
            ordered[metric],
        )

        ax.invert_yaxis()
        ax.set_xlabel(f"{metric.upper()} (lower is better)")
        ax.set_title(
            f"Pooled forecast comparison: {metric.upper()}\n"
            "Development evaluation period; not an untouched holdout"
        )

        for bar, value in zip(bars, ordered[metric]):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  {value:.5f}",
                va="center",
                fontsize=8,
            )

        save_figure(
            fig,
            figures_dir / f"model_comparison_{metric}.png",
        )

    return overall


def plot_qlike_fold_differences(
    fold_metrics: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    pivot = fold_metrics.pivot(
        index="fold",
        columns="model",
        values="qlike",
    ).sort_index()

    comparison = pd.DataFrame(
        {
            "fold": pivot.index,
            "GJR-GARCH-Student-t": pivot["GJR-GARCH-Student-t"],
            "Risk-feature + regime LSTM": pivot[
                "Risk-feature + regime LSTM"
            ],
        }
    )

    comparison["qlike_difference"] = (
        comparison["Risk-feature + regime LSTM"]
        - comparison["GJR-GARCH-Student-t"]
    )

    comparison.to_csv(
        TABLES_DIR / "diagnostic_fold_qlike_comparison.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.plot(
        comparison["fold"],
        comparison["qlike_difference"],
        marker=".",
        markersize=4,
        linewidth=1,
    )

    worst = comparison.nlargest(5, "qlike_difference")

    for _, row in worst.iterrows():
        ax.annotate(
            str(int(row["fold"])),
            (row["fold"], row["qlike_difference"]),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Walk-forward test fold")
    ax.set_ylabel("QLIKE difference: LSTM − GJR-GARCH")
    ax.set_title(
        "Fold-wise QLIKE differences\n"
        "Positive values favour GJR-GARCH; negative values favour LSTM"
    )

    save_figure(
        fig,
        figures_dir / "foldwise_qlike_difference.png",
    )

    return comparison


def plot_crash_period(
    predictions: pd.DataFrame,
    figures_dir: Path,
) -> None:
    start = pd.Timestamp("2020-02-01")
    end = pd.Timestamp("2020-06-30")

    period = predictions.loc[
        predictions["date"].between(start, end)
    ].copy()

    require(
        not period.empty,
        "No predictions found for the requested 2020 diagnostic period.",
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    series = [
        ("actual", "Actual future five-observation volatility"),
        ("GJR-GARCH-Student-t", "GJR-GARCH-Student-t"),
        ("Risk-feature LSTM", "Risk-feature LSTM"),
        (
            "Risk-feature + regime LSTM",
            "Risk-feature + regime LSTM",
        ),
    ]

    for column, label in series:
        ax.plot(
            period["date"],
            period[column],
            label=label,
            linewidth=1.5,
        )

    ax.set_xlabel("Forecast origin date")
    ax.set_ylabel("Five-observation volatility")
    ax.set_title(
        "Forecast behaviour during February–June 2020\n"
        "Actual values are future outcomes, shown only for retrospective "
        "evaluation"
    )
    ax.legend(loc="best")

    save_figure(
        fig,
        figures_dir / "forecasts_during_2020_shock.png",
    )


def plot_forecast_calibration(
    predictions: pd.DataFrame,
    figures_dir: Path,
) -> None:
    models = [
        "GJR-GARCH-Student-t",
        "Risk-feature + regime LSTM",
    ]

    actual = predictions["actual"].to_numpy(dtype=float)

    maximum = max(
        float(actual.max()),
        *[
            float(predictions[model].max())
            for model in models
        ],
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
    )

    for ax, model in zip(axes, models):
        forecast = predictions[model].to_numpy(dtype=float)

        ax.scatter(
            forecast,
            actual,
            alpha=0.4,
            s=12,
        )

        ax.plot(
            [0, maximum],
            [0, maximum],
            linestyle="--",
            linewidth=1,
        )

        ax.set_title(model)
        ax.set_xlabel("Forecast volatility")
        ax.set_ylabel("Actual future volatility")

    fig.suptitle(
        "Forecast calibration across all evaluation observations\n"
        "Points on the diagonal indicate exact forecasts"
    )

    save_figure(
        fig,
        figures_dir / "forecast_calibration_scatter.png",
    )


def plot_preprocessing_features(
    risk_features: pd.DataFrame,
    feature_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    # Focused period makes the changing market conditions visible.
    start = pd.Timestamp("2019-06-01")
    end = pd.Timestamp("2020-06-30")

    period = risk_features.loc[start:end]

    require(
        not period.empty,
        "Risk features do not cover the requested feature-plot period.",
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
    )

    for ax, column in zip(axes, risk_features.columns):
        ax.plot(
            period.index,
            period[column],
            linewidth=1.1,
        )
        ax.axvline(
            pd.Timestamp("2020-02-20"),
            linestyle="--",
            linewidth=1,
        )
        ax.set_ylabel(column.replace("_", "\n"))

    axes[-1].set_xlabel("Date")

    fig.suptitle(
        "Trailing risk features used in regime detection\n"
        "Dashed line marks a reference point near the onset of the "
        "2020 market shock"
    )

    save_figure(
        fig,
        figures_dir / "preprocessing_risk_features_2020.png",
    )

    summary = feature_summary.sort_values(
        "missing_count",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        summary["feature"],
        summary["missing_count"],
    )

    ax.set_ylabel("Missing observations")
    ax.set_title(
        "Risk-feature missingness from rolling-window warm-up\n"
        "Missing values are not automatically filled"
    )
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, summary["missing_count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(value)),
            ha="center",
            va="bottom",
        )

    save_figure(
        fig,
        figures_dir / "preprocessing_feature_missingness.png",
    )


def write_validation_report(
    checks: list[dict[str, str]],
    predictions: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    feature_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    check_frame = pd.DataFrame(checks)

    checks_path = TABLES_DIR / "diagnostic_validation_checks.csv"
    check_frame.to_csv(checks_path, index=False)

    feature_summary.to_csv(
        TABLES_DIR / "preprocessing_feature_summary.csv",
        index=False,
    )

    best_mae = overall_metrics.loc[overall_metrics["mae"].idxmin()]
    best_rmse = overall_metrics.loc[overall_metrics["rmse"].idxmin()]
    best_qlike = overall_metrics.loc[overall_metrics["qlike"].idxmin()]

    report_lines = [
        "# Data preprocessing and model diagnostics",
        "",
        "## Evaluation data",
        "",
        f"- Prediction rows: {len(predictions):,}",
        f"- Folds: {predictions['fold'].nunique()}",
        f"- Evaluation dates: {predictions['date'].min().date()} "
        f"to {predictions['date'].max().date()}",
        "- Results are development results, not an untouched final holdout.",
        "",
        "## Preprocessing approach",
        "",
        "- Industry returns are loaded through the project's data loader.",
        "- The forecasting target is checked against "
        "`target_volatility_5d` in the saved processed dataset.",
        "- Rolling risk features use a trailing 21-observation window.",
        "- The features are trailing portfolio volatility, mean "
        "industry correlation, and cross-sectional dispersion.",
        "- The rolling-window warm-up remains missing; no generic "
        "backfill, interpolation, or blanket imputation is performed.",
        "- In the LSTM experiment, continuous feature scalers are fitted "
        "on each fold's training dates only.",
        "- Regime detection is fitted on training features, then applied "
        "to later observations using the frozen detector.",
        "- Input sequences use a 21-observation lookback; labels remain "
        "aligned to the target date.",
        "",
        "## Error handling and integrity checks",
        "",
        "Inputs are checked for required files and columns, valid and "
        "unique dates, chronological ordering, numeric finite values, "
        "positive volatility targets and forecasts, valid regime labels, "
        "fold completeness, target alignment, and reproducibility of "
        "saved fold metrics.",
        "",
        "Invalid or unexpected missing values cause an explicit error "
        "instead of being silently replaced.",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]

    for _, row in check_frame.iterrows():
        details = str(row["details"]).replace("|", "\\|")
        report_lines.append(
            f"| {row['check']} | {row['status']} | {details} |"
        )

    report_lines.extend(
        [
            "",
            "## Pooled metrics",
            "",
            "| Metric | Best model in this evaluation | Score |",
            "|---|---|---:|",
            f"| MAE | {best_mae['model']} | {best_mae['mae']:.8f} |",
            f"| RMSE | {best_rmse['model']} | {best_rmse['rmse']:.8f} |",
            f"| QLIKE | {best_qlike['model']} | {best_qlike['qlike']:.8f} |",
            "",
            "These metrics summarise this evaluation only. They do not "
            "establish statistical significance or performance on a "
            "future untouched holdout.",
            "",
            "## Generated figures",
            "",
            "- `model_comparison_mae.png`",
            "- `model_comparison_rmse.png`",
            "- `model_comparison_qlike.png`",
            "- `foldwise_qlike_difference.png`",
            "- `forecasts_during_2020_shock.png`",
            "- `forecast_calibration_scatter.png`",
            "- `preprocessing_risk_features_2020.png`",
            "- `preprocessing_feature_missingness.png`",
            "",
            "## Interpretation note",
            "",
            "The 2020 period is shown as a diagnostic example, not as "
            "a separate tuning target. Fold-wise results should be read "
            "alongside pooled metrics because a small number of extreme "
            "errors can dominate a pooled loss.",
            "",
        ]
    )

    report_path = (
        PROJECT_ROOT / "reports" / "preprocessing_and_diagnostics.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print(f"\n[SAVED] {checks_path.relative_to(PROJECT_ROOT)}")
    print(
        "[SAVED] reports/tables/preprocessing_feature_summary.csv"
    )
    print(
        "[SAVED] reports/preprocessing_and_diagnostics.md"
    )


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, str]] = []

    print("Validating forecast outputs...")
    predictions = load_and_validate_predictions(checks)

    print("\nChecking actual targets against the processed dataset...")
    validate_target_alignment(predictions, checks)

    print("\nValidating raw inputs and preprocessing features...")
    _, feature_summary, risk_features = (
        load_and_validate_risk_features(checks)
    )

    print("\nRecalculating and validating saved fold metrics...")
    fold_metrics = validate_saved_fold_metrics(
        predictions,
        checks,
    )

    print("\nGenerating metric tables and figures...")
    overall_metrics = plot_overall_metrics(
        predictions,
        TABLES_DIR,
        FIGURES_DIR,
    )

    plot_qlike_fold_differences(
        fold_metrics,
        FIGURES_DIR,
    )

    plot_crash_period(
        predictions,
        FIGURES_DIR,
    )

    plot_forecast_calibration(
        predictions,
        FIGURES_DIR,
    )

    plot_preprocessing_features(
        risk_features,
        feature_summary,
        FIGURES_DIR,
    )

    write_validation_report(
        checks,
        predictions,
        overall_metrics,
        feature_summary,
        FIGURES_DIR,
    )

    print("\nALL DIAGNOSTIC CHECKS PASSED")
    print("============================")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Folds: {predictions['fold'].nunique()}")
    print(f"Validated checks: {len(checks)}")
    print(f"Figures directory: {FIGURES_DIR.relative_to(PROJECT_ROOT)}")
    print("Report: reports/preprocessing_and_diagnostics.md")


if __name__ == "__main__":
    main()
