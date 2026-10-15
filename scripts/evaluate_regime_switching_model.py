
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_statistical_models import (
    DATA_PATH,
    EMBARGO,
    FINAL_TEST_SIZE,
    GARCH_VARIANTS,
    HORIZON,
    STEP_SIZE,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
)
from src.data.loader import load_industry_returns
from src.data.portfolio import build_forecasting_dataset
from src.evaluation.metrics import calculate_volatility_metrics
from src.evaluation.splits import generate_walk_forward_splits
from src.features.regimes import MarketRegimeDetector
from src.features.rolling_features import build_regime_features
from src.models.ewma import forecast_ewma_volatility
from src.models.garch import forecast_garch_volatility_for_fold
from src.models.rolling_volatility import forecast_rolling_volatility


RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "30_Industry_Portfolios_Daily.csv"
)

BASELINE_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "walk_forward_statistical_models_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "tables"

VOLATILITY_FEATURE = "portfolio_volatility_21d"
REGIMES = ("lower", "middle", "higher")

# A regime with fewer validation observations uses the global
# validation winner instead of a fragile regime-specific selection.
MIN_VALIDATION_OBS_PER_REGIME = 20

MODEL_NAMES = [
    "Rolling",
    "EWMA",
    *[name for name, _, _ in GARCH_VARIANTS],
]


def load_inputs():
    """Load data, reconstruct features and verify existing test forecasts."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw industry data not found: {RAW_DATA_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecasting dataset not found: {DATA_PATH}. "
            "Run scripts/build_dataset.py first."
        )

    if not BASELINE_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Baseline predictions not found: {BASELINE_PREDICTIONS_PATH}. "
            "Run scripts/compare_statistical_models.py first."
        )

    industry_returns = load_industry_returns(RAW_DATA_PATH)
    dataset = build_forecasting_dataset(
        industry_returns,
        horizon=HORIZON,
    )

    saved_dataset = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    )
    saved_dataset.index = pd.DatetimeIndex(saved_dataset.index)

    if not saved_dataset.index.equals(dataset.index):
        raise ValueError(
            "The saved forecasting dataset does not match the "
            "dataset reconstructed from the raw returns."
        )

    if not np.allclose(
        saved_dataset["portfolio_return"].to_numpy(dtype=float),
        dataset["portfolio_return"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Saved portfolio returns differ from source data.")

    baseline_predictions = pd.read_csv(
        BASELINE_PREDICTIONS_PATH,
        index_col=0,
        parse_dates=True,
    )
    baseline_predictions.index = pd.DatetimeIndex(
        baseline_predictions.index
    )

    baseline_predictions = baseline_predictions.sort_index()

    required_columns = {"actual", *MODEL_NAMES}
    missing_columns = required_columns.difference(
        baseline_predictions.columns
    )

    if missing_columns:
        raise ValueError(
            "Baseline prediction file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    expected_dates = generate_walk_forward_splits(
        n_samples=len(dataset),
        train_size=TRAIN_SIZE,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        final_test_size=FINAL_TEST_SIZE,
        embargo=EMBARGO,
    )

    expected_test_dates = pd.DatetimeIndex(
        np.concatenate(
            [
                dataset.index[fold.test].to_numpy()
                for fold in expected_dates
            ]
        )
    )

    if not baseline_predictions.index.equals(expected_test_dates):
        raise ValueError(
            "Saved baseline predictions do not match the configured "
            "walk-forward test dates. Regenerate them first."
        )

    actual = dataset.loc[
        expected_test_dates, "target_volatility_5d"
    ].to_numpy(dtype=float)

    if not np.allclose(
        baseline_predictions["actual"].to_numpy(dtype=float),
        actual,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Saved target values differ from the reconstructed dataset."
        )

    if not np.isfinite(
        baseline_predictions[
            ["actual", *MODEL_NAMES]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Baseline predictions contain missing or non-finite values."
        )

    return industry_returns, dataset, baseline_predictions, expected_dates


def rank_cluster_labels(
    detector: MarketRegimeDetector,
) -> dict[int, str]:
    """Map arbitrary K-means IDs to relative training-volatility ranks."""
    centers = detector.cluster_centers()

    if VOLATILITY_FEATURE not in centers.columns:
        raise ValueError(
            f"Missing volatility feature: {VOLATILITY_FEATURE}"
        )

    ordered_ids = centers[VOLATILITY_FEATURE].sort_values(
        kind="stable"
    ).index.tolist()

    if len(ordered_ids) != len(REGIMES):
        raise ValueError(
            f"Expected {len(REGIMES)} clusters, got {len(ordered_ids)}."
        )

    return {
        int(cluster_id): regime
        for cluster_id, regime in zip(ordered_ids, REGIMES)
    }


def select_by_qlike(
    validation_rows: pd.DataFrame,
    model_names: list[str],
) -> tuple[str, dict[str, float]]:
    """Return the lowest-QLIKE model on a given validation subset."""
    scores = {}

    for model_name in model_names:
        scores[model_name] = calculate_volatility_metrics(
            validation_rows["actual"],
            validation_rows[model_name],
        )["qlike"]

    selected = min(model_names, key=lambda name: scores[name])

    return selected, scores


def main() -> None:
    (
        industry_returns,
        dataset,
        baseline_predictions,
        splits,
    ) = load_inputs()

    portfolio_returns = dataset["portfolio_return"].astype(float)
    actual_volatility = dataset["target_volatility_5d"].astype(float)

    # Features are causal: each row uses observations no later than
    # its own date. The scaler and cluster centers are still fitted
    # separately inside every fold below.
    features = build_regime_features(
        industry_returns,
        window=21,
    ).reindex(dataset.index)

    rolling_forecast = forecast_rolling_volatility(
        portfolio_returns,
        window=21,
        horizon=HORIZON,
    )

    ewma_forecast = forecast_ewma_volatility(
        portfolio_returns,
        decay=0.94,
        horizon=HORIZON,
        min_periods=21,
    )

    test_frames = []
    selection_records = []

    print(f"Evaluation folds: {len(splits)}")
    print(f"Test observations per model: {len(baseline_predictions)}")
    print("Primary selection metric: QLIKE")
    print(
        "Minimum regime-specific validation count: "
        f"{MIN_VALIDATION_OBS_PER_REGIME}"
    )
    print("\nBuilding validation-selected regime forecasts...")

    for split in splits:
        train_dates = dataset.index[split.train]
        validation_dates = dataset.index[split.validation]
        test_dates = dataset.index[split.test]

        training_returns = portfolio_returns.loc[train_dates]

        training_features = features.loc[train_dates].dropna()
        validation_features = features.loc[validation_dates]
        test_features = features.loc[test_dates]

        if training_features.empty:
            raise ValueError(
                f"Fold {split.fold}: no usable training regime features."
            )

        # Fit all data-dependent regime transformations on training data.
        detector = MarketRegimeDetector(
            n_regimes=3,
            random_state=42,
        ).fit(training_features)

        cluster_to_regime = rank_cluster_labels(detector)

        validation_regimes = detector.predict(
            validation_features
        ).map(cluster_to_regime)

        test_regimes = detector.predict(
            test_features
        ).map(cluster_to_regime)

        if validation_regimes.isna().any() or test_regimes.isna().any():
            raise ValueError(
                f"Fold {split.fold}: missing regime labels."
            )

        # Validation predictions for the two fixed-parameter baselines.
        validation_predictions = pd.DataFrame(
            {
                "Rolling": rolling_forecast.loc[validation_dates],
                "EWMA": ewma_forecast.loc[validation_dates],
            },
            index=validation_dates,
        )

        # Estimate GARCH parameters on training returns only.
        # Validation-period returns update each model's variance state,
        # but are never used to re-estimate its parameters.
        for model_name, distribution, asymmetric in GARCH_VARIANTS:
            try:
                validation_predictions[model_name] = (
                    forecast_garch_volatility_for_fold(
                        training_returns=training_returns,
                        all_returns=portfolio_returns,
                        forecast_dates=validation_dates,
                        horizon=HORIZON,
                        distribution=distribution,
                        asymmetric=asymmetric,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{model_name} failed to forecast validation "
                    f"fold {split.fold}."
                ) from exc

        validation_frame = pd.DataFrame(
            {
                "actual": actual_volatility.loc[validation_dates],
                "regime": validation_regimes,
            },
            index=validation_dates,
        ).join(validation_predictions)

        if validation_frame.isna().any().any():
            raise ValueError(
                f"Fold {split.fold}: validation data contains missing values."
            )

        # Find the best overall validation model as the fallback.
        global_model, global_scores = select_by_qlike(
            validation_frame,
            MODEL_NAMES,
        )

        # Choose a model separately for each regime using validation data.
        # Test targets are never used to make these choices.
        selected_by_regime: dict[str, str] = {}

        for regime in REGIMES:
            regime_validation = validation_frame.loc[
                validation_frame["regime"] == regime
            ]

            n_validation = len(regime_validation)
            fallback = n_validation < MIN_VALIDATION_OBS_PER_REGIME

            if fallback:
                selected_model = global_model
                selection_scores = global_scores
                selection_basis = "global_validation_fallback"
            else:
                selected_model, selection_scores = select_by_qlike(
                    regime_validation,
                    MODEL_NAMES,
                )
                selection_basis = "regime_validation_qlike"

            selected_by_regime[regime] = selected_model

            regime_score = (
                selection_scores[selected_model]
                if not fallback
                else global_scores[global_model]
            )

            selection_records.append(
                {
                    "fold": split.fold,
                    "regime": regime,
                    "validation_n": n_validation,
                    "selection_basis": selection_basis,
                    "global_validation_winner": global_model,
                    "selected_model": selected_model,
                    "selection_qlike": regime_score,
                    **{
                        f"validation_qlike_{name}": (
                            calculate_volatility_metrics(
                                regime_validation["actual"],
                                regime_validation[name],
                            )["qlike"]
                            if n_validation > 0
                            else np.nan
                        )
                        for name in MODEL_NAMES
                    },
                }
            )

        # Apply the frozen per-regime choices to the identical test dates.
        test_frame = baseline_predictions.loc[
            test_dates,
            ["actual", *MODEL_NAMES],
        ].copy()

        test_frame["fold"] = split.fold
        test_frame["regime"] = test_regimes

        test_frame["selected_model"] = test_frame["regime"].map(
            selected_by_regime
        )

        test_frame["regime_switching_forecast"] = [
            float(test_frame.loc[date, model_name])
            for date, model_name in zip(
                test_frame.index,
                test_frame["selected_model"],
            )
        ]

        if test_frame[
            ["actual", *MODEL_NAMES, "regime_switching_forecast"]
        ].isna().any().any():
            raise ValueError(
                f"Fold {split.fold}: incomplete test forecasts."
            )

        test_frames.append(test_frame)

        print(
            f"Fold {split.fold:03d}/{len(splits):03d}: "
            f"validation choices "
            f"{selected_by_regime}"
        )

    test_results = pd.concat(test_frames).sort_index()

    if test_results.index.has_duplicates:
        raise ValueError("Test folds produced duplicate dates.")

    if not test_results.index.equals(baseline_predictions.index):
        raise ValueError(
            "Regime-aware predictions do not match baseline test dates."
        )

    selection_table = pd.DataFrame(selection_records)
    candidate_models = [*MODEL_NAMES, "Regime-switching"]

    overall_records = []

    for model_name in candidate_models:
        prediction_column = (
            "regime_switching_forecast"
            if model_name == "Regime-switching"
            else model_name
        )

        overall_records.append(
            {
                "model": model_name,
                "n_test": len(test_results),
                **calculate_volatility_metrics(
                    test_results["actual"],
                    test_results[prediction_column],
                ),
            }
        )

    overall_metrics = pd.DataFrame(overall_records)

    fold_records = []
    regime_records = []

    for fold_number, fold_frame in test_results.groupby("fold", sort=True):
        for model_name in candidate_models:
            prediction_column = (
                "regime_switching_forecast"
                if model_name == "Regime-switching"
                else model_name
            )
            fold_records.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "n_test": len(fold_frame),
                    **calculate_volatility_metrics(
                        fold_frame["actual"],
                        fold_frame[prediction_column],
                    ),
                }
            )

    for regime, regime_frame in test_results.groupby("regime", sort=True):
        for model_name in candidate_models:
            prediction_column = (
                "regime_switching_forecast"
                if model_name == "Regime-switching"
                else model_name
            )
            regime_records.append(
                {
                    "regime": regime,
                    "model": model_name,
                    "n_test": len(regime_frame),
                    **calculate_volatility_metrics(
                        regime_frame["actual"],
                        regime_frame[prediction_column],
                    ),
                }
            )

    fold_metrics = pd.DataFrame(fold_records)
    regime_metrics = pd.DataFrame(regime_records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "predictions": OUTPUT_DIR / "regime_switching_predictions.csv",
        "overall_metrics": OUTPUT_DIR / "regime_switching_metrics.csv",
        "fold_metrics": OUTPUT_DIR / "regime_switching_fold_metrics.csv",
        "regime_metrics": OUTPUT_DIR / "regime_switching_test_metrics_by_regime.csv",
        "selections": OUTPUT_DIR / "regime_switching_model_selections.csv",
    }

    test_results.to_csv(outputs["predictions"], index_label="date")
    overall_metrics.to_csv(outputs["overall_metrics"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    regime_metrics.to_csv(outputs["regime_metrics"], index=False)
    selection_table.to_csv(outputs["selections"], index=False)

    print("\nPOOLED OUT-OF-SAMPLE RESULTS")
    print("===========================")
    print(
        overall_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    print("\nREGIME-SWITCHING PERFORMANCE BY TEST REGIME")
    print("===========================================")
    print(
        regime_metrics[
            regime_metrics["model"].isin(
                ["GARCH-normal", "GJR-GARCH-Student-t", "Regime-switching"]
            )
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )

    print("\nSaved outputs:")
    for description, path in outputs.items():
        print(f"{description}: {path.relative_to(PROJECT_ROOT)}")

    print(
        "\nInterpretation note: model choices are made using each fold's "
        "validation data, not that fold's test targets. These results are "
        "still part of model development, not an untouched final holdout."
    )


if __name__ == "__main__":
    main()
