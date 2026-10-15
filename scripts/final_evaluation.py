"""Freeze, check, and execute the prospective evaluation without changing the protocol."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research_config import load_research_config
from src.data.loader import load_industry_returns
from src.data.portfolio import build_forecasting_dataset
from src.features.rolling_features import build_regime_features
from src.evaluation.final_protocol import final_evaluation_plan, history_digest
from src.evaluation.metrics import calculate_volatility_metrics, calculate_volatility_loss_contributions
from src.models.rolling_volatility import forecast_rolling_volatility
from src.models.ewma import forecast_ewma_volatility
from src.models.garch import forecast_garch_volatility_for_fold
from scripts.compare_lstm_models import build_regime_one_hot, make_model_config, evaluate_model
from scripts.compare_lstm_risk_features import scale_returns, scale_risk_features, build_model_inputs
from scripts.compare_lstm_qlike_objective import evaluate_qlike_model
from scripts.analyze_dependence_robust_comparison import hac_mean_difference, holm_adjust, moving_block_bootstrap_ci

FREEZE = ROOT / "configs/frozen_protocol_v2.json"
GARCH = [("GARCH-normal", "normal", False), ("GARCH-Student-t", "t", False), ("GJR-GARCH-Student-t", "t", True)]


def code_hashes():
    """Fingerprint prospective dependencies, excluding unrelated analyses/tests."""
    script_names = (
        "final_evaluation.py", "compare_statistical_models.py", "compare_lstm_models.py",
        "compare_lstm_risk_features.py", "compare_lstm_qlike_objective.py",
        "analyze_dependence_robust_comparison.py",
    )
    paths = [p for p in (ROOT / "src").rglob("*.py")]
    paths.extend(ROOT / "scripts" / name for name in script_names)
    # Text normalization survives Git's Windows CRLF checkout conversion.
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in paths}


def evaluate(returns, config, plan):
    """All specifications use identical prospective origins and fold boundaries."""
    torch.set_num_threads(1)
    returns = returns.iloc[:plan["raw_stop_position"]]
    ev, model = config["evaluation"], config["models"]
    dataset = build_forecasting_dataset(returns, horizon=ev["horizon"])
    target = dataset["target_volatility_5d"]
    portfolio = dataset["portfolio_return"]
    risk = build_regime_features(returns).reindex(dataset.index)
    industry = returns.reindex(dataset.index)
    fixed = pd.DataFrame({
        "Rolling": forecast_rolling_volatility(portfolio, window=21, horizon=ev["horizon"]),
        "EWMA": forecast_ewma_volatility(portfolio, decay=0.94, horizon=ev["horizon"], min_periods=21),
    })
    frames, choices, training_log = [], [], []
    for split in plan["splits"]:
        train_dates, val_dates, dates = (dataset.index[s] for s in (split.train, split.validation, split.test))
        one_hot, detector = build_regime_one_hot(risk, train_dates)
        labels = one_hot.dropna().idxmax(axis=1).str.replace("regime_", "", regex=False)
        validation = fixed.loc[val_dates].copy()
        frame = fixed.loc[dates].copy()
        for name, distribution, asymmetric in GARCH:
            predictions = forecast_garch_volatility_for_fold(
                portfolio.loc[train_dates], portfolio, val_dates.append(dates),
                horizon=ev["horizon"], distribution=distribution, asymmetric=asymmetric,
            )
            validation[name] = predictions.loc[val_dates]
            frame[name] = predictions.loc[dates]
        candidates = frame.columns.tolist()
        def best(rows):
            return min(candidates, key=lambda c: calculate_volatility_metrics(target.loc[rows], validation.loc[rows, c])["qlike"])
        fallback = best(val_dates)
        selected = {}
        for regime in ("lower", "middle", "higher"):
            subset = val_dates[labels.loc[val_dates] == regime]
            selected[regime] = best(subset) if len(subset) >= model["min_regime_validation"] else fallback
            choices.append({"fold": split.fold, "regime": regime, "validation_n": len(subset), "selected": selected[regime]})
        frame["Regime-switching"] = [frame.loc[d, selected[labels.loc[d]]] for d in dates]
        inputs = build_model_inputs(scale_returns(industry, train_dates), scale_risk_features(risk, train_dates), one_hot)
        config_lstm = make_model_config(split.fold, model["epochs"], model["hidden_size"], model["batch_size"])
        for name, features in inputs.items():
            prediction, _, prediction_dates = evaluate_model(features, target, split, config_lstm)
            if not prediction_dates.equals(dates):
                raise ValueError("LSTM date mismatch.")
            frame[name] = prediction
        prediction, _, prediction_dates, fitted = evaluate_qlike_model(
            inputs["Risk-feature + regime LSTM"], target, split, replace(config_lstm, objective="qlike"))
        if not prediction_dates.equals(dates):
            raise ValueError("QLIKE date mismatch.")
        frame["QLIKE-objective Risk-feature + regime LSTM"] = prediction
        training_log.append({"fold": split.fold, "qlike_best_epoch": fitted.best_epoch_, "qlike_validation_loss": fitted.best_validation_loss_})
        frame["actual"], frame["regime"], frame["fold"] = target.loc[dates], labels.loc[dates], split.fold
        frames.append(frame)
        print(f"Final fold {split.fold}/{len(plan['splits'])} complete", flush=True)
    return pd.concat(frames), pd.DataFrame(choices), pd.DataFrame(training_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true", help="Only runs once the full prospective period is available.")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--protocol", type=Path, default=FREEZE)
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    cfg = load_research_config()
    raw = args.raw or ROOT / cfg["data"]["raw_path"]
    returns = load_industry_returns(raw)
    environment = {"python": sys.version, "packages": subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True)}
    if args.freeze:
        if args.run or protocol.exists():
            raise ValueError("Freeze is immutable; do not replace an existing registration.")
        if returns.index[-1] > pd.Timestamp(cfg["final_evaluation"]["exposed_data_through"]):
            raise ValueError("Exposure boundary must cover all currently inspected data.")
        record = {"registered_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                  "environment": environment,
                  "code_sha256": code_hashes(), "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                  "historical_returns_sha256": history_digest(returns, cfg["final_evaluation"]["exposed_data_through"])}
        protocol.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Frozen protocol: {protocol}")
        return
    if not protocol.exists():
        raise FileNotFoundError("Register the protocol with --freeze before evaluating.")
    frozen = json.loads(protocol.read_text())
    if frozen["config"] != cfg or frozen["code_sha256"] != code_hashes():
        raise ValueError("Code/config differs from the frozen protocol; document an amendment before proceeding.")
    if frozen["environment"] != environment:
        raise ValueError("Environment differs from the frozen protocol; restore it or document an amendment.")
    if history_digest(returns, cfg["final_evaluation"]["exposed_data_through"]) != frozen["historical_returns_sha256"]:
        raise ValueError("Historical source returns changed. Preserve the frozen vintage or document an amendment.")
    plan = final_evaluation_plan(returns.index, cfg)
    print(json.dumps({k: v for k, v in plan.items() if k != "splits"}, indent=2))
    if plan["status"] != "ready" or not args.run:
        return
    output = ROOT / "reports/final_evaluation"
    output.mkdir(exist_ok=False)
    try:
        predictions, choices, training = evaluate(returns, cfg, plan)
        predictions.to_csv(output / "predictions.csv", index_label="date")
        choices.to_csv(output / "validation_choices.csv", index=False)
        training.to_csv(output / "training.csv", index=False)
        models = predictions.columns.drop(["actual", "regime", "fold"])
        metrics, regimes, tests = [], [], []
        actual = predictions["actual"]
        reference = cfg["final_evaluation"]["reference"]
        reference_losses = calculate_volatility_loss_contributions(actual, predictions[reference])
        for model in models:
            metrics.append({"model": model, "n_test": len(predictions), **calculate_volatility_metrics(actual, predictions[model])})
            for regime, rows in predictions.groupby("regime"):
                regimes.append({"model": model, "regime": regime, "n_test": len(rows), **calculate_volatility_metrics(rows.actual, rows[model])})
            if model != reference:
                difference = reference_losses["qlike"] - calculate_volatility_loss_contributions(actual, predictions[model])["qlike"]
                for lag in (4, 10, 20):
                    tests.append({"model": model, "lag": lag, **hac_mean_difference(difference, lag),
                                  **moving_block_bootstrap_ci(difference, block_length=21)})
        tests = pd.DataFrame(tests)
        tests["holm_p"] = tests.groupby("lag")["p_value_two_sided"].transform(holm_adjust)
        pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(regimes).to_csv(output / "metrics_by_regime.csv", index=False)
        tests.to_csv(output / "primary_qlike_comparisons.csv", index=False)
        (output / "manifest.json").write_text(json.dumps({"status": "complete", "protocol": frozen,
            "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "n_test": len(predictions)}, indent=2))
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"status": "failed", "error": repr(error)}))
        raise


if __name__ == "__main__":
    main()
