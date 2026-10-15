"""Run all development comparisons with recorded inputs, logs and artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research_config import load_research_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_qlike_outputs(directory):
    """Preserve controlled-experiment artifacts before an explicit new run."""
    for suffix in ("predictions", "metrics", "fold_metrics", "training", "failures"):
        path = ROOT / "reports/tables" / f"lstm_qlike_objective_{suffix}.csv"
        if path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(directory / path.name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/research.yaml"))
    parser.add_argument("--resume", action="store_true", help="Reuse successful stages only when code, data, config and environment match.")
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_research_config(cfg_path)
    if cfg["evaluation"]["development_size"] != 2520 or cfg["evaluation"]["test_size"] != 21:
        raise ValueError("The development diagnostic suite requires 120 blocks of 21 observations.")
    raw = ROOT / cfg["data"]["raw_path"]
    if not raw.is_file():
        raise FileNotFoundError(raw)
    directory = ROOT / "reports/runs/development"
    directory.mkdir(parents=True, exist_ok=True)
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    sources = {str(p.relative_to(ROOT)): digest(p) for folder in ("src", "scripts", "configs")
               for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".yaml")}
    identity = {"raw_sha256": digest(raw), "config": cfg, "sources": sources,
                "packages": packages, "python": sys.version}
    manifest_path = directory / "manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.resume and prior and prior.get("identity") != identity:
        raise ValueError("Cannot resume: inputs/code/environment changed. Start a new full run.")
    archive = directory / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    if prior and not args.resume:
        archive.mkdir(parents=True)
        for path in directory.iterdir():
            if path.is_file():
                shutil.copy2(path, archive / path.name)
    manifest = prior if args.resume and prior else {
        "identity": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "stages": {}, "status": "running",
    }
    (directory / "environment.txt").write_text(packages, encoding="utf-8")
    env = dict(os.environ, REGIVANCE_CONFIG=str(cfg_path), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    stages = [
        ("build_dataset", []), ("compare_statistical_models", []),
        ("analyze_regime_performance", []), ("evaluate_regime_switching_model", []),
        ("compare_lstm_risk_features", []),
        ("compare_lstm_qlike_objective", ["--max-folds", str(cfg["evaluation"]["development_size"] // cfg["evaluation"]["test_size"])]),
        ("generate_research_diagnostics", []), ("analyze_forecast_failures", []),
        ("analyze_dependence_robust_comparison", []),
        ("analyze_statistical_model_significance", []),
        ("build_research_report", []),
    ]
    for name, options in stages:
        previous = manifest["stages"].get(name, {})
        if args.resume and previous.get("returncode") == 0:
            artifacts = previous.get("artifacts", {})
            if artifacts and all((ROOT / p).is_file() and digest(ROOT / p) == h for p, h in artifacts.items()):
                print(f"Reusing {name}", flush=True)
                continue
        print(f"Running {name} (log: reports/runs/development/{name}.log)", flush=True)
        if name == "compare_lstm_qlike_objective":
            archive_qlike_outputs(archive / "qlike")
        before = {p: p.stat().st_mtime_ns for folder in ("data/processed", "reports/tables", "reports/figures", "reports")
                  for p in (ROOT / folder).glob("*") if p.is_file()}
        command = [sys.executable, "-B", str(ROOT / "scripts" / f"{name}.py"), *options]
        with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        artifacts = {str(p.relative_to(ROOT)): digest(p)
                     for folder in ("data/processed", "reports/tables", "reports/figures", "reports")
                     for p in (ROOT / folder).glob("*")
                     if p.is_file() and p.stat().st_mtime_ns != before.get(p)}
        manifest["stages"][name] = {"command": command, "returncode": result.returncode, "artifacts": artifacts}
        manifest["status"] = "failed" if result.returncode else "running"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect its log. No partial run is marked complete.")
    current_sources = {str(p.relative_to(ROOT)): digest(p) for folder in ("src", "scripts", "configs")
                       for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".yaml")}
    if current_sources != sources or digest(raw) != identity["raw_sha256"]:
        manifest["status"] = "invalidated"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise RuntimeError("Code/config/data changed during the run; results are not a coherent reproduction.")
    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Development reproduction complete.", flush=True)


if __name__ == "__main__":
    main()
