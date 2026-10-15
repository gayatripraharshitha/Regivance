# Completion summary — 2026-10-15

## Added

- `configs/research.yaml` and `src/research_config.py`: active, validated experiment configuration.
- `scripts/run_research.py`: complete development workflow, stage logs, environment/source/output fingerprints, safe resume, and preservation of previous controlled outputs.
- `src/evaluation/final_protocol.py`, `scripts/final_evaluation.py` and `configs/frozen_protocol.json`: prospective date gates, immutable registration, revision checks, and an executable 11-model final evaluator.
- `scripts/build_research_report.py`: combined 11-model report, paired uncertainty comparisons, yearly/regime/shock/nonoverlapping-target checks.
- `tests/test_final_protocol.py`, `tests/test_research_report.py`, `tests/test_research_runner.py`: boundary, tamper, archival, consistency and real synthetic integration tests.
- `requirements-lock-windows-py313.txt`: exact installed environment snapshot.
- `.github/workflows/tests.yml`: Windows/Linux Python 3.13 test workflow.
- `reports/research_report.md`, compact `research_*.csv/json` summary artifacts and `reports/figures/all_models_qlike.png`.

## Updated

- Dataset, statistical/LSTM/regime and diagnostic scripts now read active configuration where appropriate. Schema-specific choices are explicitly constrained.
- QLIKE experiment distinguishes a full development comparison from a subset smoke test.
- `requirements.txt` explicitly declares statsmodels.
- `.gitignore` keeps large generated predictions/logs local while retaining compact research evidence.
- `README.md`, data contract, roadmap and raw-data provenance reflect the actual implementation and results.
- Existing diagnostic reports/plots were refreshed from regenerated forecasts.
- One test's datetime shift was modernized to remove a dependency deprecation warning.

## What was executed

All 11 model variants were evaluated on the same 2,520 development origins across 120 folds, including a complete QLIKE-objective LSTM experiment. Statistical/regime comparisons, neural comparisons, diagnostic checks, forecast-failure attribution, dependence-robust inference and report synthesis were run. The complete local test suite passes: 121 tests, no warnings. Dependency consistency is clean.

The new prospective evaluator was exercised with real model fitting on synthetic data; synthetic results are not presented as real market evidence. Its real-data invocation correctly waits rather than fitting on exposed history.

## Findings and honest boundary

Student-t GJR-GARCH remains the best pooled development benchmark (MAE 0.007468, RMSE 0.011129, QLIKE 0.419191). Regimes help some neural comparisons but do not demonstrate general superiority.

Today's implementation and development deliverables are complete. A final untouched market evaluation still requires 257 post-freeze source observations. No commit, push, remote CI success, or final market result is implied by these local changes.
