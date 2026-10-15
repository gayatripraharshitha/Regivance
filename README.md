# Regivance

A reproducible research system for forecasting five-trading-day volatility of a fixed-weight US industry portfolio. It compares classical volatility benchmarks, market-regime features, and a small LSTM under chronological, leakage-controlled evaluation.

## Research question

Does market-regime information improve five-day realized-volatility forecasts, and does a small LSTM outperform conventional volatility models out of sample?

## Windows / PowerShell setup

Use Python 3.10–3.13 (64-bit). Run each command from the repository root in **PowerShell**. Invoking Python through the virtual environment avoids accidentally installing packages into a different Python installation.

```powershell
# Create the isolated environment (only needed once)
py -3.13 -m venv .venv

# Activate it for the current PowerShell session
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install all project dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Verify dependencies, build the dataset, and execute tests
python -c "import arch, numpy, pandas, sklearn, torch; print('Dependencies verified')"
python scripts/validate_data.py
python scripts/build_dataset.py
python -m pytest tests -v -p no:cacheprovider
```

If `py -3.13` is unavailable, replace it with `python`; use `py --list` to see installed Python versions. If PowerShell blocks activation, run the following **once for the current terminal only**, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

You may also skip activation entirely by replacing `python` above with `.\.venv\Scripts\python.exe`.

### Installation troubleshooting

- `No matching distribution found`: update pip with `python -m pip install --upgrade pip`, then retry. Confirm the command is using the virtual environment: `python -c "import sys; print(sys.executable)"`.
- `Access is denied` or temp-directory errors: close editors/terminals using the environment, then retry with a local temp directory:

  ```powershell
  New-Item -ItemType Directory -Force .pip-tmp | Out-Null
  $env:TMP = (Resolve-Path .pip-tmp)
  $env:TEMP = $env:TMP
  python -m pip install --no-cache-dir -r requirements.txt
  ```

The dataset builder writes `data/processed/portfolio_forecasting_dataset.csv`. An older `portfolio_returns.parquet` may exist from the scaffold but is not used by the current comparison scripts. There is no current `src/pipeline.py` entry point. Scripts use constants defined inside them; `configs/experiment.yaml` contains planned settings and is not yet connected to this workflow.

## Layout

```text
configs/       Experiment parameters
data/raw/      Downloaded source data (not committed)
data/processed/ Cleaned, feature-ready datasets (not committed)
src/           Research pipeline package
tests/         Unit and integration tests
reports/       Generated charts, tables, and write-ups
scripts/       Repeatable operational commands
docs/          Data contract and implementation plan
```

## Reproducibility and leakage policy

All transformations that learn from data (scalers, PCA, clustering, neural-network fitting) must be fit on each training window only. Forecast targets use the following five trading days; train/test splits therefore require an embargo of at least five observations.

## Data source

The intended input is the Kenneth R. French Data Library's daily 30 Industry Portfolios dataset. Record the downloaded file name, retrieval date, source URL, and checksum in `data/raw/README.md` before a run.

## Status

Current overall progress is approximately **75%**, a planning estimate against the complete research goal. The data pipeline, statistical benchmarks, walk-forward evaluation, regime analysis, regime-switching experiment, LSTM variants, diagnostics and dependence-robust comparisons are implemented. An untouched final evaluation and final report synthesis remain.

Workspace verification on 2026-10-15: **95 tests passed** with one dependency deprecation warning.

New workflow:

```powershell
python scripts/build_dataset.py
python scripts/compare_statistical_models.py
python scripts/analyze_regime_performance.py
```

`src/evaluation/regime_analysis.py` fits a fresh scaler and K-means model on each fold's training features. It ranks clusters by training-center volatility as lower/middle/higher, preserving raw cluster IDs and centers for inspection. These ranks are relative to each training window, not fixed economic states. Features are calculated through the forecast date, consistent with after-close forecasts. Warm-up rows are removed only within training slices, so evaluation dates never shift.

`scripts/analyze_regime_performance.py` requires existing statistical predictions and verifies their dates and actual targets against the current dataset. It writes `regime_forecasts.csv`, `regime_centers_by_fold.csv`, and `regime_model_metrics.csv` under `reports/tables/`. Metrics describe existing models conditional on the assigned regime; they do not demonstrate that regime inputs improve forecasts. The prediction CSV must be generated locally before running this analysis.

The current results remain exploratory because the evaluation period informed model development. Next: freeze the selected specifications, establish an untouched final evaluation period, and complete the research report.

| Component | Status | Files |
| --- | --- | --- |
| Dataset parsing | Implemented | `src/data/loader.py` |
| Data checks, portfolio and forward target | Implemented | `src/data/portfolio.py`, `scripts/validate_data.py` |
| Rolling volatility and EWMA | Implemented | `src/models/rolling_volatility.py`, `src/models/ewma.py` |
| Normal GARCH, Student-t GARCH, Student-t GJR-GARCH | Implemented | `src/models/garch.py` |
| Walk-forward evaluation | Implemented | `src/evaluation/splits.py` |
| MAE, RMSE, variance-based QLIKE | Implemented | `src/evaluation/metrics.py` |
| Pooled and fold-level comparisons | Implemented | `scripts/compare_statistical_models.py` |
| Exploratory HAC loss comparisons | Implemented | `scripts/analyze_statistical_model_significance.py` |
| Market-regime features and clustering | Implemented | `src/features/rolling_features.py`, `src/features/regimes.py` |
| Fold-by-fold regime analysis | Implemented | `src/evaluation/regime_analysis.py`, `scripts/analyze_regime_performance.py` |
| Regime-switching forecasting experiment | Implemented | `scripts/evaluate_regime_switching_model.py` |
| LSTM and regime-aware LSTM experiments | Implemented | `src/models/lstm.py`, `src/models/lstm_data.py`, comparison scripts |
| Research diagnostics and failure attribution | Implemented | `scripts/generate_research_diagnostics.py`, `scripts/analyze_forecast_failures.py` |
| Dependence-robust comparisons | Implemented | `scripts/analyze_dependence_robust_comparison.py` |
| Untouched final holdout and final report | Pending | `reports/` and `docs/` |

Earlier planning documents may still describe the original scaffold; this README records the current implementation and sequence.

## Target and forecast timing

Industry inputs come from the source's value-weighted table. We then give each industry equal weight each day, implying daily rebalancing across industries. At the close of observation `t`:

```text
portfolio_return[t] = mean of the 30 industry returns at t
target_volatility_5d[t] = sqrt(sum(r[t+j]^2 for j = 1,...,5))
rolling_forecast[t] = sqrt(5 * mean(r^2 over the latest 21 observations))
```

The target is five-observation realized volatility, not annualized volatility or the root mean square of daily returns. The final five rows have no complete target and are dropped. Preserve valid historical observations, including historical Saturday observations; windows and gaps count source observations, not calendar days.

## Statistical comparison workflow

After building the dataset, run:

```powershell
python scripts/compare_statistical_models.py
python scripts/analyze_statistical_model_significance.py
```

The comparison fits each GARCH variant per fold and writes these files to `data/processed/`:

- `walk_forward_statistical_models_predictions.csv`
- `statistical_model_comparison_metrics.csv`
- `statistical_model_comparison_fold_metrics.csv`

The significance script writes `reports/tables/gjr_garch_hac_significance.csv`. Generated datasets and reports are ignored by Git and must be regenerated locally. Individual-stage scripts also include `run_rolling_baseline.py`, `evaluate_rolling_baseline.py`, and `compare_baselines.py`.

The full comparison specifies 1,260 training observations, 252 validation observations, five-observation gaps at both boundaries, and 21-observation test blocks advancing by 21 observations. A 2,520-observation evaluation period yields **120 folds and 2,520 forecasts per model** on identical dates.

GARCH parameters are fitted on each training window. Subsequently observed returns update the variance state while parameters remain fixed within the test block. Validation windows exist, but this comparison currently uses fixed specifications rather than a tuning procedure.

## Reported results and limitations

The reported best pooled benchmark is **GJR-GARCH with Student-t errors**:

| Metric | Reported value |
| --- | ---: |
| MAE | 0.007468 |
| RMSE | 0.011129 |
| QLIKE | 0.419191 |
| Out-of-sample observations | 2,520 |

These scores were supplied from a previous experiment. Its prediction and metric CSVs were absent from the inspected workspace on 2026-10-15, so the scores were not independently reproduced during this documentation update. The 49-test suite was rerun successfully.

The HAC code compares paired losses against normal GARCH and EWMA at lag choices 4 and 20; squared error is the observation-level loss underlying RMSE. These comparisons are exploratory. The reported evidence does not establish superiority across every metric or against every competing model.

The evaluation period has already informed model comparisons. Despite the parameter name `final_test_size`, it should not be described as an untouched final holdout. Establish a separate final evaluation protocol before further model selection.

## Remaining sequence

1. **Freeze the research specification.** Select the statistical and LSTM variants using the completed development evidence, record every fixed parameter, and stop tuning on the current evaluation period.
2. **Create an untouched final evaluation.** Choose a later period that has not informed model development, rerun all selected models on identical dates, and preserve all preprocessing and clustering boundaries.
3. **Complete robustness analysis.** Confirm conclusions across regimes, dependence assumptions and major shock periods without changing the frozen models.
4. **Finish the research presentation.** Consolidate configuration, record dependency versions and experiment provenance, assemble the final plots and tables, and write the methodology, results, limitations and conclusion sections.

Success means answering: **Does market-regime information improve five-day portfolio-volatility forecasts beyond strong statistical alternatives on unseen future data?** Improvement, limited improvement and no improvement are all valid outcomes when supported by rigorous evaluation.
