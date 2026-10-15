# Regivance

A reproducible research project testing whether market-regime information improves five-observation portfolio-volatility forecasts beyond strong statistical models.

## Current status — 2026-10-15

The implementation, full development experiments, robustness analysis and development research report are complete. **121 tests pass** locally; `pip check` reports no broken requirements. GitHub Actions is configured for Windows/Linux Python 3.13, but has not been run remotely as part of this update.

The only mandatory research milestone still pending is the **prospective final evaluation**. Existing data has already informed development, so it cannot honestly be called an untouched holdout. The frozen protocol needs 257 new source observations after 2026-10-15: 252 forecast origins plus five observations to complete their targets. Approximate overall research progress: **95%**, not a measured score or a claim of final validation.

Read the [development research report](reports/research_report.md), [implementation status](docs/implementation-plan.md), and [change summary](docs/completion-summary.md).

## Interactive React dashboard

The project includes a responsive React/Vite presentation layer under [`website/`](website/README.md). It uses React Router, Framer Motion, Recharts and Lucide for page navigation, motion, research visualizations and interface icons. The browser reads curated development summaries and never retrains models on page load.

```powershell
cd website
npm.cmd install
npm.cmd test
npm.cmd run build
npm.cmd run dev
```

Using `npm.cmd` avoids PowerShell script-policy issues that can block `npm.ps1` on Windows.

## Setup — Windows PowerShell

Run from the repository root. Python 3.13 (64-bit) is the tested version.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B -m pytest tests -v -p no:cacheprovider
```

Activation is optional. Using the explicit virtual-environment executable avoids installation into the wrong Python. If you prefer activation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip check
```

If activation is blocked, use the explicit executable above, or enable scripts for this terminal only with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. Do not change machine-wide policy.

For the exact installed Windows/Python 3.13.14 environment used in this reproduction, use `requirements-lock-windows-py313.txt` in place of `requirements.txt`. It is an environment snapshot, not a cross-platform, hash-locked package archive. Other versions/platforms may produce slightly different floating-point results.

Troubleshooting:

- Confirm the interpreter with `python -c "import sys; print(sys.executable)"`.
- For missing wheels, check 64-bit Python 3.13 and update pip.
- For temporary-directory permission errors, use a writable temporary directory or run in a normal local terminal. Do not interpret fixture setup failures as model failures.
- Linux/macOS: use `python3 -m venv .venv` and `.venv/bin/python` instead of the Windows executable.

## Dataset

Download the [daily 30 Industry Portfolios CSV ZIP](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_Daily_CSV.zip) from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Extract the CSV to:

```text
data/raw/30_Industry_Portfolios_Daily.csv
```

The reproduced local vintage spans 1926-07-01 through 2026-08-31 (26,317 observations). See [source provenance](data/raw/README.md) and the [data contract](docs/data-contract.md). Raw data is not committed. A newer download may revise history and will not necessarily reproduce this vintage; archive the original file separately.

The loader selects the **value-weighted industry table**, converts percentages to decimals, and preserves valid historical Saturday observations. The portfolio then weights the 30 industries equally each day.

At the close of observation `t`:

```text
portfolio_return[t] = mean(industry_returns[t])
target[t] = sqrt(sum(portfolio_return[t+j] ** 2 for j = 1,...,5))
```

This is five-observation realized volatility, not annualized volatility or centered sample standard deviation. The last five observations lack complete targets.

## Reproduce the complete development workflow

```powershell
.\.venv\Scripts\python.exe -B scripts/validate_data.py
.\.venv\Scripts\python.exe -B scripts/run_research.py
```

This builds the dataset, evaluates statistical and regime-switching models, trains all four LSTM input variants plus the QLIKE-objective variant, and generates diagnostics, uncertainty estimates, robustness tables, plots and the report. Model training runs on CPU.

Configuration: [configs/research.yaml](configs/research.yaml). The old `configs/experiment.yaml` is an unused scaffold. The diagnostic suite intentionally requires 120 blocks of 21 observations; some schema-encoded choices (five-observation horizon, 21-observation risk features, three regimes) are validated rather than freely tunable.

Run records: `reports/runs/development/manifest.json`, per-stage logs and `environment.txt`. Records include code/config, source-data and artifact checksums. A new full run refreshes generated outputs and archives previous logs/manifest and controlled QLIKE artifacts under `reports/runs/development/archive/`. Raw source data is not changed.

To resume an interrupted run with unchanged inputs:

```powershell
.\.venv\Scripts\python.exe -B scripts/run_research.py --resume
```

Resume refuses changed code/config/data/environment and checks saved output hashes. Most generated prediction files are ignored by Git. Compact research summary tables and the final comparison plot are retained for review.

## Models, features and evaluation

The 11 compared forecasts are:

- Rolling volatility (21 observations), EWMA (decay 0.94), normal GARCH, Student-t GARCH and Student-t GJR-GARCH.
- Regime-switching selection among those statistical models, based only on validation QLIKE.
- LSTM; regime-aware LSTM; risk-feature LSTM; risk-feature-plus-regime LSTM.
- Risk-feature-plus-regime LSTM trained with QLIKE instead of log-volatility MSE.

Regime inputs are trailing portfolio RMS volatility, mean pairwise cross-industry correlation, and same-day cross-sectional dispersion. Training-only standardization and K-means discover three clusters, ranked lower/middle/higher by training-center volatility. These ranks are relative to each fold, not permanent economic labels.

Each fold uses 1,260 training origins, 252 validation origins, five-observation gaps at both boundaries, and 21 test origins. The development period contains 120 folds and **2,520 forecasts per model on identical dates**. Preprocessing/clustering fit only on training data; validation controls early stopping and regime-specific selection. Later folds may learn from earlier test observations after they become historical.

Metrics: MAE, RMSE and variance-based QLIKE (lower is better). Dependence-aware comparisons use HAC lags 4/10/20, Holm corrections and moving-block bootstrap intervals. Robustness tables cover years, regimes, nonoverlapping-target offsets and a 2020 shock sensitivity check.

## Reproduced findings

GJR-GARCH with Student-t errors has the lowest pooled MAE, RMSE and QLIKE in the development comparison:

| MAE | RMSE | QLIKE | Forecasts |
| ---: | ---: | ---: | ---: |
| 0.007468 | 0.011129 | 0.419191 | 2,520 |

QLIKE for regime-switching is 0.437712; the risk-plus-regime LSTM scores 0.697074, and its QLIKE-trained variant scores 0.641875. Regimes improve some neural comparisons but do not establish a general advantage over statistical models. Pooled rankings alone do not establish statistical significance.

See [all model scores](reports/tables/research_all_model_metrics.csv), [robustness results](reports/tables/research_robustness_metrics.csv), [paired QLIKE comparisons](reports/tables/research_qlike_comparisons.csv), and [incremental regime tests](reports/dependence_robust_comparison.md).

These are reproduced development results, not investment advice or confirmatory final-test results. We did not conduct exhaustive tuning or repeated-seed experiments.

## Prospective final evaluation

[configs/frozen_protocol.json](configs/frozen_protocol.json) records the registered specification, normalized code fingerprints, source-history fingerprint and environment. Do not edit/delete it to bypass a failed check.

Check readiness:

```powershell
.\.venv\Scripts\python.exe -B scripts/final_evaluation.py
```

When the full future period exists, keep the archived source vintage intact and pass a separate updated file:

```powershell
.\.venv\Scripts\python.exe -B scripts/final_evaluation.py --run --raw data/raw/30_Industry_Portfolios_Daily_updated.csv
```

The evaluator selects the **first** 252 post-freeze origins, not a moving favorable window. It checks history, code, configuration and environment; uses the same 11 models; and saves final predictions, metrics, regime summaries, validation choices and paired QLIKE tests under `reports/final_evaluation/`. Existing final outputs are not overwritten. Changed historical source values require preserving the frozen vintage or explicitly documenting a protocol amendment, not silently continuing.

The registration command `python scripts/final_evaluation.py --freeze` is a one-time initialization and refuses an existing registration. The supplied registration is already initialized. A dated repository record is useful provenance, but is not an external preregistration service.

## Repository map

```text
configs/       Active research config and frozen prospective protocol
data/raw/      Archived source CSV and provenance
data/processed/ Generated portfolio targets and statistical predictions
src/data/      Dataset loader and portfolio/target construction
src/features/  Trailing features and training-only regime discovery
src/models/    Rolling, EWMA, GARCH, LSTM and sequence construction
src/evaluation/ Splits, metrics, regime analysis, prospective boundaries
scripts/       Reproduction runner, comparisons, report and final evaluation
tests/         Unit and synthetic end-to-end integration tests
reports/       Development report, diagnostics, plots and compact tables
docs/          Data contract, roadmap and implementation change summary
.github/       Automated test workflow
```

No future market outcomes have been simulated or fabricated for the final research claim.
