# MarketRisk Lab

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

# Verify dependencies, run the scaffold, and execute tests
python -c "import arch, numpy, pandas, sklearn, torch; print('Dependencies verified')"
python -m src.pipeline --config configs/experiment.yaml
python -m pytest
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

The pipeline currently validates the project configuration and creates the expected output directories. The next milestone is the data-ingestion slice described in `docs/implementation-plan.md`.

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

Project scaffold complete. No model results should be interpreted until the data contract, loader, and target construction have been implemented and tested.
