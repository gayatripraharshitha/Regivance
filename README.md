# MarketRisk Lab

A reproducible research system for forecasting five-trading-day volatility of a fixed-weight US industry portfolio. It compares classical volatility benchmarks, market-regime features, and a small LSTM under chronological, leakage-controlled evaluation.

## Research question

Does market-regime information improve five-day realized-volatility forecasts, and does a small LSTM outperform conventional volatility models out of sample?

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline --config configs/experiment.yaml
pytest
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
