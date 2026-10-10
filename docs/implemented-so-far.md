# What has been implemented so far

This document records the scaffold and data-pipeline work already added to the repository. It is a guide for understanding the code, not a claim that the research models are complete.

## 1. Project setup

### `README.md`

- Explains the research question: forecasting five-day volatility for an equal-weight portfolio of 30 US industry portfolios.
- Documents a Windows PowerShell setup flow using a virtual environment.
- Explains how to install dependencies, run the pipeline, and run tests.
- Includes common fixes for PowerShell execution policy and temporary-directory permission problems.

### `requirements.txt`

The project dependencies are:

- `numpy`, `pandas`, `scipy`: numerical and tabular calculation.
- `pyarrow`: writes the processed dataset in Parquet format.
- `scikit-learn`: intended for scaling, PCA, and K-means later.
- `arch`: intended for the GARCH model later.
- `torch`: intended for the LSTM later.
- `PyYAML`: reads the experiment configuration.
- `matplotlib`: intended for plots later.
- `pytest`: runs automated checks.

### `.gitignore`

- Excludes the virtual environment, downloaded data, processed data, model files, and temporary/test-cache folders.
- This means raw data is available locally but is not uploaded to GitHub.

### `configs/experiment.yaml`

Contains the experiment settings in one place instead of scattering numbers throughout the code. It defines:

- The five-day forecast horizon.
- Three regime clusters for the future regime-detection stage.
- Planned training/validation/test periods and a five-day embargo.
- Planned parameters for rolling volatility, EWMA, GARCH, and LSTM.

The embargo is important: it leaves a gap between chronological training and testing so the five-day targets cannot overlap and leak future information.

## 2. Configuration validation

### `src/config.py`

Two functions were added:

- `load_config(path)`: reads the YAML configuration file.
- `validate_config(config)`: checks that every required top-level section exists, that the forecast horizon is positive, and that the embargo is at least as long as the forecast horizon.

This catches an invalid experimental setup before any data or model work starts.

### `tests/test_config.py` and `pytest.ini`

- Tests confirm that the real configuration has a safe embargo and that an unsafe one is rejected.
- `pytest.ini` tells pytest to only collect tests inside `tests/`, preventing accidental collection of temporary folders.

## 3. Raw-data ingestion

### Dataset used

The local raw data file is the Kenneth French Data Library's **30 Industry Portfolios [Daily]** CSV. It contains daily percentage returns for 30 US industry portfolios.

The source file is unusual: it starts with explanatory text and contains both a value-weighted return table and an equal-weighted return table. The project intentionally uses the **value-weighted industry table** as the input. We later create our own equal-weight portfolio across those 30 industry return series.

### `src/data/loader.py`

This is the main completed component. It contains four functions.

#### `_value_weighted_csv_block(lines)`

Finds the section named `Average Value Weighted Returns -- Daily`, identifies its CSV header, and extracts only that block. This prevents the parser from mistakenly reading the explanatory text or the later equal-weighted table.

#### `load_industry_returns(path)`

Loads the extracted value-weighted table and standardizes it:

1. Reads the source file.
2. Extracts its value-weighted daily table.
3. Converts date text such as `19260701` into actual dates.
4. Converts source percentages into decimal returns: `1.00` becomes `0.01`.
5. Treats the source missing-value markers `-99.99` and `-999` as missing values.
6. Runs the validation rules before returning the table.

The output is a pandas DataFrame: dates are rows, the 30 industries are columns, and every value is a decimal daily return.

#### `equal_weight_portfolio_return(industry_returns)`

Takes the 30 industry returns for each date and calculates their simple average. This creates one portfolio return per trading day, giving every industry a weight of `1/30`.

#### `forward_realized_volatility(returns, horizon_days)`

Builds the prediction target. At date `t`, it calculates:

`sqrt(mean(r[t+1]^2, r[t+2]^2, ..., r[t+h]^2))`

For this project, `h = 5`. The calculation begins at `t+1`, not `t`, so the target genuinely represents the next five trading days. The final five dates have no target because their future returns do not exist yet.

#### `build_modeling_frame(industry_returns, horizon_days)`

Combines the 30 input columns, the `portfolio_return` column, and the `forward_realized_volatility_5d` target column into one modeling table.

### `src/data/validation.py`

`validate_returns_frame(returns)` rejects input data if it has:

- A non-date index.
- Repeated or out-of-order dates.
- No rows or no industry columns.
- Non-numeric values.
- Missing values.
- Infinite values.

It is deliberately strict. Silent filling of bad financial data is dangerous because it can make later model results appear better than they really are.

## 4. Pipeline

### `src/pipeline.py`

The command below runs the current data pipeline:

```powershell
python -m src.pipeline --config configs/experiment.yaml
```

It performs these steps:

1. Loads and validates the configuration.
2. Creates the report-output folders if needed.
3. Checks whether the configured raw CSV exists.
4. Loads and validates raw industry returns.
5. Constructs the equal-weight portfolio and five-day target.
6. Saves the result to `data/processed/portfolio_returns.parquet`.

With the downloaded source available, the resulting table has 26,317 trading-day rows and 32 columns: 30 industry returns, one portfolio return, and one five-day volatility target.

## 5. Tests added

### `tests/test_loader.py`

Uses a tiny synthetic version of the French file stored in `tests/fixtures/` to check that:

- The loader selects the value-weighted section, not the equal-weighted one.
- Percentage inputs are converted to decimal returns.
- The equal-weight portfolio return is calculated correctly.
- The volatility target uses future observations only.
- The final modeling table contains both the portfolio and target columns.

At the end of the completed data-pipeline work, the test suite reported `6 passed`.

## 6. What has not been implemented

These files exist as planned modules but still need to be coded by you:

- `src/features/rolling_features.py`
- `src/features/regimes.py`
- `src/models/baselines.py`
- `src/models/garch.py`
- `src/models/lstm.py`
- `src/evaluation/walk_forward.py`
- `src/evaluation/metrics.py`

The recommended next implementation task is `src/models/baselines.py`: implement a rolling-volatility forecast first, then EWMA, and compare their predictions with the target already created by the pipeline.
