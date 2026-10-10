# Implementation plan

## First vertical slice: data to a validated target

1. Download and archive the 30-industry daily return source with provenance.
2. Implement `src.data.loader.load_industry_returns` for the source's nonstandard header/footer layout.
3. Implement validation and an equal-weight portfolio constructor.
4. Build the five-day forward realized-volatility target and test its date alignment with a tiny synthetic series.
5. Save a processed dataset and produce one diagnostic plot/table.

This slice is the right next step because every later model depends on correct units, dates, and target alignment. It also lets us establish a naive rolling-volatility benchmark before investing in GARCH, clustering, or PyTorch.

## Subsequent milestones

1. Rolling-volatility and EWMA baselines plus MAE/RMSE/QLIKE.
2. Walk-forward splitter with horizon embargo and tests.
3. Training-window-only regime clustering and regime diagnostics.
4. GARCH(1,1) benchmark.
5. LSTM baseline, then a regime-aware LSTM with identical splits.
