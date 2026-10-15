# V2 LSTM initialization sensitivity

## Design

The risk-feature-plus-regime LSTM is retrained with five prespecified seed bases (7, 42, 101, 202, 999) on eight fixed folds (1, 21, 41, 43, 61, 81, 101, 120). Each run uses seed base plus fold number and retains the frozen 21-observation lookback, hidden size 12, maximum 12 epochs and validation early stopping. The subset contains 168 test origins and deliberately includes Fold 43, the February–March 2020 stress fold.

This compact experiment measures initialization sensitivity. It is exploratory, uses a historical subset and is not a substitute for full repeated-seed evaluation or prospective evidence.

## Per-seed results

| Seed base | Test origins | MAE | RMSE | QLIKE |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 168 | 0.015588 | 0.036264 | 3.094592 |
| 42 | 168 | 0.015350 | 0.035655 | 2.519233 |
| 101 | 168 | 0.015524 | 0.035868 | 2.713847 |
| 202 | 168 | 0.015915 | 0.036977 | 3.463906 |
| 999 | 168 | 0.015570 | 0.036298 | 3.146982 |

## Across-seed dispersion

| Statistic | MAE | RMSE | QLIKE |
| --- | ---: | ---: | ---: |
| mean | 0.015589 | 0.036213 | 2.987712 |
| std | 0.000205 | 0.000506 | 0.373505 |
| min | 0.015350 | 0.035655 | 2.519233 |
| max | 0.015915 | 0.036977 | 3.463906 |

On the identical subset, the originally saved LSTM has QLIKE 2.519233; Student-t GJR-GARCH has QLIKE 0.739978. Seed dispersion quantifies one source of neural uncertainty, while model comparison remains secondary because these folds were selected for diagnostic coverage.
