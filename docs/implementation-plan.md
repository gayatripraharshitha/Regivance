# Implementation status — 2026-10-15

The original scaffold has become a functioning development research system. Approximate overall completion is 95%; this is a planning estimate, not a measured score.

| Phase | Status | Evidence |
| --- | --- | --- |
| Data foundation | Complete | Source loader, validation, daily equal-industry portfolio, five-observation target, tests |
| Statistical evaluation | Complete | Rolling/EWMA/three GARCH variants; 120 chronological folds; 2,520 common dates |
| Market regimes and neural models | Complete | Train-only K-means; regime selector; four LSTM inputs; full QLIKE-objective experiment |
| Development validation and presentation | Complete | HAC/bootstrap uncertainty, year/regime/shock/offset checks, plots, research report, provenance |
| Prospective final evaluation implementation | Complete | Fixed future origins, frozen code/config/history/environment, real synthetic integration test |
| Prospective real-data evidence | Waiting for future observations | First 252 origins strictly after 2026-10-15 plus five target observations |

## Verification

121 automated tests pass locally without warnings. Dependency consistency passes. A two-platform GitHub Actions workflow has been added; remote execution is not yet verified.

## Next real-world action

Preserve the archived source and frozen protocol. Once the 257 required post-freeze observations are available, follow the README's prospective command using a separate updated source file. Do not tune models on final-period outcomes. Preserve and report any historical source revisions or necessary protocol amendments.

Development evidence currently favors Student-t GJR-GARCH. Lack of a regime improvement is a valid research outcome, not a reason to change the evaluation after seeing results.
