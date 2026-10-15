# Raw data provenance

Keep the original CSV unmodified and archive it outside the working copy before obtaining updates. Raw data is ignored by Git.

- Source: Kenneth R. French Data Library, daily 30 Industry Portfolios.
- Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- CSV ZIP: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_Daily_CSV.zip
- Local file: `30_Industry_Portfolios_Daily.csv`.
- Inspected/reproduced: 2026-10-15. Original retrieval date was not recorded.
- Value-weighted table: 1926-07-01 through 2026-08-31; 26,317 observations and 30 industries.
- SHA-256: `dd2ca61d3cfe66c279e347a91535e12520907ceb98808ca42fff05c0657dc924`.
- Source units: percent; loader output: decimal returns.
- Preserve valid historical Saturdays and the source-confirmed Services return of 111.85% on 1931-01-29. No automatic clipping.

The development evaluation ends 2026-08-24 because the following five observations are needed for its last target. All available history is considered exposed. The prospective final protocol begins strictly after 2026-10-15.

A later download can revise historical values. The final evaluator rejects changes to the frozen parsed history through 2026-08-31. Preserve the original vintage or document a formal amendment; do not silently replace the evidence.
