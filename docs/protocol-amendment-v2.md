# Prospective protocol amendment V2

Registered on October 15, 2026, before any post-freeze observations were available.

## Reason

The original registration fingerprinted every Python file under `scripts/`, including report generators and unrelated exploratory analyses. Adding the V2 stress, VaR, portfolio and dashboard-export scripts therefore triggered protocol drift even though the prospective forecast specifications and their dependencies were unchanged.

## Amendment

The V2 registration fingerprints all Python modules under `src/` and the six scripts imported by or defining the prospective evaluation:

- `final_evaluation.py`
- `compare_statistical_models.py`
- `compare_lstm_models.py`
- `compare_lstm_risk_features.py`
- `compare_lstm_qlike_objective.py`
- `analyze_dependence_robust_comparison.py`

It continues to fingerprint the configuration, parsed historical returns, source file and full Python package environment. New unrelated reporting scripts no longer invalidate the forecast protocol. Changes to model, feature, evaluation or dependency code still do.

The original [frozen protocol](../configs/frozen_protocol.json) is preserved. V2 uses `configs/frozen_protocol_v2.json`. The evaluation dates, model set, hyperparameters, primary QLIKE metric, reference Student-t GJR-GARCH model, freeze date and exposed-data boundary are unchanged.

This is an engineering correction to fingerprint scope. It does not use new market outcomes or change the scientific hypothesis.
