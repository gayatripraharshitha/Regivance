# Regivance V2 assessment

## Decision

The review identifies real weaknesses rather than cosmetic gaps. V2 implements every recommendation that can be evaluated with the existing development data. The prospective result remains pending because no method can create genuinely unseen future observations today.

| Review item | Decision | V2 response |
| --- | --- | --- |
| No untouched prospective results | Preserve as a high-severity limitation | Keep the registered protocol and describe the required 257 future observations |
| Stress-period failures | Address now | Add exclusive normal/high-volatility/2020 metrics and expose them in the dashboard |
| Limited economic value | Address now, with bounded claims | Add a five-observation conditional VaR coverage demonstration |
| One portfolio construction | Address now as sensitivity analysis | Add a lagged 63-observation inverse-industry-volatility portfolio |
| Small fixed neural configuration | Partially address now | Run five seed bases on eight fixed, stress-inclusive folds; reserve full tuning for future work |
| K-means semantics | Clarify rather than relabel | Retain relative lower/middle/higher volatility ranks and avoid economic-state claims |
| Static dashboard | Improve now | Generate dashboard model/stress values from verified CSVs; a live API remains future engineering work |
| CI/report wording | Address now | Add frontend CI and distinguish committed summaries from locally regenerated bulk artifacts |

## Why these choices are proportionate

The stress, underprediction and VaR analyses directly answer the risk-manager objection. The seed experiment measures a known source of neural uncertainty without turning V2 into an open-ended tuning exercise. The second portfolio checks whether the statistical ranking immediately collapses under a reasonable alternative construction. Automated dashboard export removes a preventable consistency problem.

None of these development-sample additions can replace prospective evidence. They are labeled exploratory throughout the report and dashboard.

## Headline evidence

- During the exclusive February–April 2020 group, the risk-plus-regime LSTM has QLIKE 8.1162 and underpredicts 88.7% of origins. Student-t GJR-GARCH has QLIKE 1.0605 and underpredicts 43.5%.
- In the most extreme 5% of volatility increases, LSTM QLIKE is 3.0528 versus 1.9791 for GJR-GARCH.
- The Normal VaR demonstration shows GJR-GARCH near nominal 95% coverage but inadequate 99% tail coverage; the LSTM undercovers at both levels.
- Across five seed bases on 168 fixed stress-inclusive origins, LSTM QLIKE ranges from 2.5192 to 3.4639.
- On the inverse-volatility portfolio, GJR-GARCH remains strongest among Rolling, EWMA and GJR-GARCH, with QLIKE 0.4457.

## Remaining scientific validation

- Execute the registered future evaluation when eligible.
- Repeat multi-seed evaluation over all folds if compute and time permit.
- Add independent datasets or asset universes.
- Evaluate crisis-specific calibration without selecting thresholds after seeing outcomes.
- Connect volatility forecasts to a prespecified risk or allocation decision with costs and utility.

## Remaining engineering extension

- Publish versioned experiment bundles automatically.
- Add a read-only API for large forecast histories if live exploration is required.
- Add browser-level end-to-end tests, deployment automation, observability and load testing.
- Add authentication only if the project becomes a multi-user private research service.
