# Data contract

## Input

The loader accepts a daily industry-return table with a date index and one numeric column per industry. Returns must be decimal simple returns (for example, `0.015` for 1.5%), not percentages.

## Validation rules

- Dates are parsed, unique, strictly increasing, and business-day-like.
- Industry columns are numeric and contain no duplicated names.
- Missing values and source sentinel values are surfaced rather than silently filled.
- Return-unit conversion is explicit and recorded in metadata.
- Portfolio weights sum to one and align exactly to the loaded columns.

## Output

The processed return frame includes the daily fixed-weight portfolio return. Targets are forward-looking realized volatility computed from days `t+1` through `t+h`, never including information available after a forecast origin.
