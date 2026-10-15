"""Export verified compact research artifacts to the React dashboard."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/tables/research_all_model_metrics.csv"
OUTPUT = ROOT / "website/src/data/research.generated.json"
STRESS_SOURCE = ROOT / "reports/tables/v2_stress_period_metrics.csv"

FAMILIES = {
    "Rolling": "Baseline", "EWMA": "Baseline", "GARCH-normal": "Statistical",
    "GARCH-Student-t": "Statistical", "GJR-GARCH-Student-t": "Statistical",
    "Regime-switching": "Regime", "LSTM": "Neural", "Regime-aware LSTM": "Neural",
    "Risk-feature LSTM": "Neural", "Risk-feature + regime LSTM": "Neural",
    "QLIKE-objective Risk-feature + regime LSTM": "Neural",
}


def export(source=SOURCE, output=OUTPUT, stress_source=STRESS_SOURCE):
    data = pd.read_csv(source)
    required = {"model", "n_test", "mae", "rmse", "qlike"}
    if set(data.columns) != required or set(data.model) != set(FAMILIES) or data.model.duplicated().any():
        raise ValueError("Research metric schema/model set differs from the dashboard contract.")
    if not data.n_test.eq(2520).all() or not (data[["mae", "rmse", "qlike"]] > 0).all().all():
        raise ValueError("Dashboard metrics must be positive full-development results.")
    records = [{"model": row.model, "family": FAMILIES[row.model], "nTest": int(row.n_test),
                "mae": row.mae, "rmse": row.rmse, "qlike": row.qlike}
               for row in data.sort_values("qlike").itertuples(index=False)]
    stress_records = []
    if stress_source.is_file():
        stress = pd.read_csv(stress_source)
        stress_required = {"period", "model", "n", "mae", "rmse", "qlike", "underprediction_rate"}
        if not stress_required.issubset(stress) or len(stress) != 6:
            raise ValueError("Stress metric schema differs from the dashboard contract.")
        stress_records = stress[list(stress_required)].to_dict(orient="records")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schemaVersion": 1, "models": records,
        "stressPeriods": stress_records}, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    print(f"Exported {len(export())} verified models to {OUTPUT.relative_to(ROOT)}")
