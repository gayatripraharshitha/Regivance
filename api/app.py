"""FastAPI service for reviewed Regivance development artifacts.

This application never trains a model or writes research outputs. It exposes
versioned, paginated views over compact artifacts created by the research
pipeline.
"""
from functools import lru_cache
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports/tables/research_all_model_predictions.csv"
METRICS = ROOT / "reports/tables/research_all_model_metrics.csv"
NON_MODELS = {"date", "actual", "fold", "regime"}


def _origins() -> list[str]:
    configured = os.getenv("REGIVANCE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _signature(path: Path) -> tuple[str, int, int]:
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"Required artifact is unavailable: {path.name}")
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _read_csv(signature: tuple[str, int, int]) -> pd.DataFrame:
    data = pd.read_csv(signature[0])
    if data.empty:
        raise HTTPException(status_code=503, detail="The reviewed artifact is empty.")
    return data


def predictions() -> pd.DataFrame:
    data = _read_csv(_signature(PREDICTIONS)).copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    return data


def metrics() -> pd.DataFrame:
    return _read_csv(_signature(METRICS)).copy()


app = FastAPI(
    title="Regivance Research API",
    version="1.0.0",
    description="Read-only access to reviewed historical development artifacts. No live inference or model training.",
)
app.add_middleware(CORSMiddleware, allow_origins=_origins(), allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])


@app.get("/api/v1/health", tags=["system"])
def health():
    available = PREDICTIONS.is_file() and METRICS.is_file()
    return {"status": "ok" if available else "degraded", "readOnly": True, "artifactsAvailable": available}


@app.get("/api/v1/metadata", tags=["research"])
def metadata():
    data = predictions()
    model_names = [column for column in data.columns if column not in NON_MODELS]
    return {
        "schemaVersion": 1,
        "scope": "historical_development",
        "observations": len(data),
        "startDate": data.date.min().date().isoformat(),
        "endDate": data.date.max().date().isoformat(),
        "models": model_names,
        "target": "five-observation root mean squared future portfolio returns",
        "warning": "Development evidence only; not live or prospective investment advice.",
    }


@app.get("/api/v1/models", tags=["research"])
def model_scores():
    data = metrics().sort_values("qlike")
    return {"count": len(data), "items": data.to_dict(orient="records")}


@app.get("/api/v1/forecasts", tags=["research"])
def forecast_history(
    model: str = Query(..., description="Exact model name returned by /metadata"),
    start: str | None = Query(None, description="Inclusive ISO date"),
    end: str | None = Query(None, description="Inclusive ISO date"),
    offset: int = Query(0, ge=0),
    limit: int = Query(250, ge=1, le=500),
):
    data = predictions()
    available = [column for column in data.columns if column not in NON_MODELS]
    if model not in available:
        raise HTTPException(status_code=404, detail={"message": "Unknown model.", "availableModels": available})
    try:
        if start:
            data = data[data.date >= pd.Timestamp(start)]
        if end:
            data = data[data.date <= pd.Timestamp(end)]
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="start and end must be valid ISO dates.") from None
    if start and end and pd.Timestamp(start) > pd.Timestamp(end):
        raise HTTPException(status_code=422, detail="start must not be after end.")
    total = len(data)
    page = data.iloc[offset: offset + limit]
    items = [{
        "date": row["date"].date().isoformat(), "actual": float(row["actual"]),
        "forecast": float(row[model]), "fold": int(row["fold"]),
        "regime": str(row["regime"]),
    } for _, row in page.iterrows()]
    return {"model": model, "total": total, "offset": offset, "limit": limit, "count": len(items), "items": items}
