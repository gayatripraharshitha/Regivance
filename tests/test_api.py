from fastapi.testclient import TestClient
import pandas as pd

from api import app as api_module


def artifacts(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.csv"
    metrics = tmp_path / "metrics.csv"
    pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "actual": [.1, .2, .3], "Model A": [.11, .19, .28],
        "fold": [1, 1, 1], "regime": ["lower", "middle", "higher"],
    }).to_csv(predictions, index=False)
    pd.DataFrame({"model": ["Model A"], "n_test": [3], "mae": [.01], "rmse": [.02], "qlike": [.3]}).to_csv(metrics, index=False)
    monkeypatch.setattr(api_module, "PREDICTIONS", predictions)
    monkeypatch.setattr(api_module, "METRICS", metrics)
    api_module._read_csv.cache_clear()
    return TestClient(api_module.app)


def test_health_and_metadata(tmp_path, monkeypatch):
    client = artifacts(tmp_path, monkeypatch)
    assert client.get("/api/v1/health").json() == {"status":"ok", "readOnly":True, "artifactsAvailable":True}
    result = client.get("/api/v1/metadata").json()
    assert result["models"] == ["Model A"]
    assert result["observations"] == 3


def test_models_and_paginated_forecasts(tmp_path, monkeypatch):
    client = artifacts(tmp_path, monkeypatch)
    assert client.get("/api/v1/models").json()["items"][0]["model"] == "Model A"
    response = client.get("/api/v1/forecasts", params={"model":"Model A", "start":"2020-01-02", "limit":1})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2 and body["count"] == 1
    assert body["items"][0] == {"date":"2020-01-02", "actual":.2, "forecast":.19, "fold":1, "regime":"middle"}


def test_forecasts_reject_unknown_model_and_invalid_range(tmp_path, monkeypatch):
    client = artifacts(tmp_path, monkeypatch)
    assert client.get("/api/v1/forecasts", params={"model":"missing"}).status_code == 404
    response = client.get("/api/v1/forecasts", params={"model":"Model A", "start":"2020-02-01", "end":"2020-01-01"})
    assert response.status_code == 422
