from importlib import import_module

from fastapi.testclient import TestClient


api_module = import_module("api.app")


def make_payload() -> dict:
    observation = {
        "unit_id": 1,
        "cycle": 1,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
    }
    observation.update({f"sensor_{number}": 0.0 for number in range(1, 22)})
    return {
        "subset_id": "fd004",
        "model_type": "RandomForestRegressor",
        "observations": [observation],
    }


def test_predict_resolves_run_and_calls_inference(monkeypatch):
    calls = {}
    payload = make_payload()

    def fake_fetch_run_id(**kwargs):
        calls["fetch"] = kwargs
        return "training-run-id"

    def fake_predict_rul(subset_id, training_run_id, observations):
        calls["predict"] = (subset_id, training_run_id, observations)
        return 47.63

    monkeypatch.setattr(api_module, "fetch_run_id", fake_fetch_run_id)
    monkeypatch.setattr(api_module, "predict_rul", fake_predict_rul)

    with TestClient(api_module.app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "subset_id": "FD004",
        "model_type": "RandomForestRegressor",
        "predicted_rul": 47.63,
    }
    assert calls["fetch"] == {
        "subset_id": "FD004",
        "experiment_type": "training",
        "model_type": "RandomForestRegressor",
    }
    assert calls["predict"] == (
        "FD004",
        "training-run-id",
        payload["observations"],
    )


def test_health_reports_ready():
    with TestClient(api_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
