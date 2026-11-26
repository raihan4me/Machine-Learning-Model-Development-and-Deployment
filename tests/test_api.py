import pytest
from fastapi.testclient import TestClient

from app.main import app

GOOD = {"age": 34, "income": 52000, "loan_amount": 12000, "credit_score": 690,
        "employment_years": 5, "num_open_accounts": 3, "home_ownership": "RENT"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # 'with' runs the lifespan (model loading)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_single(client):
    r = client.post("/predict", json={"applications": [GOOD]})
    assert r.status_code == 200
    p = r.json()["predictions"][0]
    assert 0.0 <= p["default_probability"] <= 1.0
    assert p["decision"] in {"approve", "reject"}


def test_predict_batch(client):
    r = client.post("/predict", json={"applications": [GOOD] * 5})
    assert len(r.json()["predictions"]) == 5


@pytest.mark.parametrize("field,value", [("credit_score", 900), ("age", 12),
                                         ("home_ownership", "CASTLE"), ("income", -1)])
def test_invalid_input_rejected(client, field, value):
    bad = {**GOOD, field: value}
    assert client.post("/predict", json={"applications": [bad]}).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict", json={"applications": []}).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json={"applications": [GOOD]})
    body = client.get("/metrics").text
    assert "loan_predictions_total" in body
    assert "loan_prediction_latency_seconds_bucket" in body
