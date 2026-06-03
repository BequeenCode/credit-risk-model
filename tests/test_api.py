"""
Tests for the FastAPI credit-risk service (Task 6).

These exercise request validation and the prediction contract using FastAPI's
TestClient. They are skipped gracefully if the trained artifacts are missing
(e.g. before `python src/train.py` has been run).
"""

import os

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

MODEL_READY = os.path.exists("models/best_model.pkl") and os.path.exists(
    "models/feature_pipeline.pkl"
)

VALID_PAYLOAD = {
    "age": 45,
    "income": 60000,
    "loan_amount": 150000,
    "loan_term": 36,
    "interest_rate": 5.5,
    "employment_years": 8,
    "num_accounts": 5,
    "num_delinquencies": 0,
    "credit_score": 650,
    "employment_type": "Employed",
    "home_ownership": "Mortgage",
    "loan_purpose": "Auto",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body and "model_loaded" in body


def test_predict_rejects_invalid_input(client):
    bad = dict(VALID_PAYLOAD)
    bad["credit_score"] = 9999  # outside the allowed 300-850 range
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # pydantic validation error


def test_predict_rejects_unknown_category(client):
    bad = dict(VALID_PAYLOAD)
    bad["home_ownership"] = "Spaceship"  # not in the allowed Literal set
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


@pytest.mark.skipif(not MODEL_READY, reason="trained model artifacts not present")
def test_predict_returns_valid_probability(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_label"] in (0, 1)
