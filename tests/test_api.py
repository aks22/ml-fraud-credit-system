"""
tests/test_api.py
------------------
Integration tests for FastAPI endpoints.

These tests use TestClient (synchronous) and mock the model store
so that tests run without requiring trained models on disk.

Run with:
    pytest tests/test_api.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Mock model setup ──────────────────────────────────────────────────────────

def make_mock_fraud_model():
    """Returns a mock XGBoost model that returns low fraud probability."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.95, 0.05]])  # 5% fraud prob
    return model


def make_mock_credit_model():
    """Returns a mock RF model that predicts 'Good' credit risk."""
    model = MagicMock()
    model.predict.return_value = np.array([0])  # Good
    model.predict_proba.return_value = np.array([[0.80, 0.15, 0.05]])
    return model


def make_mock_pipeline():
    """Returns a mock pipeline with a preprocessor step."""
    pipeline = MagicMock()
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.zeros((1, 15))
    pipeline.named_steps = {"preprocessor": preprocessor, "smote": MagicMock()}
    return pipeline


@pytest.fixture
def client():
    """
    FastAPI test client with mocked models.
    Avoids needing trained model files to run API tests.
    """
    from src.api.main import create_app
    from src.api import model_loader

    app = create_app()

    # Inject mock models into the global model store
    store = model_loader.model_store
    store.fraud_pipeline = make_mock_pipeline()
    store.credit_pipeline = make_mock_pipeline()
    store.fraud_model = make_mock_fraud_model()
    store.credit_model = make_mock_credit_model()
    store.fraud_model_version = "test-mock-v1"
    store.credit_model_version = "test-mock-v1"

    # Patch DB to avoid needing PostgreSQL in tests
    with patch("src.api.routes.get_db") as mock_db:
        mock_session = MagicMock()
        mock_db.return_value = iter([mock_session])

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Sample payloads ───────────────────────────────────────────────────────────

FRAUD_PAYLOAD = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 181.0,
    "oldbalance_org": 181.0,
    "newbalance_orig": 0.0,
    "oldbalance_dest": 0.0,
    "newbalance_dest": 181.0,
}

CREDIT_PAYLOAD = {
    "age": 35,
    "duration_months": 24,
    "credit_amount": 5000.0,
    "installment_rate": 2,
    "residence_since": 3,
    "existing_credits": 1,
    "num_dependents": 1,
    "checking_account": "0-200 DM",
    "credit_history": "existing_paid",
    "purpose": "car",
    "savings": "< 100 DM",
    "employment": "1-4 years",
    "housing": "own",
    "job": "skilled",
}


# ── Tests: health endpoint ────────────────────────────────────────────────────

def test_health_returns_200(client):
    """Health endpoint should always return 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client):
    """Health response should include required fields."""
    response = client.get("/health")
    body = response.json()
    assert "status" in body
    assert "fraud_model_loaded" in body
    assert "credit_model_loaded" in body
    assert body["fraud_model_loaded"] is True
    assert body["credit_model_loaded"] is True


def test_root_endpoint(client):
    """Root endpoint should return API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "endpoints" in body


# ── Tests: fraud prediction ───────────────────────────────────────────────────

def test_fraud_prediction_returns_200(client):
    """Fraud prediction endpoint should return 200 with valid input."""
    response = client.post("/predict/fraud", json=FRAUD_PAYLOAD)
    assert response.status_code == 200


def test_fraud_prediction_response_fields(client):
    """Fraud prediction response should include all required fields."""
    response = client.post("/predict/fraud", json=FRAUD_PAYLOAD)
    body = response.json()
    assert "fraud_probability" in body
    assert "is_fraud" in body
    assert "threshold" in body
    assert "model_version" in body


def test_fraud_probability_range(client):
    """Fraud probability should be between 0 and 1."""
    response = client.post("/predict/fraud", json=FRAUD_PAYLOAD)
    prob = response.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0


def test_fraud_validation_rejects_negative_amount(client):
    """Pydantic should reject negative transaction amounts."""
    invalid_payload = {**FRAUD_PAYLOAD, "amount": -100.0}
    response = client.post("/predict/fraud", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_fraud_validation_rejects_invalid_type(client):
    """Pydantic should reject invalid transaction types."""
    invalid_payload = {**FRAUD_PAYLOAD, "type": "INVALID_TYPE"}
    response = client.post("/predict/fraud", json=invalid_payload)
    assert response.status_code == 422


# ── Tests: credit prediction ──────────────────────────────────────────────────

def test_credit_prediction_returns_200(client):
    response = client.post("/predict/credit-risk", json=CREDIT_PAYLOAD)
    assert response.status_code == 200


def test_credit_prediction_response_fields(client):
    response = client.post("/predict/credit-risk", json=CREDIT_PAYLOAD)
    body = response.json()
    assert "credit_tier" in body
    assert "confidence_scores" in body
    assert "model_version" in body


def test_credit_tier_valid_value(client):
    """Credit tier should be one of Good / Moderate / Poor."""
    response = client.post("/predict/credit-risk", json=CREDIT_PAYLOAD)
    tier = response.json()["credit_tier"]
    assert tier in ("Good", "Moderate", "Poor")


def test_credit_confidence_scores_sum_to_one(client):
    """Confidence scores across all three tiers should sum to approximately 1."""
    response = client.post("/predict/credit-risk", json=CREDIT_PAYLOAD)
    scores = response.json()["confidence_scores"]
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.01


# ── Tests: combined prediction ────────────────────────────────────────────────

def test_combined_prediction_returns_200(client):
    combined_payload = {"fraud": FRAUD_PAYLOAD, "credit": CREDIT_PAYLOAD}
    response = client.post("/predict/combined", json=combined_payload)
    assert response.status_code == 200


def test_combined_prediction_includes_both_results(client):
    combined_payload = {"fraud": FRAUD_PAYLOAD, "credit": CREDIT_PAYLOAD}
    response = client.post("/predict/combined", json=combined_payload)
    body = response.json()
    assert "fraud" in body
    assert "credit" in body
    assert "request_id" in body


def test_combined_prediction_request_id_format(client):
    """Request ID should be a valid UUID string."""
    import re
    combined_payload = {"fraud": FRAUD_PAYLOAD, "credit": CREDIT_PAYLOAD}
    response = client.post("/predict/combined", json=combined_payload)
    request_id = response.json()["request_id"]
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    assert uuid_pattern.match(request_id)
