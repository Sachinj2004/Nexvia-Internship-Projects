"""
Automated Test Suite for Nexvia Fraud Detection REST API (src/api.py)
Tests all endpoints: /health, /model-info, /predict, /predict/batch, and schema validation.
"""

from pathlib import Path
import sys
import time

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app
from src.predict import EXPECTED_FEATURES

client = TestClient(app)


# -----------------------------------------------------------------------------
# Fixtures & Sample Payloads
# -----------------------------------------------------------------------------
@pytest.fixture
def legitimate_payload():
    """Synthetic standard transaction with low fraud probability."""
    payload = {col: 0.0 for col in EXPECTED_FEATURES}
    payload["Time"] = 100.0
    payload["Amount"] = 49.99
    payload["threshold"] = 0.5
    return payload


@pytest.fixture
def fraudulent_payload():
    """Known extreme anomaly transaction matching typical fraud PCA patterns."""
    payload = {col: 0.0 for col in EXPECTED_FEATURES}
    payload["Time"] = 406.0
    payload["V1"] = -2.3122
    payload["V2"] = 1.9520
    payload["V3"] = -1.6099
    payload["V4"] = 3.9979
    payload["V7"] = -2.5374
    payload["V10"] = -2.7723
    payload["V12"] = -2.8999
    payload["V14"] = -4.2893
    payload["V17"] = -2.8301
    payload["Amount"] = 0.0
    payload["threshold"] = 0.5
    return payload


# -----------------------------------------------------------------------------
# Endpoint Tests
# -----------------------------------------------------------------------------
def test_root_endpoint():
    """Verify that root endpoint provides documentation links and online status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data
    assert "/health" in data["endpoints"].values()
    assert "/predict" in data["endpoints"].values()


def test_health_endpoint():
    """Verify health endpoint reports healthy status and model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "XGBoost" in data["model_type"]
    assert data["memory_rss_mb"] > 0
    assert data["uptime_seconds"] >= 0


def test_model_info_endpoint():
    """Verify model-info endpoint returns expected schema metadata and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == 30
    assert len(data["expected_features"]) == 30
    assert "Time" in data["expected_features"]
    assert "Amount" in data["expected_features"]
    assert data["default_threshold"] == 0.50
    assert "benchmarks" in data
    # Check that saved evaluation metrics are populated
    assert "pr_auc" in data["benchmarks"]


def test_predict_legitimate_transaction(legitimate_payload):
    """Verify that a standard transaction is classified as APPROVED / Legitimate."""
    response = client.post("/predict", json=legitimate_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 0
    assert data["risk_label"] == "Legitimate"
    assert data["status"] == "APPROVED"
    assert 0.0 <= data["fraud_probability"] < 0.5
    assert data["threshold_used"] == 0.5
    assert data["inference_latency_ms"] > 0
    assert "timestamp" in data


def test_predict_fraudulent_transaction(fraudulent_payload):
    """Verify that an anomalous transaction is flagged as FLAGGED_FOR_REVIEW / Fraud."""
    response = client.post("/predict", json=fraudulent_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 1
    assert data["risk_label"] == "Fraud"
    assert data["status"] == "FLAGGED_FOR_REVIEW"
    assert data["fraud_probability"] >= 0.5
    assert data["threshold_used"] == 0.5


def test_predict_custom_threshold(legitimate_payload):
    """Verify that custom decision cutoff thresholds are respected."""
    # With an extremely low threshold (0.0000001), even small risk gets flagged
    low_thresh_payload = legitimate_payload.copy()
    low_thresh_payload["threshold"] = 0.0000001
    resp_low = client.post("/predict", json=low_thresh_payload)
    assert resp_low.status_code == 200
    assert resp_low.json()["threshold_used"] == 0.0000001

    # With a high threshold (0.9999), it is approved
    high_thresh_payload = legitimate_payload.copy()
    high_thresh_payload["threshold"] = 0.9999
    resp_high = client.post("/predict", json=high_thresh_payload)
    assert resp_high.status_code == 200
    assert resp_high.json()["threshold_used"] == 0.9999
    assert resp_high.json()["predicted_class"] == 0


def test_predict_missing_feature_validation():
    """Verify that omitting required features raises 422 Unprocessable Entity."""
    incomplete_payload = {"Time": 100.0, "Amount": 25.0}  # Missing V1-V28
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_negative_amount_validation(legitimate_payload):
    """Verify that negative amounts fail schema validation with 422."""
    invalid_payload = legitimate_payload.copy()
    invalid_payload["Amount"] = -10.0
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_out_of_bounds_threshold(legitimate_payload):
    """Verify that thresholds outside [0.0, 1.0] fail validation with 422."""
    invalid_payload = legitimate_payload.copy()
    invalid_payload["threshold"] = 1.5
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_batch(legitimate_payload, fraudulent_payload):
    """Verify batch prediction aggregates metrics and processes multiple items."""
    batch_payload = {
        "transactions": [legitimate_payload, fraudulent_payload],
        "threshold": 0.5
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert data["approved_count"] == 1
    assert data["flagged_count"] == 1
    assert data["fraud_rate_percent"] == 50.0
    assert len(data["results"]) == 2
    assert data["results"][0]["predicted_class"] == 0
    assert data["results"][1]["predicted_class"] == 1


def test_predict_batch_empty():
    """Verify that empty batch list is rejected with 422 validation error."""
    response = client.post("/predict/batch", json={"transactions": []})
    assert response.status_code == 422


def test_latency_performance_target(legitimate_payload):
    """Verify that single transaction scoring achieves the sub-10ms design target."""
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=legitimate_payload)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        latencies.append((t1 - t0) * 1000.0)

    # Average latency excluding first potential cold call
    avg_latency = sum(latencies[1:]) / len(latencies[1:])
    print(f"\nAverage client latency: {avg_latency:.2f} ms")
    assert avg_latency < 10.0, f"Average latency ({avg_latency:.2f} ms) exceeded 10ms threshold!"


# -----------------------------------------------------------------------------
# Feast Velocity Feature Enrichment Tests (Stage 2)
# -----------------------------------------------------------------------------
def test_predict_with_valid_feast_transaction_id(legitimate_payload):
    """Verify that specifying a valid transaction_id enriches the response with Feast features."""
    payload = legitimate_payload.copy()
    payload["transaction_id"] = 0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == 0
    assert data["enriched_features"] is not None
    assert "rolling_amount_mean_1h" in data["enriched_features"]
    assert "amount_to_window_mean_ratio" in data["enriched_features"]
    assert "rolling_tx_count_1h" in data["enriched_features"]
    assert data["predicted_class"] == 0
    assert data["status"] == "APPROVED"


def test_predict_with_missing_feast_transaction_id(legitimate_payload):
    """Verify that an unmaterialized transaction_id returns enriched_features: null without error."""
    payload = legitimate_payload.copy()
    payload["transaction_id"] = 999999
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == 999999
    assert data["enriched_features"] is None
    assert data["predicted_class"] == 0


def test_predict_with_feast_worker_failure_graceful_recovery(legitimate_payload):
    """Verify that if Feast feature lookup fails/errors, prediction still succeeds gracefully."""
    payload = legitimate_payload.copy()
    payload["transaction_id"] = 0
    with patch("src.api.get_feast_features", side_effect=RuntimeError("Simulated Feast lookup crash")):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == 0
        assert data["enriched_features"] is None
        assert data["status"] == "APPROVED"


def test_predict_backward_compatibility_without_transaction_id(legitimate_payload):
    """Verify full backward compatibility when transaction_id is completely omitted."""
    assert "transaction_id" not in legitimate_payload
    response = client.post("/predict", json=legitimate_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] is None
    assert data["enriched_features"] is None
    assert data["predicted_class"] == 0
    assert data["status"] == "APPROVED"
