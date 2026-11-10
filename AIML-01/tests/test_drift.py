"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Evidently AI Drift Monitoring Test Suite: tests/test_drift.py

Tests:
1. Reference dataset schema, row count, null check, and feature alignment.
2. Thread-safe in-memory transaction buffer mechanics.
3. Graceful handling of insufficient data (<10 samples).
4. End-to-end drift evaluation on baseline data (HTML and JSON generation).
5. Deterministic drift detection on deliberately perturbed features.
6. FastAPI endpoints: GET /drift/status, GET /drift/report, POST /drift/evaluate.
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api import app
from src.drift import (
    DEFAULT_REFERENCE_PATH,
    EXPECTED_FEATURES,
    FraudDriftMonitor,
    drift_monitor
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


# -----------------------------------------------------------------------------
# Test 1: Reference Dataset Schema & Integrity
# -----------------------------------------------------------------------------
def test_reference_dataset_schema_and_integrity():
    """Verify data/reference_data.parquet conforms to exact expected specifications."""
    assert DEFAULT_REFERENCE_PATH.exists(), f"Reference Parquet missing at {DEFAULT_REFERENCE_PATH}"

    df_ref = pd.read_parquet(DEFAULT_REFERENCE_PATH)
    assert len(df_ref) == 227845, f"Expected 227,845 rows, received {len(df_ref)}"
    assert len(df_ref.columns) == 33, f"Expected 33 columns, received {len(df_ref.columns)}"

    # All 30 input features must be present in exact order as the first 30 columns
    for idx, feature in enumerate(EXPECTED_FEATURES):
        assert df_ref.columns[idx] == feature, f"Mismatch at index {idx}: expected {feature}, got {df_ref.columns[idx]}"

    # Monitoring columns
    assert "target" in df_ref.columns
    assert "prediction" in df_ref.columns
    assert "probability" in df_ref.columns

    # Zero missing values across all features
    assert df_ref.isnull().sum().sum() == 0, "Reference dataset contains NaN values"

    # Target class breakdown
    fraud_count = int((df_ref["target"] == 1).sum())
    legit_count = int((df_ref["target"] == 0).sum())
    assert fraud_count == 394, f"Expected 394 fraud cases, got {fraud_count}"
    assert legit_count == 227451, f"Expected 227,451 legit cases, got {legit_count}"


# -----------------------------------------------------------------------------
# Test 2: In-Memory Sliding Window Buffer
# -----------------------------------------------------------------------------
def test_drift_monitor_buffer_mechanics():
    """Verify thread-safe transaction recording and bounded buffer behavior."""
    monitor = FraudDriftMonitor(buffer_size=10)
    assert monitor.get_buffer_size() == 0

    # Create dummy transaction
    sample_tx = {col: 0.5 for col in EXPECTED_FEATURES}
    sample_tx["Amount"] = 99.99

    # Record single transaction
    monitor.record_transaction(sample_tx)
    assert monitor.get_buffer_size() == 1

    # Record 15 transactions to test eviction limit
    batch_txs = [{col: float(i) for col in EXPECTED_FEATURES} for i in range(15)]
    monitor.record_batch(batch_txs)

    # Buffer must be bounded to buffer_size (10)
    assert monitor.get_buffer_size() == 10

    # Retrieve DataFrame
    buf_df = monitor.get_buffer_dataframe()
    assert isinstance(buf_df, pd.DataFrame)
    assert len(buf_df) == 10
    assert list(buf_df.columns) == EXPECTED_FEATURES

    # Clear buffer
    monitor.clear_buffer()
    assert monitor.get_buffer_size() == 0


# -----------------------------------------------------------------------------
# Test 3: Insufficient Data Handling
# -----------------------------------------------------------------------------
def test_drift_evaluation_insufficient_data():
    """Verify evaluate_drift() gracefully handles samples below minimum threshold (<10)."""
    monitor = FraudDriftMonitor(buffer_size=50)
    monitor.clear_buffer()

    # Empty buffer
    res_empty = monitor.evaluate_drift()
    assert res_empty["status"] == "insufficient_data"
    assert res_empty["dataset_drift"] is False
    assert "Minimum 10 samples required" in res_empty["message"]

    # 3 samples (still below 10)
    sample_tx = {col: 1.0 for col in EXPECTED_FEATURES}
    monitor.record_batch([sample_tx] * 3)
    res_few = monitor.evaluate_drift()
    assert res_few["status"] == "insufficient_data"
    assert res_few["current_rows"] == 3


# -----------------------------------------------------------------------------
# Test 4: End-to-End Baseline Evaluation & Artifact Generation
# -----------------------------------------------------------------------------
def test_drift_evaluation_baseline_execution(tmp_path):
    """Verify drift evaluation runs in isolated environment and generates valid HTML and JSON."""
    test_html = tmp_path / "test_report.html"
    test_json = tmp_path / "test_metrics.json"

    monitor = FraudDriftMonitor(
        output_html=test_html,
        output_json=test_json
    )

    # Load 50 baseline rows from reference data
    ref_df = pd.read_parquet(DEFAULT_REFERENCE_PATH)
    sample_cur = ref_df[EXPECTED_FEATURES].iloc[:50].copy()

    # Run evaluation
    metrics = monitor.evaluate_drift(current_data=sample_cur, sample_size=1000)

    assert metrics["status"] == "success"
    assert "dataset_drift" in metrics
    assert "drift_share" in metrics
    assert "number_of_drifted_columns" in metrics
    assert "drift_by_columns" in metrics
    assert metrics["evaluated_features"] == 30

    # Verify artifacts were written to disk
    assert test_html.exists()
    assert test_json.exists()

    with open(test_html, "r", encoding="utf-8") as f:
        html_str = f.read()
    assert len(html_str) > 500
    assert "<html" in html_str.lower()


# -----------------------------------------------------------------------------
# Test 5: Deterministic Detection of Injected Data Drift
# -----------------------------------------------------------------------------
def test_drift_evaluation_detects_deliberate_drift(tmp_path):
    """Verify that severe, synthetic feature shifts are deterministically detected."""
    test_html = tmp_path / "drifted_report.html"
    test_json = tmp_path / "drifted_metrics.json"

    monitor = FraudDriftMonitor(
        output_html=test_html,
        output_json=test_json
    )

    ref_df = pd.read_parquet(DEFAULT_REFERENCE_PATH)
    drifted_df = ref_df[EXPECTED_FEATURES].iloc[:200].copy()

    # Inject extreme distribution shifts into specific features
    drifted_df["Amount"] = drifted_df["Amount"] * 50.0 + 5000.0
    drifted_df["V1"] = drifted_df["V1"] + 25.0
    drifted_df["V2"] = drifted_df["V2"] - 20.0
    drifted_df["V3"] = drifted_df["V3"] + 15.0
    drifted_df["V4"] = drifted_df["V4"] * 10.0

    metrics = monitor.evaluate_drift(current_data=drifted_df, sample_size=2000)

    assert metrics["status"] == "success"
    assert metrics["number_of_drifted_columns"] >= 4, (
        f"Expected at least 4 drifted columns, got {metrics['number_of_drifted_columns']}"
    )

    # The shifted columns must be specifically flagged
    drift_cols = metrics["drift_by_columns"]
    assert "Amount" in drift_cols and drift_cols["Amount"]["drift_detected"] is True
    assert "V1" in drift_cols and drift_cols["V1"]["drift_detected"] is True
    assert "V2" in drift_cols and drift_cols["V2"]["drift_detected"] is True


# -----------------------------------------------------------------------------
# Test 6: FastAPI Drift Endpoints Integration
# -----------------------------------------------------------------------------
def test_api_drift_status_endpoint():
    """Verify GET /drift/status returns 200 and expected monitoring dictionary."""
    response = client.get("/drift/status")
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "dataset_drift" in data
    assert "buffered_transactions" in data
    assert "report_available" in data


def test_api_drift_evaluate_and_report_endpoints():
    """Verify POST /drift/evaluate and GET /drift/report endpoints."""
    # Seed buffer with valid transactions
    sample_tx = {col: 0.0 for col in EXPECTED_FEATURES}
    sample_tx["Amount"] = 15.0

    # Score 12 transactions through /predict to populate drift buffer
    for i in range(12):
        tx = sample_tx.copy()
        tx["Time"] = float(i)
        res = client.post("/predict", json=tx)
        assert res.status_code == 200

    # Verify buffer is populated
    status_resp = client.get("/drift/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["buffered_transactions"] >= 12

    # Trigger drift evaluation via API
    eval_resp = client.post("/drift/evaluate?sample_size=1000")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["status"] == "success"
    assert "dataset_drift" in eval_data

    # Fetch generated HTML report
    report_resp = client.get("/drift/report")
    assert report_resp.status_code == 200
    assert "text/html" in report_resp.headers.get("content-type", "")
    assert "<html" in report_resp.text.lower()
