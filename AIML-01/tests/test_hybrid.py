"""
Unit Tests for Hybrid Fraud Predictor: tests/test_hybrid.py
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid import HybridFraudPredictor
from src.predict import EXPECTED_FEATURES


@pytest.fixture
def sample_legit_tx():
    tx = {f"V{i}": 0.0 for i in range(1, 29)}
    tx["Time"] = 100.0
    tx["Amount"] = 45.00
    return tx


@pytest.fixture
def sample_fraud_tx():
    tx = {f"V{i}": 0.0 for i in range(1, 29)}
    tx["Time"] = 406.0
    tx["Amount"] = 0.0
    tx["V1"] = -2.3122
    tx["V2"] = 1.9520
    tx["V3"] = -1.6099
    tx["V4"] = 3.9979
    tx["V14"] = -4.2893
    tx["V10"] = -2.7723
    tx["V12"] = -2.8999
    return tx


def test_hybrid_single_prediction_normal(sample_legit_tx):
    """Test hybrid scoring on normal legitimate transaction."""
    predictor = HybridFraudPredictor(enable_gnn=True)
    res = predictor.predict_single(sample_legit_tx)

    assert "fraud_probability" in res
    assert "xgb_probability" in res
    assert "gnn_probability" in res
    assert res["predicted_class"] == 0
    assert res["risk_label"] == "Legitimate"
    assert res["status"] == "APPROVED"
    assert res["scoring_mode"] in ["hybrid", "xgboost_fallback", "xgboost_only"]


def test_hybrid_single_prediction_fraud(sample_fraud_tx):
    """Test hybrid scoring on known anomaly transaction."""
    predictor = HybridFraudPredictor(enable_gnn=True)
    res = predictor.predict_single(sample_fraud_tx)

    assert res["fraud_probability"] > 0.80
    assert res["predicted_class"] == 1
    assert res["risk_label"] == "Fraud"
    assert res["status"] == "FLAGGED_FOR_REVIEW"


def test_hybrid_batch_prediction(sample_legit_tx, sample_fraud_tx):
    """Test batch scoring returns enriched DataFrame with all probability columns."""
    df = pd.DataFrame([sample_legit_tx, sample_fraud_tx])
    predictor = HybridFraudPredictor(enable_gnn=True)
    res_df = predictor.predict_batch(df)

    assert len(res_df) == 2
    for col in ["fraud_probability", "xgb_probability", "gnn_probability", "scoring_mode"]:
        assert col in res_df.columns
    assert res_df.iloc[0]["predicted_class"] == 0
    assert res_df.iloc[1]["predicted_class"] == 1


def test_hybrid_fallback_when_gnn_disabled(sample_legit_tx):
    """Test 100% XGBoost fallback when GNN is explicitly disabled."""
    predictor = HybridFraudPredictor(enable_gnn=False)
    res = predictor.predict_single(sample_legit_tx)

    assert res["scoring_mode"] == "xgboost_only"
    assert res["gnn_probability"] is None
    assert res["fraud_probability"] == res["xgb_probability"]


def test_hybrid_fallback_when_gnn_artifacts_missing(sample_legit_tx):
    """Test graceful fallback when GNN artifact paths do not exist."""
    predictor = HybridFraudPredictor(
        gnn_model_path=Path("/tmp/nonexistent_gnn_model.pt"),
        enable_gnn=True
    )
    assert predictor.gnn_available is False
    res = predictor.predict_single(sample_legit_tx)

    assert res["scoring_mode"] in ["xgboost_only", "xgboost_fallback"]
    assert res["fraud_probability"] == res["xgb_probability"]
    assert res["gnn_probability"] is None


def test_hybrid_runtime_gnn_failure_fallback(sample_legit_tx):
    """Test graceful fallback if GNN raises a runtime exception during prediction."""
    predictor = HybridFraudPredictor(enable_gnn=True)
    # Simulate runtime GNN failure
    if predictor.gnn_predictor is not None:
        mock_gnn = MagicMock()
        mock_gnn.predict_single.side_effect = RuntimeError("Simulated CUDA OOM or tensor error")
        predictor.gnn_predictor = mock_gnn

        res = predictor.predict_single(sample_legit_tx)
        assert res["scoring_mode"] == "xgboost_fallback"
        assert res["fraud_probability"] == res["xgb_probability"]
        assert res["gnn_probability"] is None


def test_hybrid_weight_boundary_math(sample_legit_tx):
    """Test exact mathematical blending at weights 1.0, 0.0, and 0.5."""
    predictor = HybridFraudPredictor(enable_gnn=True)
    if not predictor.gnn_available:
        pytest.skip("GNN artifacts unavailable for weight blending test")

    res_100_xgb = predictor.predict_single(sample_legit_tx, xgb_weight=1.0)
    assert res_100_xgb["fraud_probability"] == res_100_xgb["xgb_probability"]

    res_100_gnn = predictor.predict_single(sample_legit_tx, xgb_weight=0.0)
    assert res_100_gnn["fraud_probability"] == res_100_gnn["gnn_probability"]

    res_50_50 = predictor.predict_single(sample_legit_tx, xgb_weight=0.5)
    expected_avg = round((res_50_50["xgb_probability"] + res_50_50["gnn_probability"]) / 2.0, 6)
    assert abs(res_50_50["fraud_probability"] - expected_avg) < 1e-5


def test_hybrid_invalid_input_schema():
    """Test that input validation raises ValueError on missing features."""
    predictor = HybridFraudPredictor()
    with pytest.raises(ValueError):
        predictor.predict_single({"Time": 100.0, "Amount": 10.0})
