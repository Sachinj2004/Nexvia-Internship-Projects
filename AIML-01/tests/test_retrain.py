"""
Tests for Controlled Model Retraining Controller: tests/test_retrain.py
"""

from pathlib import Path
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrain import EXPECTED_FEATURES, RetrainingController


@pytest.fixture
def temp_retrain_env():
    """Create isolated temporary environment for retraining controller testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_path = tmp_path / "model.joblib"
        backup_path = tmp_path / "backup.joblib"
        candidate_path = tmp_path / "candidate.joblib"
        history_path = tmp_path / "history.json"
        mlflow_db = f"sqlite:///{tmp_path / 'test_mlflow.db'}"

        # Create a mock baseline model
        scaler = StandardScaler()
        clf = xgb.XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
        pipe = Pipeline([("scaler", scaler), ("model", clf)])

        # Generate synthetic dataset where V1 is predictive of Class
        np.random.seed(42)
        n_samples = 300
        data = {feat: np.random.randn(n_samples) for feat in EXPECTED_FEATURES}
        # Strong signal so candidate easily learns and achieves high PR-AUC
        data["Class"] = (data["V1"] > 0.5).astype(int)
        df = pd.DataFrame(data)
        pipe.fit(df[EXPECTED_FEATURES], df["Class"])
        pipe.tag = "ORIGINAL_BASELINE"
        joblib.dump(pipe, model_path)

        data_csv_path = tmp_path / "creditcard.csv"
        df.to_csv(data_csv_path, index=False)

        controller = RetrainingController(
            model_path=model_path,
            backup_path=backup_path,
            candidate_path=candidate_path,
            history_path=history_path,
            drift_share_threshold=0.30,
            drifted_columns_threshold=8,
            min_pr_auc_threshold=0.20,
            min_performance_ratio=0.50,
            cooldown_seconds=10.0,
            mlflow_tracking_uri=mlflow_db
        )

        yield {
            "controller": controller,
            "data_csv_path": data_csv_path,
            "model_path": model_path,
            "backup_path": backup_path,
            "candidate_path": candidate_path,
            "history_path": history_path
        }


def test_cooldown_enforcement(temp_retrain_env):
    """Test that retraining is blocked when within the cooldown window."""
    env = temp_retrain_env
    controller = env["controller"]

    # Initial check: should be allowed (last_retrained = 0)
    allowed, msg = controller.can_retrain()
    assert allowed is True

    # Record a recent run
    now = time.time()
    controller._save_history({
        "last_retrained_at": "2026-10-30T00:00:00Z",
        "last_retrained_timestamp": now,
        "total_runs": 1,
        "promoted_runs": 0,
        "rejected_runs": 0,
        "history": []
    })

    # Should now be blocked
    allowed, msg = controller.can_retrain(current_time=now + 2.0)
    assert allowed is False
    assert "Remaining: 8s" in msg

    # After cooldown elapsed
    allowed, msg = controller.can_retrain(current_time=now + 12.0)
    assert allowed is True


def test_drift_decision_gating(temp_retrain_env):
    """Test should_trigger evaluates drift metrics correctly."""
    controller = temp_retrain_env["controller"]

    # No drift
    stable_metrics = {
        "dataset_drift": False,
        "drift_share": 0.05,
        "number_of_drifted_columns": 1
    }
    should_run, reason = controller.should_trigger(stable_metrics)
    assert should_run is False

    # High drift share
    drifted_share_metrics = {
        "dataset_drift": False,
        "drift_share": 0.45,
        "number_of_drifted_columns": 2
    }
    should_run, reason = controller.should_trigger(drifted_share_metrics)
    assert should_run is True
    assert "Drift share" in reason

    # High drifted column count
    drifted_cols_metrics = {
        "dataset_drift": False,
        "drift_share": 0.20,
        "number_of_drifted_columns": 10
    }
    should_run, reason = controller.should_trigger(drifted_cols_metrics)
    assert should_run is True
    assert "Drifted columns (10)" in reason


def test_candidate_training_without_autopromote(temp_retrain_env):
    """Test candidate training generates candidate artifact without overwriting active model."""
    env = temp_retrain_env
    controller = env["controller"]
    model_mtime_before = env["model_path"].stat().st_mtime

    result = controller.run_retraining(
        data_path=env["data_csv_path"],
        auto_promote=False,
        force=True,
        trigger_reason="Unit test"
    )

    assert result["promotable"] is True
    assert result["promoted"] is False
    assert env["candidate_path"].exists()
    assert env["backup_path"].exists() is False
    assert env["model_path"].stat().st_mtime == model_mtime_before


def test_candidate_training_with_autopromote_and_rollback(temp_retrain_env):
    """Test candidate training with auto_promote creates backup, updates model, and rollback works."""
    env = temp_retrain_env
    controller = env["controller"]

    result = controller.run_retraining(
        data_path=env["data_csv_path"],
        auto_promote=True,
        force=True,
        trigger_reason="Unit test auto-promote"
    )

    assert result["promotable"] is True
    assert result["promoted"] is True
    assert env["backup_path"].exists()

    # Verify backup contains original baseline with tag
    backup_pipe = joblib.load(env["backup_path"])
    assert getattr(backup_pipe, "tag", None) == "ORIGINAL_BASELINE"

    # Now execute rollback
    rollback_res = controller.rollback()
    assert rollback_res["status"] == "success"

    restored_pipe = joblib.load(env["model_path"])
    assert getattr(restored_pipe, "tag", None) == "ORIGINAL_BASELINE"

