"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Controlled Model Retraining Controller: src/retrain.py

Implements automated drift-triggered retraining safeguards:
1. Cooldown enforcement (prevents retraining loops).
2. Drift decision gating (drift share and drifted column thresholds).
3. Candidate model evaluation against production baseline PR-AUC.
4. Model backup, controlled promotion, and rollback capabilities.
5. MLflow experiment and artifact logging.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger("nexvia.retrain")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_pipeline.joblib"
DEFAULT_BACKUP_PATH = PROJECT_ROOT / "models" / "xgboost_pipeline_backup.joblib"
DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "models" / "candidate_pipeline.joblib"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "outputs" / "retraining_history.json"
DEFAULT_MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

EXPECTED_FEATURES: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


class RetrainingController:
    """
    Manages controlled candidate retraining, candidate validation,
    safeguards, and promotion for the fraud detection system.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        backup_path: Optional[Path] = None,
        candidate_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
        drift_share_threshold: float = 0.30,
        drifted_columns_threshold: int = 8,
        min_pr_auc_threshold: float = 0.80,
        min_performance_ratio: float = 0.95,
        cooldown_seconds: float = 3600.0,
        mlflow_tracking_uri: Optional[str] = None
    ):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.backup_path = Path(backup_path or DEFAULT_BACKUP_PATH)
        self.candidate_path = Path(candidate_path or DEFAULT_CANDIDATE_PATH)
        self.history_path = Path(history_path or DEFAULT_HISTORY_PATH)
        self.drift_share_threshold = float(drift_share_threshold)
        self.drifted_columns_threshold = int(drifted_columns_threshold)
        self.min_pr_auc_threshold = float(min_pr_auc_threshold)
        self.min_performance_ratio = float(min_performance_ratio)
        self.cooldown_seconds = float(cooldown_seconds)
        self.mlflow_tracking_uri = mlflow_tracking_uri or f"sqlite:///{DEFAULT_MLFLOW_DB}"

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def load_history(self) -> Dict[str, Any]:
        """Load retraining history and audit records."""
        if not self.history_path.exists():
            return {
                "last_retrained_at": None,
                "last_retrained_timestamp": 0.0,
                "total_runs": 0,
                "promoted_runs": 0,
                "rejected_runs": 0,
                "history": []
            }
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read retraining history: {exc}")
            return {
                "last_retrained_at": None,
                "last_retrained_timestamp": 0.0,
                "total_runs": 0,
                "promoted_runs": 0,
                "rejected_runs": 0,
                "history": []
            }

    def _save_history(self, history_data: Dict[str, Any]) -> None:
        """Persist retraining history to disk."""
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=4)

    def can_retrain(self, current_time: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check whether retraining is permitted or currently blocked by cooldown.
        """
        now = current_time if current_time is not None else time.time()
        history = self.load_history()
        last_ts = history.get("last_retrained_timestamp", 0.0) or 0.0
        elapsed = now - last_ts

        if elapsed < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - elapsed)
            return False, f"Retraining on cooldown. Remaining: {remaining}s (cooldown: {int(self.cooldown_seconds)}s)."

        return True, "Retraining permitted."

    def should_trigger(self, drift_metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate drift report metrics against configured thresholds to decide
        if candidate retraining should be initiated.
        """
        if not isinstance(drift_metrics, dict):
            return False, "Invalid drift metrics provided."

        drift_share = float(drift_metrics.get("drift_share", 0.0))
        drifted_cols = int(drift_metrics.get("number_of_drifted_columns", 0))
        dataset_drift = bool(drift_metrics.get("dataset_drift", False))

        reasons = []
        if dataset_drift:
            reasons.append("Dataset drift flag is True")
        if drift_share >= self.drift_share_threshold:
            reasons.append(f"Drift share {drift_share:.1%} exceeds threshold {self.drift_share_threshold:.1%}")
        if drifted_cols >= self.drifted_columns_threshold:
            reasons.append(f"Drifted columns ({drifted_cols}) exceed threshold ({self.drifted_columns_threshold})")

        if reasons:
            return True, "; ".join(reasons)

        return False, f"Drift below threshold (share={drift_share:.1%}, columns={drifted_cols})."

    def evaluate_model(
        self,
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Compute standardized performance metrics on a validation set."""
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        pr_auc = float(average_precision_score(y_test, y_prob))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        return {
            "pr_auc": round(pr_auc, 6),
            "roc_auc": round(roc_auc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6)
        }

    def train_candidate(
        self,
        data_path: Optional[Path] = None,
        sample_rows: Optional[int] = None,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42
    ) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame, pd.Series]:
        """
        Train a new candidate XGBoost pipeline safely in memory.
        """
        target_path = Path(data_path or DEFAULT_DATA_PATH)
        if not target_path.exists():
            raise FileNotFoundError(f"Training data not found at {target_path}")

        if sample_rows is not None:
            df = pd.read_csv(target_path, nrows=sample_rows)
        else:
            df = pd.read_csv(target_path)

        if "Class" not in df.columns:
            raise ValueError("Dataset missing target 'Class' column.")

        X = df[EXPECTED_FEATURES]
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = float(neg / max(pos, 1))

        scaler = StandardScaler()
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_state,
            tree_method="hist",
            n_jobs=2
        )

        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        metrics = self.evaluate_model(pipeline, X_test, y_test)

        return pipeline, metrics, X_test, y_test

    def run_retraining(
        self,
        data_path: Optional[Path] = None,
        sample_rows: Optional[int] = None,
        auto_promote: bool = False,
        force: bool = False,
        trigger_reason: str = "Manual / Scheduled"
    ) -> Dict[str, Any]:
        """
        Execute controlled retraining cycle:
        1. Check cooldown.
        2. Train candidate pipeline.
        3. Validate candidate against baseline on identical validation split.
        4. Promote if criteria are met and auto_promote is enabled.
        5. Log to MLflow and record history.
        """
        now = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()

        if not force:
            can_run, reason = self.can_retrain(current_time=now)
            if not can_run:
                return {
                    "status": "cooldown_blocked",
                    "message": reason,
                    "timestamp": now_iso
                }

        # Step 1: Train candidate
        try:
            candidate_pipe, cand_metrics, X_test, y_test = self.train_candidate(
                data_path=data_path,
                sample_rows=sample_rows
            )
        except Exception as exc:
            logger.error(f"Candidate training failed: {exc}")
            return {
                "status": "training_failed",
                "error": str(exc),
                "timestamp": now_iso
            }

        # Step 2: Evaluate baseline if exists
        baseline_metrics: Optional[Dict[str, float]] = None
        if self.model_path.exists():
            try:
                base_pipe = joblib.load(self.model_path)
                baseline_metrics = self.evaluate_model(base_pipe, X_test, y_test)
            except Exception as base_err:
                logger.warning(f"Could not evaluate baseline model: {base_err}")

        # Step 3: Assess promotion criteria
        cand_pr_auc = cand_metrics["pr_auc"]
        base_pr_auc = baseline_metrics["pr_auc"] if baseline_metrics else 0.0

        threshold_met = cand_pr_auc >= self.min_pr_auc_threshold
        comparative_met = (cand_pr_auc >= (base_pr_auc * self.min_performance_ratio)) if base_pr_auc > 0 else True

        promotable = threshold_met and comparative_met
        promoted = False

        # Save candidate model artifact
        self.candidate_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(candidate_pipe, self.candidate_path)

        if promotable and auto_promote:
            # Backup active model
            if self.model_path.exists():
                shutil.copy2(self.model_path, self.backup_path)
            # Promote candidate
            joblib.dump(candidate_pipe, self.model_path)
            promoted = True

        # Step 4: MLflow Logging
        mlflow_run_id = None
        try:
            import mlflow
            os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("fraud-detection-retrain")
            with mlflow.start_run(run_name=f"retrain_{int(now)}") as run:
                mlflow_run_id = run.info.run_id
                mlflow.log_params({
                    "trigger_reason": trigger_reason,
                    "auto_promote": auto_promote,
                    "promoted": promoted,
                    "sample_rows": sample_rows or "full"
                })
                for k, v in cand_metrics.items():
                    mlflow.log_metric(f"candidate_{k}", v)
                if baseline_metrics:
                    for k, v in baseline_metrics.items():
                        mlflow.log_metric(f"baseline_{k}", v)
        except Exception as ml_err:
            logger.warning(f"MLflow logging error: {ml_err}")

        # Step 5: Update Retraining History
        history = self.load_history()
        history["last_retrained_at"] = now_iso
        history["last_retrained_timestamp"] = now
        history["total_runs"] += 1
        if promoted:
            history["promoted_runs"] += 1
        elif not promotable:
            history["rejected_runs"] += 1

        run_summary = {
            "timestamp": now_iso,
            "trigger_reason": trigger_reason,
            "candidate_metrics": cand_metrics,
            "baseline_metrics": baseline_metrics,
            "promotable": promotable,
            "promoted": promoted,
            "mlflow_run_id": mlflow_run_id,
            "candidate_path": str(self.candidate_path)
        }
        history["history"].append(run_summary)
        self._save_history(history)

        return {
            "status": "promoted" if promoted else ("candidate_ready" if promotable else "rejected"),
            "candidate_metrics": cand_metrics,
            "baseline_metrics": baseline_metrics,
            "promotable": promotable,
            "promoted": promoted,
            "candidate_path": str(self.candidate_path),
            "mlflow_run_id": mlflow_run_id,
            "timestamp": now_iso
        }

    def rollback(self) -> Dict[str, Any]:
        """
        Roll back production model to previous backup if available.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        if not self.backup_path.exists():
            return {
                "status": "failed",
                "message": f"Backup model artifact not found at {self.backup_path}",
                "timestamp": now_iso
            }

        shutil.copy2(self.backup_path, self.model_path)
        return {
            "status": "success",
            "message": f"Successfully rolled back {self.model_path} from {self.backup_path}",
            "timestamp": now_iso
        }


# Singleton retraining controller instance
retraining_controller = RetrainingController()
