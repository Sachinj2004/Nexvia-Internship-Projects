"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Hybrid Fraud Inference Engine: src/hybrid.py

Combines XGBoost tabular decision trees with PyTorch Geometric GraphSAGE
relational embeddings with configurable weighting and defensive fallback to 100% XGBoost.
"""

import os
# Prevent OpenMP runtime collision between XGBoost and PyTorch on macOS ARM64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.predict import (
    DEFAULT_MODEL_PATH,
    EXPECTED_FEATURES,
    FraudPredictor,
    validate_and_prepare_features,
)
from src.gnn import (
    DEFAULT_ANCHORS_PATH,
    DEFAULT_GNN_MODEL_PATH,
    DEFAULT_GNN_SCALER_PATH,
    GNNFraudPredictor,
)

logger = logging.getLogger("nexvia.hybrid")


class HybridFraudPredictor:
    """
    Hybrid Fraud Inference Engine combining XGBoost tabular scoring with
    PyTorch Geometric GraphSAGE relational scoring.

    Formula:
        P_hybrid = alpha * P_xgb + (1 - alpha) * P_gnn
        where alpha = xgb_weight (default: 0.85)

    Defensive Fallback:
        If GNN artifacts are missing, disabled (enable_gnn=False), or encounter an error,
        the predictor safely falls back 100% to the verified XGBoost model without failing.

    Limitation Note:
        The underlying credit card dataset lacks real entity identifiers (card numbers,
        device fingerprints, merchant IDs). GraphSAGE operates over a feature-space
        k-NN similarity graph, capturing metric-space clusters rather than real-world
        transaction syndicates or multi-account fraud rings.
    """

    def __init__(
        self,
        xgb_model_path: Optional[Union[str, Path]] = None,
        gnn_model_path: Optional[Union[str, Path]] = None,
        gnn_scaler_path: Optional[Union[str, Path]] = None,
        gnn_anchors_path: Optional[Union[str, Path]] = None,
        default_threshold: float = 0.50,
        default_xgb_weight: float = 0.85,
        enable_gnn: bool = True,
    ):
        self.default_threshold = float(default_threshold)
        self.default_xgb_weight = float(default_xgb_weight)
        self.enable_gnn = enable_gnn

        # 1. Primary XGBoost predictor (always required)
        self.xgb_predictor = FraudPredictor(
            model_path=xgb_model_path or DEFAULT_MODEL_PATH,
            default_threshold=self.default_threshold,
        )

        # 2. PyG GraphSAGE predictor (optional with graceful fallback)
        self.gnn_predictor: Optional[GNNFraudPredictor] = None
        self.gnn_available = False

        if self.enable_gnn:
            try:
                g_model = Path(gnn_model_path or DEFAULT_GNN_MODEL_PATH)
                g_scaler = Path(gnn_scaler_path or DEFAULT_GNN_SCALER_PATH)
                g_anchors = Path(gnn_anchors_path or DEFAULT_ANCHORS_PATH)

                if g_model.exists() and g_scaler.exists():
                    self.gnn_predictor = GNNFraudPredictor(
                        model_path=g_model,
                        scaler_path=g_scaler,
                        anchors_path=g_anchors,
                        default_threshold=self.default_threshold,
                    )
                    self.gnn_available = True
                else:
                    logger.warning("GNN model or scaler missing. Operating in XGBoost fallback mode.")
            except Exception as init_err:
                logger.warning(f"Failed to initialize GNN predictor: {init_err}. Falling back to XGBoost.")
                self.gnn_predictor = None
                self.gnn_available = False

    def predict_single(
        self,
        transaction: Dict[str, Any],
        threshold: Optional[float] = None,
        xgb_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute combined fraud score for an individual transaction dictionary.
        """
        t = threshold if threshold is not None else self.default_threshold
        w = xgb_weight if xgb_weight is not None else self.default_xgb_weight
        w = max(0.0, min(1.0, float(w)))

        # Validate features
        validate_and_prepare_features(transaction)

        # 1. XGBoost score
        xgb_res = self.xgb_predictor.predict(transaction, threshold=t)
        p_xgb = float(xgb_res["fraud_probability"])

        # 2. GNN score (with defensive error boundary)
        p_gnn: Optional[float] = None
        mode = "xgboost_only"

        if self.enable_gnn and self.gnn_available and self.gnn_predictor is not None:
            try:
                gnn_res = self.gnn_predictor.predict_single(transaction, threshold=t)
                p_gnn = float(gnn_res["fraud_probability"])
                mode = "hybrid"
            except Exception as gnn_err:
                logger.warning(f"GNN inference error: {gnn_err}. Reverting to XGBoost.")
                p_gnn = None
                mode = "xgboost_fallback"

        if p_gnn is not None and mode == "hybrid":
            p_final = (w * p_xgb) + ((1.0 - w) * p_gnn)
        else:
            p_final = p_xgb

        is_fraud = int(p_final >= t)
        risk_label = "Fraud" if is_fraud == 1 else "Legitimate"

        return {
            "fraud_probability": round(p_final, 6),
            "xgb_probability": round(p_xgb, 6),
            "gnn_probability": round(p_gnn, 6) if p_gnn is not None else None,
            "predicted_class": is_fraud,
            "risk_label": risk_label,
            "threshold_used": float(t),
            "status": "FLAGGED_FOR_REVIEW" if is_fraud == 1 else "APPROVED",
            "scoring_mode": mode,
            "xgb_weight": round(w, 2),
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        xgb_weight: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute combined fraud scores for a batch DataFrame of transactions.
        """
        t = threshold if threshold is not None else self.default_threshold
        w = xgb_weight if xgb_weight is not None else self.default_xgb_weight
        w = max(0.0, min(1.0, float(w)))

        validate_and_prepare_features(df)

        # 1. XGBoost batch prediction
        result_df = self.xgb_predictor.predict(df, threshold=t)
        p_xgb = result_df["fraud_probability"].values

        # 2. GNN batch prediction
        p_gnn = None
        mode = "xgboost_only"

        if self.enable_gnn and self.gnn_available and self.gnn_predictor is not None:
            try:
                gnn_df = self.gnn_predictor.predict_batch(df, threshold=t)
                p_gnn = gnn_df["fraud_probability"].values
                mode = "hybrid"
            except Exception as gnn_err:
                logger.warning(f"GNN batch inference error: {gnn_err}. Reverting to XGBoost.")
                p_gnn = None
                mode = "xgboost_fallback"

        if p_gnn is not None and mode == "hybrid":
            p_final = (w * p_xgb) + ((1.0 - w) * p_gnn)
        else:
            p_final = p_xgb

        result_df["xgb_probability"] = np.round(p_xgb, 6)
        result_df["gnn_probability"] = np.round(p_gnn, 6) if p_gnn is not None else None
        result_df["fraud_probability"] = np.round(p_final, 6)
        result_df["predicted_class"] = (p_final >= t).astype(int)
        result_df["risk_label"] = np.where(result_df["predicted_class"] == 1, "Fraud", "Legitimate")
        result_df["threshold_used"] = float(t)
        result_df["status"] = np.where(result_df["predicted_class"] == 1, "FLAGGED_FOR_REVIEW", "APPROVED")
        result_df["scoring_mode"] = mode
        result_df["xgb_weight"] = round(w, 2)

        return result_df

    def predict(
        self,
        data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
        threshold: Optional[float] = None,
        xgb_weight: Optional[float] = None,
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Unified prediction dispatcher handling single dicts, batch DataFrames, or lists of dicts.
        """
        if isinstance(data, dict):
            return self.predict_single(data, threshold=threshold, xgb_weight=xgb_weight)
        elif isinstance(data, (pd.DataFrame, list)):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            return self.predict_batch(data, threshold=threshold, xgb_weight=xgb_weight)
        else:
            raise ValueError(f"Unsupported input data format: {type(data)}")
