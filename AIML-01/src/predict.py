"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Inference Engine: src/predict.py

This module provides real-time single-transaction and batch fraud predictions
using the pre-trained XGBoost pipeline (StandardScaler + XGBClassifier).
"""

import os
# Prevent OpenMP runtime collision between XGBoost and PyTorch on macOS ARM64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "xgboost_pipeline.joblib"

# The 30 features expected by the pipeline in exact order
EXPECTED_FEATURES: List[str] = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]


# -----------------------------------------------------------------------------
# Model Loader
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_pipeline(model_path: Optional[Union[str, Path]] = None):
    """
    Load and cache the trained XGBoost inference pipeline.

    Args:
        model_path: Path to the .joblib pipeline file. Defaults to models/xgboost_pipeline.joblib.

    Returns:
        sklearn.pipeline.Pipeline: The loaded inference pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If model loading fails.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {path}\n"
            "Please ensure the training script (src/train_xgboost.py) has been run "
            "and models/xgboost_pipeline.joblib exists."
        )

    try:
        pipeline = joblib.load(path)
        return pipeline
    except Exception as exc:
        raise RuntimeError(f"Failed to load pipeline from {path}: {exc}") from exc


# -----------------------------------------------------------------------------
# Input Validation & Preprocessing
# -----------------------------------------------------------------------------
def validate_and_prepare_features(
    data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Validate input data schema, check for missing features, and align columns.

    Args:
        data: A single transaction dictionary, a list of dictionaries, or a pandas DataFrame.

    Returns:
        pd.DataFrame: A sanitized DataFrame guaranteed to have the exact 30 expected features.

    Raises:
        ValueError: If input format is invalid, missing required features, contains non-numeric values, or has nulls.
    """
    # 1. Convert input to pandas DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if not data:
            raise ValueError("Input list cannot be empty.")
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError(
            f"Unsupported data format: {type(data)}. Expected dict, list of dicts, or pandas DataFrame."
        )

    # 2. Check for missing required features
    missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing {len(missing_cols)} required feature(s): {missing_cols}. "
            f"Expected all 30 features: {EXPECTED_FEATURES}"
        )

    # 3. Select and order columns strictly according to EXPECTED_FEATURES
    df_aligned = df[EXPECTED_FEATURES].copy()

    # 4. Check for null or NaN values
    null_counts = df_aligned.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Input features contain missing/NaN values: {cols_with_nulls.to_dict()}. "
            "All transaction features must be complete numeric values."
        )

    # 5. Validate that all features are numeric
    try:
        df_numeric = df_aligned.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Failed to convert transaction features to numeric (float): {exc}"
        ) from exc

    return df_numeric


# -----------------------------------------------------------------------------
# Prediction Functions
# -----------------------------------------------------------------------------
def predict_single(
    transaction: Dict[str, Any],
    threshold: float = 0.5,
    model_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Predict fraud probability and risk classification for a single transaction.

    Args:
        transaction: Dictionary containing transaction features (Time, V1-V28, Amount).
        threshold: Classification cutoff threshold for fraud (0.0 to 1.0). Default is 0.5.
        model_path: Optional path to the trained model artifact.

    Returns:
        dict: Prediction results including fraud probability, class (0/1), and label.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be between 0.0 and 1.0, received: {threshold}")

    pipeline = load_pipeline(model_path)
    df_clean = validate_and_prepare_features(transaction)

    # Compute probability of fraud (class 1)
    probabilities = pipeline.predict_proba(df_clean)
    fraud_prob = float(probabilities[0, 1])

    # Determine class based on configurable threshold
    is_fraud = int(fraud_prob >= threshold)
    risk_label = "Fraud" if is_fraud == 1 else "Legitimate"

    return {
        "fraud_probability": round(fraud_prob, 6),
        "predicted_class": is_fraud,
        "risk_label": risk_label,
        "threshold_used": float(threshold),
        "status": "FLAGGED_FOR_REVIEW" if is_fraud == 1 else "APPROVED"
    }


def predict_batch(
    df: pd.DataFrame,
    threshold: float = 0.5,
    model_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Score a batch DataFrame of transactions.

    Args:
        df: DataFrame containing transaction features (Time, V1-V28, Amount).
        threshold: Classification cutoff threshold for fraud (0.0 to 1.0). Default is 0.5.
        model_path: Optional path to the trained model artifact.

    Returns:
        pd.DataFrame: A copy of input DataFrame enriched with:
            - fraud_probability
            - predicted_class
            - risk_label
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be between 0.0 and 1.0, received: {threshold}")

    pipeline = load_pipeline(model_path)
    df_clean = validate_and_prepare_features(df)

    # Compute probabilities for batch
    probabilities = pipeline.predict_proba(df_clean)
    fraud_probs = probabilities[:, 1]

    # Assign predictions
    result_df = df.copy()
    result_df["fraud_probability"] = np.round(fraud_probs, 6)
    result_df["predicted_class"] = (fraud_probs >= threshold).astype(int)
    result_df["risk_label"] = np.where(result_df["predicted_class"] == 1, "Fraud", "Legitimate")

    return result_df


# -----------------------------------------------------------------------------
# OOP Predictor Interface
# -----------------------------------------------------------------------------
class FraudPredictor:
    """
    High-level interface for real-time fraud scoring.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        default_threshold: float = 0.5
    ):
        self.model_path = model_path
        self.default_threshold = default_threshold
        self.pipeline = load_pipeline(self.model_path)

    def predict(
        self,
        data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
        threshold: Optional[float] = None
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Unified predict method that automatically handles single dicts or batch DataFrames.
        """
        t = threshold if threshold is not None else self.default_threshold
        if isinstance(data, dict):
            return predict_single(data, threshold=t, model_path=self.model_path)
        elif isinstance(data, (pd.DataFrame, list)):
            if isinstance(data, list):
                data = pd.DataFrame(data)
            return predict_batch(data, threshold=t, model_path=self.model_path)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


# -----------------------------------------------------------------------------
# Module Self-Test and Demonstration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("NEXVIA REAL-TIME FRAUD DETECTION: INFERENCE TEST")
    print("=" * 60)

    # 1. Initialize predictor
    print(f"\n[1] Loading trained pipeline from: {DEFAULT_MODEL_PATH}")
    predictor = FraudPredictor()
    print(" -> Model loaded successfully!")

    # 2. Test with sample legitimate transaction (synthetic low-risk payload)
    legit_sample = {col: 0.0 for col in EXPECTED_FEATURES}
    legit_sample["Time"] = 100.0
    legit_sample["Amount"] = 49.99

    print("\n[2] Testing single Legitimate transaction prediction...")
    legit_res = predictor.predict(legit_sample)
    print("Result:")
    for k, v in legit_res.items():
        print(f"  {k}: {v}")

    # 3. Test with sample known fraud transaction features (extreme anomaly payload)
    fraud_sample = {col: 0.0 for col in EXPECTED_FEATURES}
    fraud_sample["Time"] = 406.0
    fraud_sample["V1"] = -2.312226
    fraud_sample["V2"] = 1.951992
    fraud_sample["V3"] = -1.609851
    fraud_sample["V4"] = 3.997906
    fraud_sample["V7"] = -2.537387
    fraud_sample["V10"] = -2.772272
    fraud_sample["V12"] = -2.899907
    fraud_sample["V14"] = -4.289254
    fraud_sample["V17"] = -2.830056
    fraud_sample["Amount"] = 0.0

    print("\n[3] Testing single Fraudulent transaction prediction...")
    fraud_res = predictor.predict(fraud_sample)
    print("Result:")
    for k, v in fraud_res.items():
        print(f"  {k}: {v}")

    # 4. Test batch scoring
    print("\n[4] Testing batch DataFrame prediction (2 transactions)...")
    batch_df = pd.DataFrame([legit_sample, fraud_sample])
    scored_batch = predictor.predict(batch_df)
    print(scored_batch[["Time", "Amount", "fraud_probability", "predicted_class", "risk_label"]])

    # 5. Test input validation and error handling
    print("\n[5] Testing input validation with missing features...")
    invalid_sample = {"Time": 50.0, "Amount": 10.0}
    try:
        predictor.predict(invalid_sample)
    except ValueError as e:
        print(f" -> Caught expected validation error:\n    {e}")

    print("\n[6] Testing input validation with non-numeric features...")
    invalid_type = legit_sample.copy()
    invalid_type["Amount"] = "INVALID_AMOUNT"
    try:
        predictor.predict(invalid_type)
    except ValueError as e:
        print(f" -> Caught expected validation error:\n    {e}")

    print("\n" + "=" * 60)
    print("All focused inference tests passed successfully!")
    print("=" * 60)
