"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Main Application Entry Point: main.py

This script serves as the primary command-line interface and demonstration runner
for the Nexvia Financial Fraud Detection platform. It integrates with src.predict
to showcase real-time single-transaction scoring, batch processing, and input validation
using the pre-trained production XGBoost pipeline without retraining.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional

import pandas as pd

# Ensure project root is in sys.path for clean src imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    DEFAULT_MODEL_PATH,
    EXPECTED_FEATURES,
    FraudPredictor,
    load_pipeline,
    predict_batch,
    predict_single
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def print_banner() -> None:
    """Print the platform title banner."""
    print("=" * 72)
    print("  NEXVIA AI/ML INTERNSHIP: FINANCIAL FRAUD DETECTION PLATFORM (AIML-01)  ")
    print("  Real-Time Inference & Transaction Scoring Engine                        ")
    print("=" * 72)


def check_prerequisites() -> bool:
    """Verify that required model artifacts exist before execution."""
    if not DEFAULT_MODEL_PATH.exists():
        print(f"\n[ERROR] Model artifact missing: {DEFAULT_MODEL_PATH}")
        print("Please train the model first by running: python src/train_xgboost.py\n")
        return False
    return True


def run_single_transaction_demo(predictor: FraudPredictor, threshold: float = 0.5) -> None:
    """
    Demonstrate real-time scoring of individual transactions (Legitimate and Fraud).
    """
    print("\n" + "-" * 72)
    print("1. REAL-TIME SINGLE TRANSACTION SCORING DEMONSTRATION")
    print("-" * 72)

    legit_tx: Optional[Dict] = None
    fraud_tx: Optional[Dict] = None

    # Load real transactions from raw dataset if available
    if DATA_PATH.exists():
        try:
            sample_df = pd.read_csv(DATA_PATH, nrows=1000)
            legit_rows = sample_df[sample_df["Class"] == 0]
            fraud_rows = sample_df[sample_df["Class"] == 1]

            if not legit_rows.empty:
                legit_tx = legit_rows.iloc[0].drop("Class").to_dict()
            if not fraud_rows.empty:
                fraud_tx = fraud_rows.iloc[0].drop("Class").to_dict()
        except Exception as e:
            print(f"[Warning] Could not load sample from {DATA_PATH}: {e}")

    # Fallback to realistic synthetic transactions if dataset not available
    if legit_tx is None:
        legit_tx = {col: 0.0 for col in EXPECTED_FEATURES}
        legit_tx["Time"] = 120.0
        legit_tx["Amount"] = 35.50

    if fraud_tx is None:
        fraud_tx = {col: 0.0 for col in EXPECTED_FEATURES}
        fraud_tx["Time"] = 406.0
        fraud_tx["V1"] = -2.3122
        fraud_tx["V2"] = 1.9520
        fraud_tx["V3"] = -1.6099
        fraud_tx["V4"] = 3.9979
        fraud_tx["V14"] = -4.2893
        fraud_tx["Amount"] = 0.0

    # Score Legitimate Transaction
    start_time = time.perf_counter()
    legit_res = predictor.predict(legit_tx, threshold=threshold)
    legit_latency_ms = (time.perf_counter() - start_time) * 1000

    print("\n[Case A] Incoming Normal / Legitimate Transaction:")
    print(f"  Transaction Amount:   ${legit_tx.get('Amount', 0.0):.2f}")
    print(f"  Fraud Probability:    {legit_res['fraud_probability']:.6f} ({legit_res['fraud_probability']*100:.2f}%)")
    print(f"  Decision Threshold:   {legit_res['threshold_used']:.2f}")
    print(f"  Predicted Class:      {legit_res['predicted_class']} ({legit_res['risk_label']})")
    print(f"  System Action:        {legit_res['status']}")
    print(f"  Inference Latency:    {legit_latency_ms:.2f} ms")

    # Score Fraudulent Transaction
    start_time = time.perf_counter()
    fraud_res = predictor.predict(fraud_tx, threshold=threshold)
    fraud_latency_ms = (time.perf_counter() - start_time) * 1000

    print("\n[Case B] Incoming Suspicious / Fraudulent Transaction:")
    print(f"  Transaction Amount:   ${fraud_tx.get('Amount', 0.0):.2f}")
    print(f"  Fraud Probability:    {fraud_res['fraud_probability']:.6f} ({fraud_res['fraud_probability']*100:.2f}%)")
    print(f"  Decision Threshold:   {fraud_res['threshold_used']:.2f}")
    print(f"  Predicted Class:      {fraud_res['predicted_class']} ({fraud_res['risk_label']})")
    print(f"  System Action:        {fraud_res['status']}")
    print(f"  Inference Latency:    {fraud_latency_ms:.2f} ms")


def run_batch_demo(predictor: FraudPredictor, threshold: float = 0.5, batch_size: int = 10) -> None:
    """
    Demonstrate scoring a batch of transactions.
    """
    print("\n" + "-" * 72)
    print(f"2. BATCH TRANSACTION SCORING DEMONSTRATION (Sample: {batch_size} transactions)")
    print("-" * 72)

    batch_df: Optional[pd.DataFrame] = None

    if DATA_PATH.exists():
        try:
            # Select mixed legitimate and fraud transactions for an insightful demonstration
            df_preview = pd.read_csv(DATA_PATH, nrows=1000)
            fraud_rows = df_preview[df_preview["Class"] == 1]
            legit_rows = df_preview[df_preview["Class"] == 0].iloc[: batch_size - len(fraud_rows)]
            batch_df = pd.concat([legit_rows, fraud_rows]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            true_labels = batch_df["Class"].copy()
            batch_df = batch_df.drop("Class", axis=1)
        except Exception as e:
            print(f"[Warning] Could not load batch from {DATA_PATH}: {e}")

    if batch_df is None:
        # Synthetic batch
        records = []
        for i in range(batch_size):
            row = {col: 0.0 for col in EXPECTED_FEATURES}
            row["Time"] = float(i * 10)
            row["Amount"] = float(10.0 + i * 5.0)
            records.append(row)
        batch_df = pd.DataFrame(records)
        true_labels = None

    start_time = time.perf_counter()
    scored_df = predictor.predict(batch_df, threshold=threshold)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    display_cols = ["Time", "Amount", "fraud_probability", "predicted_class", "risk_label"]
    print("\nScored Batch Preview:")
    print(scored_df[display_cols].to_string(index=True))

    fraud_count = int((scored_df["predicted_class"] == 1).sum())
    legit_count = int((scored_df["predicted_class"] == 0).sum())
    total_count = len(scored_df)

    print("\nBatch Summary Statistics:")
    print(f"  Total Processed:      {total_count}")
    print(f"  Approved (Legit):     {legit_count} ({legit_count/total_count*100:.1f}%)")
    print(f"  Flagged (Fraud):      {fraud_count} ({fraud_count/total_count*100:.1f}%)")
    print(f"  Batch Scoring Time:   {elapsed_ms:.2f} ms ({elapsed_ms/total_count:.2f} ms/tx)")


def run_validation_demo(predictor: FraudPredictor) -> None:
    """
    Demonstrate robust error handling and schema validation.
    """
    print("\n" + "-" * 72)
    print("3. SCHEMA VALIDATION AND ERROR HANDLING DEMONSTRATION")
    print("-" * 72)

    # Test 1: Missing features
    print("\n[Test 1] Supplying payload with missing features:")
    bad_payload = {"Time": 10.0, "Amount": 100.0}
    try:
        predictor.predict(bad_payload)
    except ValueError as exc:
        print(f"  -> Successfully caught ValueError:\n     {exc}")

    # Test 2: Non-numeric invalid value
    print("\n[Test 2] Supplying non-numeric feature value:")
    bad_type = {col: 0.0 for col in EXPECTED_FEATURES}
    bad_type["Amount"] = "FIFTY_DOLLARS"
    try:
        predictor.predict(bad_type)
    except ValueError as exc:
        print(f"  -> Successfully caught ValueError:\n     {exc}")


def score_custom_csv(
    predictor: FraudPredictor,
    input_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0.5
) -> None:
    """
    Score a custom CSV file provided via CLI arguments.
    """
    in_file = Path(input_path)
    if not in_file.exists():
        print(f"[ERROR] Input file does not exist: {in_file}")
        sys.exit(1)

    print(f"\nLoading transactions from: {in_file}")
    df = pd.read_csv(in_file)
    print(f"Loaded {len(df)} transactions.")

    # Drop target column if present in evaluation file
    eval_df = df.drop("Class", axis=1) if "Class" in df.columns else df

    scored_df = predictor.predict(eval_df, threshold=threshold)

    fraud_count = int((scored_df["predicted_class"] == 1).sum())
    print("\nScoring Completed:")
    print(f"  Total transactions: {len(scored_df)}")
    print(f"  Flagged as Fraud:   {fraud_count}")
    print(f"  Approved as Legit:  {len(scored_df) - fraud_count}")

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(out_file, index=False)
        print(f"Saved scored results to: {out_file}")
    else:
        print("\nFirst 5 scored transactions:")
        preview_cols = ["Time", "Amount", "fraud_probability", "predicted_class", "risk_label"]
        print(scored_df[preview_cols].head().to_string(index=True))


def main() -> None:
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Nexvia Real-Time Financial Fraud Detection Platform (AIML-01)"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "batch"],
        default="demo",
        help="Execution mode: 'demo' for full platform showcase, 'batch' to score a CSV file (default: demo)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification decision threshold for fraud (0.0 to 1.0, default: 0.5)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to input CSV file for batch scoring"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save scored CSV output in batch mode"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of transactions to display in batch demo (default: 10)"
    )

    args = parser.parse_args()

    # Validate threshold argument
    if not (0.0 <= args.threshold <= 1.0):
        print(f"[ERROR] Threshold must be between 0.0 and 1.0. Received: {args.threshold}")
        sys.exit(1)

    print_banner()

    # Prerequisite verification
    if not check_prerequisites():
        sys.exit(1)

    # Initialize predictor
    print(f"\n[INFO] Loading trained XGBoost pipeline from: {DEFAULT_MODEL_PATH}")
    predictor = FraudPredictor(default_threshold=args.threshold)
    print("[INFO] Model loaded and ready for real-time inference.")

    if args.mode == "demo":
        run_single_transaction_demo(predictor, threshold=args.threshold)
        run_batch_demo(predictor, threshold=args.threshold, batch_size=args.batch_size)
        run_validation_demo(predictor)

        print("\n" + "=" * 72)
        print("  DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 72 + "\n")

    elif args.mode == "batch":
        if not args.input_file:
            print("[ERROR] Please provide --input-file when running in batch mode.")
            sys.exit(1)
        score_custom_csv(predictor, args.input_file, args.output_file, threshold=args.threshold)


if __name__ == "__main__":
    main()
