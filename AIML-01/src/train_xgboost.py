import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import xgboost as xgb
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Project paths setup
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "creditcard.csv"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Step 2: Load the raw dataset
print("Loading credit card dataset...")
df = pd.read_csv(DATA_PATH)

# Step 3: Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print(f"Total transactions: {len(df)}")
print(f"Features: {X.shape[1]}")

# Step 4: Stratified Train/Test Split (identical to baseline for fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training transactions: {len(X_train)} (Fraud: {y_train.sum()})")
print(f"Testing transactions:  {len(X_test)} (Fraud: {y_test.sum()})")

# Step 5: Feature scaling (preventing leakage by fitting only on train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Calculate class imbalance scale_pos_weight
neg_count = int(len(y_train) - y_train.sum())
pos_count = int(y_train.sum())
scale_pos_weight = neg_count / pos_count
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f} (Negative: {neg_count}, Positive: {pos_count})")

# Step 7: Configure XGBoost Classifier
xgb_params = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "aucpr",
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": 4
}

model = xgb.XGBClassifier(**xgb_params)

# Step 8: Configure MLflow local tracking
MLFLOW_DB_PATH = PROJECT_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("fraud-detection")

print("\nStarting MLflow tracked run: xgboost_classifier...")
with mlflow.start_run(run_name="xgboost_classifier") as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")

    # Log parameters
    mlflow.log_params(xgb_params)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("model_type", "XGBClassifier")

    # Step 9: Train the model
    print("\nTraining XGBoost model...")
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50
    )

    # Step 10: Predictions and Probabilities
    y_pred = model.predict(X_test_scaled)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]

    # Step 11: Calculate Evaluation Metrics
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = float(average_precision_score(y_test, y_scores))
    roc_auc = float(roc_auc_score(y_test, y_scores))
    precision_val = float(precision_score(y_test, y_pred, zero_division=0))
    recall_val = float(recall_score(y_test, y_pred, zero_division=0))
    f1_val = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    cr_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    print("\n" + "=" * 50)
    print("XGBOOST EVALUATION RESULTS")
    print("=" * 50)
    print(f"Average Precision (PR-AUC): {pr_auc:.4f}")
    print(f"ROC-AUC:                    {roc_auc:.4f}")
    print(f"Fraud Precision (Class 1):  {precision_val:.4f}")
    print(f"Fraud Recall (Class 1):     {recall_val:.4f}")
    print(f"Fraud F1-Score:             {f1_val:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Log metrics to MLflow
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("precision", precision_val)
    mlflow.log_metric("recall", recall_val)
    mlflow.log_metric("f1_score", f1_val)
    mlflow.log_metric("true_negatives", int(cm[0, 0]))
    mlflow.log_metric("false_positives", int(cm[0, 1]))
    mlflow.log_metric("false_negatives", int(cm[1, 0]))
    mlflow.log_metric("true_positives", int(cm[1, 1]))

    # Step 12: Save XGBoost model artifacts
    xgb_model_path = MODELS_DIR / "xgboost_model.json"
    model.save_model(xgb_model_path)
    joblib.dump(model, MODELS_DIR / "xgboost_model.joblib")

    # Combined inference pipeline
    xgb_pipeline = Pipeline([
        ("scaler", scaler),
        ("model", model)
    ])
    joblib.dump(xgb_pipeline, MODELS_DIR / "xgboost_pipeline.joblib")
    print(f"\nSaved XGBoost artifacts to {MODELS_DIR}:")
    print(" - xgboost_model.json")
    print(" - xgboost_model.joblib")
    print(" - xgboost_pipeline.joblib")

    # Step 13: Save metrics JSON
    xgb_metrics = {
        "model_name": "XGBoost Classifier",
        "scale_pos_weight": round(scale_pos_weight, 2),
        "random_state": 42,
        "train_transactions": int(len(X_train)),
        "test_transactions": int(len(X_test)),
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "precision": round(precision_val, 6),
        "recall": round(recall_val, 6),
        "f1_score": round(f1_val, 6),
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1])
        },
        "classification_report": cr_dict
    }

    metrics_out_path = OUTPUTS_DIR / "xgboost_metrics.json"
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(xgb_metrics, f, indent=4)
    print(f"Saved evaluation metrics to {metrics_out_path}")

    # Step 14: Save Confusion Matrix Plot
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
        ax=ax_cm
    )
    ax_cm.set_title("Confusion Matrix - XGBoost Classifier")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = OUTPUTS_DIR / "xgboost_confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=300)
    plt.close(fig_cm)
    print(f"Saved confusion matrix plot to {cm_path}")

    # Step 15: Save Precision-Recall Curve Plot
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
    ax_pr.plot(recall_curve, precision_curve, color="crimson", lw=2, label=f"XGBoost PR curve (PR-AUC = {pr_auc:.4f})")
    baseline_ratio = float(y_test.sum() / len(y_test))
    ax_pr.plot([0, 1], [baseline_ratio, baseline_ratio], color="navy", lw=1.5, linestyle="--", label=f"No Skill Baseline ({baseline_ratio:.4f})")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve - XGBoost Classifier")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = OUTPUTS_DIR / "xgboost_precision_recall_curve.png"
    fig_pr.savefig(pr_path, dpi=300)
    plt.close(fig_pr)
    print(f"Saved precision-recall curve to {pr_path}")

    # Log artifacts to MLflow
    mlflow.log_artifact(str(metrics_out_path))
    mlflow.log_artifact(str(cm_path))
    mlflow.log_artifact(str(pr_path))

    # Step 16: Model Comparison Summary (Baseline vs XGBoost)
    baseline_metrics_path = OUTPUTS_DIR / "metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            base_metrics = json.load(f)

        comparison = {
            "metrics_comparison": {
                "PR-AUC": {
                    "Logistic_Regression_Baseline": base_metrics.get("pr_auc"),
                    "XGBoost": round(pr_auc, 6),
                    "Delta": round(pr_auc - base_metrics.get("pr_auc", 0), 6)
                },
                "ROC-AUC": {
                    "Logistic_Regression_Baseline": base_metrics.get("roc_auc"),
                    "XGBoost": round(roc_auc, 6),
                    "Delta": round(roc_auc - base_metrics.get("roc_auc", 0), 6)
                },
                "Fraud_Precision": {
                    "Logistic_Regression_Baseline": base_metrics.get("precision"),
                    "XGBoost": round(precision_val, 6),
                    "Delta": round(precision_val - base_metrics.get("precision", 0), 6)
                },
                "Fraud_Recall": {
                    "Logistic_Regression_Baseline": base_metrics.get("recall"),
                    "XGBoost": round(recall_val, 6),
                    "Delta": round(recall_val - base_metrics.get("recall", 0), 6)
                },
                "Fraud_F1": {
                    "Logistic_Regression_Baseline": base_metrics.get("f1_score"),
                    "XGBoost": round(f1_val, 6),
                    "Delta": round(f1_val - base_metrics.get("f1_score", 0), 6)
                },
                "False_Positives": {
                    "Logistic_Regression_Baseline": base_metrics.get("confusion_matrix", {}).get("false_positive"),
                    "XGBoost": int(cm[0, 1]),
                    "Delta": int(cm[0, 1]) - int(base_metrics.get("confusion_matrix", {}).get("false_positive", 0))
                }
            }
        }
        comparison_path = OUTPUTS_DIR / "model_comparison.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=4)
        mlflow.log_artifact(str(comparison_path))
        print(f"\nSaved model comparison summary to {comparison_path}")

print("\nXGBoost training and MLflow tracking completed successfully!")
