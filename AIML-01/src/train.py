import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Step 1: Find the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Step 2: Locate the dataset
DATA_PATH = PROJECT_DIR / "data" / "raw" / "creditcard.csv"

# Step 3: Load the dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Step 4: Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("Total transactions:", len(df))
print("Total features:", X.shape[1])

# Step 5: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training transactions:", len(X_train))
print("Testing transactions:", len(X_test))

# Step 6: Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Create the model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Step 8: Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# Step 9: Predict transaction classes
y_pred = model.predict(X_test)
# Get fraud probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate Precision-Recall metrics
precision_curve, recall_curve, thresholds = precision_recall_curve(
    y_test,
    y_scores
)

pr_auc = average_precision_score(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)
precision_val = precision_score(y_test, y_pred, zero_division=0)
recall_val = recall_score(y_test, y_pred, zero_division=0)
f1_val = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
cr_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

print(f"\nAverage Precision (PR-AUC): {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"Recall: {recall_val:.4f}")
print(f"F1-Score: {f1_val:.4f}")

# Step 10: Evaluate the model
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Step 11: Setup output directories
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Step 12: Save trained model and preprocessing pipeline
print(f"\nSaving model artifacts to {MODELS_DIR}...")
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
joblib.dump(model, MODELS_DIR / "baseline_model.joblib")

pipeline = Pipeline([
    ("scaler", scaler),
    ("model", model)
])
joblib.dump(pipeline, MODELS_DIR / "pipeline.joblib")
print("Saved scaler.joblib, baseline_model.joblib, and pipeline.joblib successfully.")

# Step 13: Save evaluation metrics as JSON
metrics_data = {
    "model_name": "Logistic Regression Baseline",
    "class_weight": "balanced",
    "random_state": 42,
    "train_transactions": int(len(X_train)),
    "test_transactions": int(len(X_test)),
    "pr_auc": round(float(pr_auc), 6),
    "roc_auc": round(float(roc_auc), 6),
    "precision": round(float(precision_val), 6),
    "recall": round(float(recall_val), 6),
    "f1_score": round(float(f1_val), 6),
    "confusion_matrix": {
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1])
    },
    "classification_report": cr_dict
}

metrics_path = OUTPUTS_DIR / "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)
print(f"Saved evaluation metrics to {metrics_path}")

# Step 14: Save Confusion Matrix image
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
ax_cm.set_title("Confusion Matrix - Logistic Regression Baseline")
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
plt.tight_layout()
cm_plot_path = OUTPUTS_DIR / "confusion_matrix.png"
fig_cm.savefig(cm_plot_path, dpi=300)
plt.close(fig_cm)
print(f"Saved confusion matrix plot to {cm_plot_path}")

# Step 15: Save Precision-Recall curve image
fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
ax_pr.plot(recall_curve, precision_curve, color="darkorange", lw=2, label=f"PR curve (PR-AUC = {pr_auc:.4f})")
baseline_ratio = float(y_test.sum() / len(y_test))
ax_pr.plot([0, 1], [baseline_ratio, baseline_ratio], color="navy", lw=1.5, linestyle="--", label=f"No Skill Baseline ({baseline_ratio:.4f})")
ax_pr.set_xlim([0.0, 1.0])
ax_pr.set_ylim([0.0, 1.05])
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve - Logistic Regression Baseline")
ax_pr.legend(loc="upper right")
ax_pr.grid(alpha=0.3)
plt.tight_layout()
pr_curve_path = OUTPUTS_DIR / "precision_recall_curve.png"
fig_pr.savefig(pr_curve_path, dpi=300)
plt.close(fig_pr)
print(f"Saved precision-recall curve to {pr_curve_path}")

print("\nModel training and artifact generation completed successfully!")
