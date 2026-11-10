# Nexvia AIML-01: REST API Reference & Integration Guide

## 1. Overview & Service Specifications

The **Nexvia Financial Fraud Detection Microservice** (`src/api.py`) is a high-throughput REST API built with FastAPI, achieving measured single-transaction local inference latency of 2.8–4.2 ms on Apple M1 (sub-10ms design target; local performance benchmark, not an SLA guarantee). It provides synchronous single-transaction and batch fraud inference, model metadata introspection, Evidently AI data-drift metrics, and on-demand drift reporting.

* **Default Host**: `http://localhost:8000` (or `http://0.0.0.0:8000`)
* **Interactive Documentation (Swagger UI)**: `http://localhost:8000/docs`
* **Alternative Documentation (ReDoc)**: `http://localhost:8000/redoc`
* **OpenAPI Specification**: `http://localhost:8000/openapi.json`
* **Response Format**: `application/json` (except `/drift/report` which serves `text/html`)

---

## 2. Quickstart & Service Launch

```bash
# 1. Activate project environment
conda activate nexvia-fraud

# 2. Launch FastAPI service via Uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 3. Verify service liveness
curl -s http://localhost:8000/health | jq .
```

---

## 3. API Endpoints Catalog

### Endpoint 1: System Health & Readiness Probe
* **Method**: `GET`
* **Path**: `/health`
* **Tags**: `Health`
* **Description**: Verifies service liveness, memory usage, Python version, uptime, and active model pipeline readiness.

#### Example Request
```bash
curl -X GET "http://localhost:8000/health"
```

#### Example 200 OK Response
```json
{
  "status": "healthy",
  "timestamp": "2026-10-30T01:15:00.000000Z",
  "version": "1.0.0",
  "model_loaded": true,
  "model_path": "/Users/sachinsahebraojadhav/my intern/models/xgboost_pipeline.joblib",
  "drift_monitor_active": true,
  "buffered_transactions": 142,
  "system_info": {
    "python_version": "3.11.17",
    "platform": "macOS-15.0-arm64-arm-64bit",
    "process_rss_mb": 118.4
  }
}
```

---

### Endpoint 2: Model Metadata & Benchmark Retrieval
* **Method**: `GET`
* **Path**: `/model/info`
* **Tags**: `Metadata`
* **Description**: Returns architecture specifications, the 30 expected feature names in order, default decision cutoff, and offline evaluation benchmarks.

#### Example Request
```bash
curl -X GET "http://localhost:8000/model/info"
```

#### Example 200 OK Response
```json
{
  "model_name": "XGBoost Production Fraud Detection Pipeline",
  "artifact_path": "/Users/sachinsahebraojadhav/my intern/models/xgboost_pipeline.joblib",
  "feature_count": 30,
  "expected_features": [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
  ],
  "default_threshold": 0.5,
  "benchmarks": {
    "model_name": "XGBoost Classifier",
    "scale_pos_weight": 577.29,
    "pr_auc": 0.868619,
    "roc_auc": 0.979963,
    "precision": 0.775701,
    "recall": 0.846939,
    "f1_score": 0.809756,
    "confusion_matrix": {
      "true_negative": 56840,
      "false_positive": 24,
      "false_negative": 15,
      "true_positive": 83
    }
  }
}
```

---

### Endpoint 3: Real-Time Single Transaction Scoring
* **Method**: `POST`
* **Path**: `/predict`
* **Tags**: `Inference`
* **Description**: Scores a single payment transaction. Automatically buffers features into the sliding-window drift queue and queries Feast for enriched velocity attributes if `transaction_id` is supplied.

#### Request Schema (`TransactionPayload`)
| Field | Type | Required | Description |
| :--- | :---: | :---: | :--- |
| `Time` | `float` | Yes | Seconds elapsed since the baseline epoch |
| `V1`–`V28` | `float` | Yes | Anonymized PCA features |
| `Amount` | `float` | Yes | Monetary value of the payment |
| `threshold` | `float` | No | Optional custom decision cutoff ($0.0 \le t \le 1.0$) |
| `transaction_id` | `int` | No | Optional ID for Feast velocity enrichment |

#### Example Request: Legitimate Transaction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 100.0,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
    "Amount": 149.62,
    "threshold": 0.50
  }'
```

#### Example 200 OK Response (Approved)
```json
{
  "fraud_probability": 0.00105,
  "predicted_class": 0,
  "risk_label": "Legitimate",
  "threshold_used": 0.5,
  "status": "APPROVED",
  "inference_latency_ms": 3.84,
  "timestamp": "2026-10-30T01:16:12.123456Z",
  "transaction_id": null,
  "enriched_features": null
}
```

#### Example Request: Fraudulent Transaction Payload
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "V1": -2.3122, "V2": 1.9520, "V3": -1.6099, "V4": 3.9979,
    "V5": -0.5222, "V6": -1.4265, "V7": -2.5374, "V8": 1.3917,
    "V9": -2.7701, "V10": -2.7723, "V11": 3.2020, "V12": -2.8999,
    "V13": -0.5952, "V14": -4.2893, "V15": 0.3898, "V16": -1.1407,
    "V17": -2.8301, "V18": -0.0155, "V19": 0.4161, "V20": 0.1269,
    "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202,
    "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1433,
    "Amount": 0.0,
    "threshold": 0.50
  }'
```

#### Example 200 OK Response (Flagged for Review)
```json
{
  "fraud_probability": 0.98421,
  "predicted_class": 1,
  "risk_label": "Fraud",
  "threshold_used": 0.5,
  "status": "FLAGGED_FOR_REVIEW",
  "inference_latency_ms": 3.62,
  "timestamp": "2026-10-30T01:16:30.987654Z",
  "transaction_id": null,
  "enriched_features": null
}
```

---

### Endpoint 4: High-Throughput Batch Screening
* **Method**: `POST`
* **Path**: `/predict/batch`
* **Tags**: `Inference`
* **Description**: Vectorized evaluation of multiple transaction payloads in a single request. Returns individual scores and summary KPI statistics.

#### Example Request
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "threshold": 0.50,
    "transactions": [
      {
        "Time": 100.0,
        "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0, "V6": 0.0,
        "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0, "V11": 0.0, "V12": 0.0,
        "V13": 0.0, "V14": 0.0, "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0,
        "V19": 0.0, "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
        "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0,
        "Amount": 45.00
      },
      {
        "Time": 406.0,
        "V1": -2.3122, "V2": 1.9520, "V3": -1.6099, "V4": 3.9979, "V5": -0.5222, "V6": -1.4265,
        "V7": -2.5374, "V8": 1.3917, "V9": -2.7701, "V10": -2.7723, "V11": 3.2020, "V12": -2.8999,
        "V13": -0.5952, "V14": -4.2893, "V15": 0.3898, "V16": -1.1407, "V17": -2.8301, "V18": -0.0155,
        "V19": 0.4161, "V20": 0.1269, "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202,
        "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1433,
        "Amount": 0.00
      }
    ]
  }'
```

#### Example 200 OK Response
```json
{
  "total_transactions": 2,
  "approved_count": 1,
  "flagged_count": 1,
  "fraud_rate_percent": 50.0,
  "threshold_used": 0.5,
  "batch_latency_ms": 6.81,
  "latency_per_tx_ms": 3.405,
  "results": [
    {
      "fraud_probability": 0.00105,
      "predicted_class": 0,
      "risk_label": "Legitimate",
      "threshold_used": 0.5,
      "status": "APPROVED",
      "inference_latency_ms": 3.405,
      "timestamp": "2026-10-30T01:17:00.000000Z",
      "transaction_id": null,
      "enriched_features": null
    },
    {
      "fraud_probability": 0.98421,
      "predicted_class": 1,
      "risk_label": "Fraud",
      "threshold_used": 0.5,
      "status": "FLAGGED_FOR_REVIEW",
      "inference_latency_ms": 3.405,
      "timestamp": "2026-10-30T01:17:00.000000Z",
      "transaction_id": null,
      "enriched_features": null
    }
  ]
}
```

---

### Endpoint 5: Data Drift Status
* **Method**: `GET`
* **Path**: `/drift/status`
* **Tags**: `Model Monitoring`
* **Description**: Returns the latest statistical drift test summary and in-memory buffer statistics.

#### Example Request
```bash
curl -X GET "http://localhost:8000/drift/status"
```

#### Example 200 OK Response
```json
{
  "status": "success",
  "dataset_drift": false,
  "drift_share": 0.033,
  "number_of_drifted_columns": 1,
  "number_of_columns": 30,
  "stat_test": "Kolmogorov-Smirnov (KS)",
  "current_rows": 500,
  "reference_rows": 15000,
  "buffered_transactions": 142,
  "report_available": true,
  "timestamp": "2026-10-30T01:14:30.000000Z"
}
```

---

### Endpoint 6: Trigger On-Demand Drift Analysis
* **Method**: `POST`
* **Path**: `/drift/evaluate`
* **Tags**: `Model Monitoring`
* **Description**: Runs Evidently AI two-sample statistical tests comparing buffered scoring data against baseline reference data (`data/reference_data.parquet`).

#### Query Parameters
* `sample_size` (int, default: 15,000): Subsample size of baseline records.

#### Example Request
```bash
curl -X POST "http://localhost:8000/drift/evaluate?sample_size=5000"
```

---

### Endpoint 7: Interactive Drift Report View
* **Method**: `GET`
* **Path**: `/drift/report`
* **Tags**: `Model Monitoring`
* **Description**: Serves the standalone, fully interactive Evidently HTML report (`outputs/drift_report.html`) for browser viewing.

#### Example Request (Open in Web Browser)
```bash
# Navigate in browser to:
open http://localhost:8000/drift/report
```

---

## 4. Error Responses & Status Codes

| HTTP Status | Reason | Payload Example |
| :---: | :--- | :--- |
| **`400 Bad Request`** | Missing required features or non-numeric feature types | `{"detail": "Schema validation error: Input is missing 2 required feature(s): ['V14', 'Amount']"}` |
| **`404 Not Found`** | Requested report or resource does not exist | `{"detail": "Drift report HTML not found. Trigger a drift evaluation first via /drift/evaluate."}` |
| **`422 Unprocessable Entity`** | Pydantic validation failure (e.g., negative Amount) | `{"detail": [{"loc": ["body", "Amount"], "msg": "Input should be greater than or equal to 0"}]}` |
| **`500 Internal Error`** | Unhandled pipeline exception | `{"detail": "Inference execution failed: Model file corrupted."}` |
