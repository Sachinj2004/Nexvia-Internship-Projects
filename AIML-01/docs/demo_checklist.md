# Nexvia AIML-01: Live Evaluation Demonstration Runbook

This checklist provides a step-by-step procedure to demonstrate the full functionality of the **Nexvia Real-Time Financial Fraud Detection Platform** during the final internship evaluation.

---

## Pre-Demo Setup & Environment Verification

* [ ] **Hardware**: MacBook Air M1 (or compatible macOS/Linux system).
* [ ] **Working Directory**: `$HOME/my intern`
* [ ] **Active Conda Environment**: `nexvia-fraud` (Python 3.11.17)
  ```bash
  cd "$HOME/my intern"
  conda activate nexvia-fraud
  ```
* [ ] **Port Availability Check**: Ensure ports `8000` (FastAPI) and `8501` (Streamlit) are free.

---

## Phase 1: Automated Test Suite Verification (30 Seconds)

Execute the full automated test suite to prove system stability and regression-free operation.

```bash
pytest tests/ -v
```

* **Expected Output**:
  ```text
  ======================= 79 passed, 3 warnings in ~36s =======================
  ```
* **Key Demonstration Points**:
  * 79 passed tests across all 6 subsystems.
  * 3 known non-critical warnings (`StarletteDeprecationWarning`, `torch.jit` deprecation, `mlflow` noload deprecation).
  * 0 failures, 0 errors.

---

## Phase 2: REST API Live Demonstration (2 Minutes)

### Step 1: Launch FastAPI Service
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
*Open a second terminal window for curl commands or navigate to `http://localhost:8000/docs` in your browser.*

### Step 2: System Health Check
```bash
curl -s http://localhost:8000/health | jq .
```
* **Verify**: `"status": "healthy"`, `"model_loaded": true`, and process RSS memory usage $<150\text{MB}$.

### Step 3: Model Metadata & Benchmarks
```bash
curl -s http://localhost:8000/model/info | jq .
```
* **Verify**: 30 expected features displayed in order; PR-AUC benchmark reported as `0.868619`.

### Step 4: Real-Time Single Transaction Scoring (Normal)
```bash
curl -s -X POST "http://localhost:8000/predict" \
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
    "Amount": 149.62
  }' | jq .
```
* **Verify**: `"predicted_class": 0`, `"risk_label": "Legitimate"`, `"status": "APPROVED"`, `"inference_latency_ms" < 10.0 (typically 2.8–4.2ms on local Apple M1).

### Step 5: Real-Time Single Transaction Scoring (Fraud Anomaly)
```bash
curl -s -X POST "http://localhost:8000/predict" \
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
    "Amount": 0.0
  }' | jq .
```
* **Verify**: `"predicted_class": 1`, `"risk_label": "Fraud"`, `"status": "FLAGGED_FOR_REVIEW"`, `fraud_probability > 0.90`.

---

## Phase 3: Interactive Streamlit Dashboard Walkthrough (4 Minutes)

Launch or navigate to the running Streamlit dashboard:
```bash
streamlit run app.py --server.port 8501
```
*Open `http://localhost:8501` in your browser.*

### Tab 1: 🔍 Real-Time Single Transaction Scoring
* [ ] Click **"🟢 Load Normal Sample"** $\rightarrow$ Click **"🚀 Analyze Transaction"**:
  * Verify green banner: **"✅ TRANSACTION APPROVED"**.
  * Note inference latency (<5ms).
* [ ] Click **"🔴 Load Fraud Sample"** $\rightarrow$ Click **"🚀 Analyze Transaction"**:
  * Verify red banner: **"🚨 TRANSACTION FLAGGED FOR REVIEW"** (Fraud Probability ~98%).
  * Verify info toast: **"📋 Alert registered into Analyst Review Queue as case TX-..."** (demonstrating human-in-the-loop automated case queuing).

### Tab 2: 📁 Batch Transaction Screening
* [ ] Click **"📋 Load Sample Batch (10 Transactions)"** $\rightarrow$ Click **"🚀 Run Batch Fraud Screening"**:
  * Verify KPI metrics: Total Transactions: 10, Approved: 8, Flagged: 2, Fraud Rate: 20%.
  * Review the interactive batch prediction table with color-coded risk flags.

### Tab 3: 📈 Model Benchmark & Evaluation
* [ ] Review the offline model comparison metrics table (Baseline Logistic Regression vs. Production XGBoost).
* [ ] Point out the 98.3% false positive reduction (from 1,386 to 24; 1,362 fewer false alarms).
* [ ] Inspect the high-resolution Confusion Matrix and Precision-Recall Curve visualizations.

### Tab 4: 📊 Evidently AI Data-Drift Monitoring
* [ ] Point out the drift KPI metrics: Status, Drifted Features, Drift Share, and Evaluated Sample size.
* [ ] Click **"⚡ Run Drift Check (Test Set Sample)"**:
  * Watch spinner execute drift analysis in the isolated conda environment.
  * Page refreshes with updated drift metrics.
* [ ] Scroll down to inspect the interactive Evidently AI HTML drift report embedded directly within the dashboard.

### Tab 5: 📋 Analyst Case Review & Feedback
* [ ] Point out the KPI summary cards: Total Cases, New Alerts, Under Review, Confirmed Fraud, Legitimate.
* [ ] Locate the newly registered case from Tab 1 in the case review table.
* [ ] Select the case in the **"Select Case to Investigate"** dropdown.
* [ ] Change status to **"Confirmed Fraud"**, enter investigation notes (e.g., *"Confirmed stolen card velocity spike with cardholder"*), and click **"💾 Save Case Decision"**.
* [ ] Verify success toast and table refresh, confirming SQLite persistence across dashboard restarts.

---

## Phase 4: Streaming & Feature Store Verification (1.5 Minutes)

Demonstrate end-to-end streaming ingestion and Feast velocity enrichment:

```bash
# Execute standalone streaming validation test
python -c "
from src.streaming.producer import TransactionProducer
from src.streaming.consumer import StreamingConsumer
from src.predict import EXPECTED_FEATURES

producer = TransactionProducer()
consumer = StreamingConsumer()

sample = {col: 0.0 for col in EXPECTED_FEATURES}
sample['Time'] = 100.0
sample['Amount'] = 50.0

success = producer.send_transaction(sample, transaction_id=12345)
print('Producer sent transaction:', success)

tx = consumer.poll_transaction()
print('Consumer received transaction:', tx is not None)
"
```
* **Verify**: Producer successfully delivers payload; consumer polls and receives transaction with DLQ protection.

---

## Phase 5: GNN & Hybrid Scorer Verification (1 Minute)

Demonstrate the `HybridFraudPredictor` and its defensive fallback to XGBoost:

```bash
python -c "
from src.hybrid import HybridFraudPredictor

hybrid_pred = HybridFraudPredictor(enable_gnn=True)
sample = {f'V{i}': 0.0 for i in range(1, 29)}
sample['Time'] = 100.0
sample['Amount'] = 50.0

res = hybrid_pred.predict_single(sample)
print('Hybrid Result:')
print(' - P(hybrid):', res['fraud_probability'])
print(' - P(xgb):   ', res['xgb_probability'])
print(' - P(gnn):   ', res['gnn_probability'])
print(' - Mode:     ', res['scoring_mode'])

# Test automated fallback when GNN is disabled
fallback_pred = HybridFraudPredictor(enable_gnn=False)
res_fb = fallback_pred.predict_single(sample)
print('\nFallback Result:')
print(' - Mode:     ', res_fb['scoring_mode'])
print(' - Fallback: P(final) == P(xgb):', res_fb['fraud_probability'] == res_fb['xgb_probability'])
"
```
* **Verify**: Both hybrid scoring and automatic 100% XGBoost fallback execute cleanly without errors.

---

## Phase 6: Graceful Shutdown

* [ ] Stop the FastAPI server (`Ctrl+C` in Terminal 1).
* [ ] Keep the Streamlit dashboard running if needed, or terminate via `Ctrl+C`.
* [ ] Confirm that no lingering processes or temporary files remain.
