"""
Real-Time Financial Fraud Detection Platform (AIML-01)
Streamlit Web Dashboard: app.py

A lightweight, beginner-friendly interactive interface for the Nexvia Fraud Detection system.
Reuses the trained production XGBoost pipeline via src.predict without retraining.
Optimized for low resource usage on MacBook Air M1 (8GB RAM).
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    DEFAULT_MODEL_PATH,
    EXPECTED_FEATURES,
    FraudPredictor,
    predict_batch,
    predict_single
)
from src.analyst_feedback import (
    VALID_STATUSES,
    get_all_cases,
    get_case_summary,
    record_case,
    update_case_status,
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "reference_data.parquet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# -----------------------------------------------------------------------------
# Environment & Remote API Configuration
# -----------------------------------------------------------------------------
def is_running_on_cloud() -> bool:
    """Detect if running on Streamlit Community Cloud or remote cloud container."""
    if os.path.exists("/home/appuser") or os.environ.get("USER") == "appuser":
        return True
    if os.environ.get("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION") == "false":
        return True
    if any(k in os.environ for k in ("STREAMLIT_SHARING_MODE", "STREAMLIT_HOST", "IS_STREAMLIT_CLOUD")):
        return True
    try:
        if hasattr(st, "context"):
            headers = getattr(st.context, "headers", {}) or {}
            host = headers.get("host", "") or headers.get("x-forwarded-host", "")
            if "streamlit.app" in host:
                return True
            url = getattr(st.context, "url", "") or ""
            if "streamlit.app" in url:
                return True
    except Exception:
        pass
    return False


def is_valid_public_url(url: Optional[str]) -> bool:
    """Validate that a URL is a valid remote/public HTTP/HTTPS endpoint."""
    if not url or not isinstance(url, str):
        return False
    url_clean = url.strip().lower()
    if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        return False
    for private in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        if private in url_clean:
            return False
    return True


def get_configured_k8s_url() -> Optional[str]:
    """Retrieve configured Kubernetes API URL from environment variables or Streamlit secrets."""
    url = os.environ.get("K8S_API_URL") or os.environ.get("KUBERNETES_API_URL")
    if not url:
        try:
            url = st.secrets.get("K8S_API_URL") or st.secrets.get("k8s_api_url")
        except Exception:
            url = None
    return str(url).strip().rstrip("/") if url else None


RUNNING_ON_CLOUD = is_running_on_cloud()
_configured_url = get_configured_k8s_url()

if RUNNING_ON_CLOUD:
    # On public cloud, only enable Kubernetes if a valid public HTTPS/HTTP URL is provided
    if is_valid_public_url(_configured_url):
        K8S_API_BASE_URL: Optional[str] = _configured_url
        K8S_ENABLED: bool = True
    else:
        K8S_API_BASE_URL = None
        K8S_ENABLED = False
else:
    # Running locally on MacBook
    K8S_API_BASE_URL = _configured_url or "http://localhost:8000"
    K8S_ENABLED = True


# -----------------------------------------------------------------------------
# Kubernetes Microservice API Client Helpers
# -----------------------------------------------------------------------------
def check_k8s_api_health(base_url: Optional[str] = None) -> Dict[str, Any]:
    """Check health and connectivity of Kubernetes microservice."""
    target_url = base_url or K8S_API_BASE_URL
    if not target_url:
        return {"online": False, "data": None, "error": "No Kubernetes API URL configured"}

    # Guard: Never connect to localhost when running on public cloud
    if RUNNING_ON_CLOUD and not is_valid_public_url(target_url):
        return {"online": False, "data": None, "error": "Localhost connection disabled on cloud deployment"}

    try:
        resp = requests.get(f"{target_url}/health", timeout=1.5)
        if resp.status_code == 200:
            return {"online": True, "data": resp.json(), "error": None}
        return {"online": False, "data": None, "error": f"HTTP status {resp.status_code}"}
    except Exception as exc:
        return {"online": False, "data": None, "error": str(exc)}


def predict_via_k8s_single(tx_payload: Dict[str, Any], base_url: Optional[str] = None) -> Dict[str, Any]:
    """Send single transaction to Kubernetes /predict endpoint."""
    target_url = base_url or K8S_API_BASE_URL
    if not target_url:
        raise RuntimeError("No Kubernetes API URL configured.")
    resp = requests.post(f"{target_url}/predict", json=tx_payload, timeout=4.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Kubernetes API error (HTTP {resp.status_code}): {resp.text}")
    return resp.json()


def predict_via_k8s_batch(records: List[Dict[str, Any]], threshold: float = 0.5, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Send batch transaction payload to Kubernetes /predict/batch endpoint."""
    target_url = base_url or K8S_API_BASE_URL
    if not target_url:
        raise RuntimeError("No Kubernetes API URL configured.")
    payload = {"transactions": records, "threshold": threshold}
    resp = requests.post(f"{target_url}/predict/batch", json=payload, timeout=10.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Kubernetes Batch API error (HTTP {resp.status_code}): {resp.text}")
    return resp.json()


# -----------------------------------------------------------------------------
# Streamlit Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexvia Fraud Detection Platform (AIML-01)",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .status-approved {
        background-color: #d4edda;
        color: #155724;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        border-left: 5px solid #28a745;
    }
    .status-flagged {
        background-color: #f8d7da;
        color: #721c24;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        border-left: 5px solid #dc3545;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# Resource Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading XGBoost Fraud Detection Pipeline...")
def get_cached_predictor() -> Optional[FraudPredictor]:
    """Load and cache the trained inference pipeline in memory."""
    if not DEFAULT_MODEL_PATH.exists():
        return None
    try:
        return FraudPredictor(model_path=DEFAULT_MODEL_PATH)
    except Exception as exc:
        st.error(f"Error loading model pipeline: {exc}")
        return None


@st.cache_data
def load_metrics_data() -> Dict:
    """Load saved evaluation metrics from outputs directory."""
    metrics_file = OUTPUTS_DIR / "xgboost_metrics.json"
    comparison_file = OUTPUTS_DIR / "model_comparison.json"
    data = {}
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            data["xgboost"] = json.load(f)
    if comparison_file.exists():
        with open(comparison_file, "r", encoding="utf-8") as f:
            data["comparison"] = json.load(f)
    return data


# -----------------------------------------------------------------------------
# Header & Sidebar
# -----------------------------------------------------------------------------
st.title("💳 Real-Time Financial Fraud Detection Platform")
st.caption("Production Inference & Transaction Screening")

predictor = get_cached_predictor()

# Sidebar: System & Model Status
st.sidebar.header("⚙️ Platform Control")

# Inference Engine Selector (Cloud defaults to Local, MacBook defaults to Kubernetes)
st.sidebar.subheader("🔌 Inference Engine")

LOCAL_ENGINE_LABEL = "Local In-Memory Pipeline"

if RUNNING_ON_CLOUD:
    # On Streamlit Cloud: Local In-Memory Pipeline is default
    if K8S_ENABLED and K8S_API_BASE_URL:
        engine_options = [LOCAL_ENGINE_LABEL, f"Kubernetes Microservice ({K8S_API_BASE_URL})"]
        inference_engine = st.sidebar.radio(
            "Select Backend",
            engine_options,
            index=0,
            help="Local In-Memory Pipeline scores transactions directly via models/xgboost_pipeline.joblib."
        )
    else:
        inference_engine = LOCAL_ENGINE_LABEL
        st.sidebar.info("⚡ Default Backend: **Local In-Memory Pipeline** (`models/xgboost_pipeline.joblib`)")
        st.sidebar.caption(
            "ℹ️ *Kubernetes microservice is disabled on public deployment. "
            "To route inference to an external cluster, configure `K8S_API_URL` in Streamlit Secrets.*"
        )
else:
    # On local MacBook: Kubernetes Microservice is default; Local Pipeline is alternative
    k8s_label = f"Kubernetes Microservice ({K8S_API_BASE_URL})"
    engine_options = [k8s_label, LOCAL_ENGINE_LABEL]
    inference_engine = st.sidebar.radio(
        "Select Backend",
        engine_options,
        index=0,
        help="Default routes live inference to the Kubernetes pod over localhost:8000."
    )

# Only check Kubernetes health if Kubernetes is explicitly selected
if inference_engine.startswith("Kubernetes") and K8S_API_BASE_URL:
    k8s_status = check_k8s_api_health(K8S_API_BASE_URL)
    if k8s_status["online"]:
        st.sidebar.success("✅ Kubernetes Pod: **ONLINE**")
        k8s_data = k8s_status.get("data", {})
        st.sidebar.caption(
            f"**Endpoint:** `{K8S_API_BASE_URL}/predict`\n\n"
            f"**Model:** {k8s_data.get('model_type', 'XGBoost Pipeline')}\n\n"
            f"**Platform:** {k8s_data.get('platform_info', 'Linux aarch64')}\n\n"
            f"**Memory RSS:** {k8s_data.get('memory_rss_mb', 'N/A')} MB"
        )
    else:
        st.sidebar.error("❌ Kubernetes Pod: **OFFLINE**")
        st.sidebar.caption(
            f"Cannot connect to `{K8S_API_BASE_URL}`: {k8s_status.get('error')}\n\n"
            "Ensure port-forwarding is running:\n"
            "`kubectl port-forward svc/nexvia-fraud-api-service 8000:8000`\n"
            "or switch back to Local In-Memory Pipeline above."
        )
else:
    k8s_status = {"online": False, "data": None, "error": None}
    if predictor is not None:
        st.sidebar.success("✅ Local Model Pipeline: **ONLINE**")
        st.sidebar.markdown(f"**Model:** XGBoost Classifier (`scale_pos_weight=577.29`)")
        st.sidebar.markdown(f"**Artifact:** `models/xgboost_pipeline.joblib`")
    else:
        st.sidebar.error("❌ Local Model Pipeline: **OFFLINE**")
        st.sidebar.warning(f"Missing artifact: `{DEFAULT_MODEL_PATH}`\nPlease train the model first.")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Transactions with fraud probability at or above this cutoff will be flagged for review."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance (Test Set)")
metrics = load_metrics_data()
xgb_m = metrics.get("xgboost", {})
st.sidebar.metric("PR-AUC (Primary)", f"{xgb_m.get('pr_auc', 0.8686):.4f}")
st.sidebar.metric("Fraud Precision", f"{xgb_m.get('precision', 0.7757)*100:.1f}%")
st.sidebar.metric("Fraud Recall", f"{xgb_m.get('recall', 0.8469)*100:.1f}%")
st.sidebar.metric("F1-Score", f"{xgb_m.get('f1_score', 0.8098):.4f}")
cm = xgb_m.get("confusion_matrix", {})
st.sidebar.caption(f"False Positives: **{cm.get('false_positive', 24)}** / 56,864")


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_single, tab_batch, tab_audit, tab_drift, tab_feedback = st.tabs([
    "🔍 Real-Time Single Transaction Scoring",
    "📁 Batch Transaction Screening",
    "📈 Model Benchmark & Evaluation",
    "📊 Evidently AI Data-Drift Monitoring",
    "📋 Analyst Case Review & Feedback"
])


# -----------------------------------------------------------------------------
# Tab 1: Single Transaction Scoring
# -----------------------------------------------------------------------------
with tab_single:
    st.subheader("Real-Time Transaction Risk Assessment")
    st.markdown("Evaluate an individual transaction payload in real time to detect fraud indicators.")

    if predictor is None and not inference_engine.startswith("Kubernetes"):
        st.error("Cannot score transactions: Local model artifact is missing from `models/`.")
    else:
        # Preset buttons for quick demonstration
        col_preset1, col_preset2, col_preset3 = st.columns([1, 1, 2])
        preset_type = None

        with col_preset1:
            if st.button("🟢 Load Normal Sample"):
                preset_type = "legit"
        with col_preset2:
            if st.button("🔴 Load Fraud Sample"):
                preset_type = "fraud"

        # Preset values configuration
        default_amount = 149.62
        default_time = 100.0
        v_values = {f"V{i}": 0.0 for i in range(1, 29)}

        if preset_type == "legit":
            default_amount = 149.62
            default_time = 0.0
            v_values = {f"V{i}": 0.0 for i in range(1, 29)}
            v_values["V1"] = -1.3598
            v_values["V2"] = -0.0728
            v_values["V3"] = 2.5363
            v_values["V4"] = 1.3782
        elif preset_type == "fraud":
            default_amount = 0.0
            default_time = 406.0
            v_values = {f"V{i}": 0.0 for i in range(1, 29)}
            v_values["V1"] = -2.3122
            v_values["V2"] = 1.9520
            v_values["V3"] = -1.6099
            v_values["V4"] = 3.9979
            v_values["V14"] = -4.2893
            v_values["V10"] = -2.7723
            v_values["V12"] = -2.8999

        with st.form("single_transaction_form"):
            col_amt, col_time = st.columns(2)
            with col_amt:
                amount_input = st.number_input("Transaction Amount ($)", value=float(default_amount), min_value=0.0, step=1.0)
            with col_time:
                time_input = st.number_input("Transaction Time (Seconds)", value=float(default_time), min_value=0.0, step=1.0)

            with st.expander("Anonymized PCA Features (V1 to V28)", expanded=False):
                st.caption("28 principal component features derived to protect user privacy. Defaults are 0.0.")
                cols_v = st.columns(4)
                input_v = {}
                for idx, v_name in enumerate([f"V{i}" for i in range(1, 29)]):
                    col_target = cols_v[idx % 4]
                    val = v_values.get(v_name, 0.0)
                    input_v[v_name] = col_target.number_input(
                        v_name,
                        value=float(val),
                        step=0.1,
                        format="%.4f"
                    )

            submitted = st.form_submit_button("🚀 Analyze Transaction", use_container_width=True)

        if submitted:
            # Construct payload
            tx_payload = {
                "Time": float(time_input),
                "Amount": float(amount_input),
                "threshold": float(threshold)
            }
            tx_payload.update({k: float(v) for k, v in input_v.items()})

            if inference_engine.startswith("Kubernetes"):
                # Strict validation: Do not silently switch when Kubernetes is offline
                if not K8S_API_BASE_URL or not k8s_status.get("online"):
                    st.error(
                        "🚨 **Kubernetes Microservice is OFFLINE.**\n\n"
                        f"Cannot connect to `{K8S_API_BASE_URL or 'unconfigured'}/predict`: {k8s_status.get('error')}.\n\n"
                        "To use the live cluster, ensure port-forwarding is running:\n"
                        "```bash\nkubectl port-forward svc/nexvia-fraud-api-service 8000:8000\n```\n"
                        "Or manually switch the **Inference Engine** in the sidebar to **Local In-Memory Pipeline**."
                    )
                else:
                    try:
                        start_time = time.perf_counter()
                        result = predict_via_k8s_single(tx_payload, base_url=K8S_API_BASE_URL)
                        total_latency_ms = (time.perf_counter() - start_time) * 1000

                        prob = float(result["fraud_probability"])
                        pred_class = int(result["predicted_class"])
                        is_fraud = pred_class == 1
                        risk_label = str(result["risk_label"])
                        status = str(result["status"])
                        server_latency_ms = result.get("inference_latency_ms", total_latency_ms)

                        st.markdown("---")
                        st.subheader("Inference Result")

                        if is_fraud:
                            st.markdown(
                                f"""
                                <div class="status-flagged">
                                    🚨 <strong>TRANSACTION FLAGGED FOR REVIEW (Status: {status})</strong><br>
                                    High fraud likelihood detected. Payment should be held or sent for multi-factor authentication.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            try:
                                tx_case_id = f"TX-{int(time.time()*1000)}"
                                record_case(
                                    transaction_id=tx_case_id,
                                    amount=float(amount_input),
                                    fraud_probability=float(prob),
                                    risk_label=risk_label,
                                    model_decision="FLAG",
                                    case_status="New",
                                    analyst_notes="Auto-flagged during real-time transaction risk assessment via Kubernetes API."
                                )
                                st.info(f"📋 Alert registered into Analyst Review Queue as case **`{tx_case_id}`**.")
                            except Exception as rec_err:
                                st.caption(f"Note: Could not queue alert: {rec_err}")
                        else:
                            st.markdown(
                                f"""
                                <div class="status-approved">
                                    ✅ <strong>TRANSACTION APPROVED (Status: {status})</strong><br>
                                    Normal transaction behavior detected. Low risk of fraudulent activity.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.write("")
                        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
                        col_r1.metric("Fraud Probability", f"{prob*100:.2f}%", help="Model estimated risk of fraud")
                        col_r2.metric("Prediction Class", f"{pred_class} ({'Fraud' if is_fraud else 'Legitimate'})", help="0 = Legitimate, 1 = Fraud")
                        col_r3.metric("Risk Label", risk_label, help="Categorical risk designation")
                        col_r4.metric("Approval Status", status, help="Decision outcome: APPROVED or FLAGGED_FOR_REVIEW")
                        col_r5.metric("Inference Latency", f"{server_latency_ms:.2f} ms", f"Roundtrip: {total_latency_ms:.1f}ms")

                        st.caption(f"⚡ **Inference Backend:** Kubernetes Microservice Pod (`{K8S_API_BASE_URL}/predict`) | Decision Threshold: `{threshold:.2f}`")

                    except Exception as err:
                        st.error(f"Kubernetes Prediction Error: {err}")

            else:
                # Manual Local In-Memory Pipeline fallback
                if predictor is None:
                    st.error("Cannot score transactions: Local model artifact is missing from `models/`.")
                else:
                    try:
                        start_time = time.perf_counter()
                        local_dict = {k: v for k, v in tx_payload.items() if k != "threshold"}
                        result = predictor.predict(local_dict, threshold=threshold)
                        latency_ms = (time.perf_counter() - start_time) * 1000

                        prob = float(result["fraud_probability"])
                        pred_class = int(result["predicted_class"])
                        is_fraud = pred_class == 1
                        risk_label = str(result["risk_label"])
                        status = str(result["status"])

                        st.markdown("---")
                        st.subheader("Inference Result")

                        if is_fraud:
                            st.markdown(
                                f"""
                                <div class="status-flagged">
                                    🚨 <strong>TRANSACTION FLAGGED FOR REVIEW (Status: {status})</strong><br>
                                    High fraud likelihood detected. Payment should be held or sent for multi-factor authentication.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            try:
                                tx_case_id = f"TX-{int(time.time()*1000)}"
                                record_case(
                                    transaction_id=tx_case_id,
                                    amount=float(amount_input),
                                    fraud_probability=float(prob),
                                    risk_label=risk_label,
                                    model_decision="FLAG",
                                    case_status="New",
                                    analyst_notes="Auto-flagged during real-time transaction risk assessment via Local Pipeline."
                                )
                                st.info(f"📋 Alert registered into Analyst Review Queue as case **`{tx_case_id}`**.")
                            except Exception as rec_err:
                                st.caption(f"Note: Could not queue alert: {rec_err}")
                        else:
                            st.markdown(
                                f"""
                                <div class="status-approved">
                                    ✅ <strong>TRANSACTION APPROVED (Status: {status})</strong><br>
                                    Normal transaction behavior detected. Low risk of fraudulent activity.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.write("")
                        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
                        col_r1.metric("Fraud Probability", f"{prob*100:.2f}%", help="Model estimated risk of fraud")
                        col_r2.metric("Prediction Class", f"{pred_class} ({'Fraud' if is_fraud else 'Legitimate'})", help="0 = Legitimate, 1 = Fraud")
                        col_r3.metric("Risk Label", risk_label, help="Categorical risk designation")
                        col_r4.metric("Approval Status", status, help="Decision outcome: APPROVED or FLAGGED_FOR_REVIEW")
                        col_r5.metric("Inference Latency", f"{latency_ms:.2f} ms")

                        st.caption(f"💻 **Inference Backend:** Local In-Memory Pipeline (`models/xgboost_pipeline.joblib`) | Decision Threshold: `{threshold:.2f}`")

                    except Exception as err:
                        st.error(f"Local Prediction Error: {err}")


# -----------------------------------------------------------------------------
# Tab 2: Batch Transaction Screening
# -----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Transaction Screening")
    st.markdown("Upload a CSV file containing multiple transactions to perform high-throughput fraud screening.")

    uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"], help="Must contain Time, V1-V28, and Amount")

    col_b1, col_b2 = st.columns([1, 3])
    with col_b1:
        load_demo_batch = st.button("📋 Load Sample Batch (10 Transactions)")

    df_to_score = None

    if uploaded_file is not None:
        try:
            df_to_score = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file loaded: **{len(df_to_score)}** transactions.")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    elif load_demo_batch:
        if DATA_PATH.exists():
            try:
                # Load a mix of legitimate and fraud rows from creditcard.csv
                raw_sample = pd.read_csv(DATA_PATH, nrows=1000)
                fraud_rows = raw_sample[raw_sample["Class"] == 1]
                legit_rows = raw_sample[raw_sample["Class"] == 0].iloc[: 10 - len(fraud_rows)]
                df_to_score = pd.concat([legit_rows, fraud_rows]).sample(frac=1.0, random_state=42).reset_index(drop=True)
                if "Class" in df_to_score.columns:
                    df_to_score = df_to_score.drop("Class", axis=1)
                st.info("Loaded 10 sample transactions (containing both legitimate and fraudulent records) from dataset.")
            except Exception as e:
                st.warning(f"Could not load sample from raw data: {e}")

        # Fallback to reference_data.parquet if creditcard.csv is not present (e.g. on Streamlit Cloud)
        if df_to_score is None and REFERENCE_DATA_PATH.exists():
            try:
                ref_df = pd.read_parquet(REFERENCE_DATA_PATH)
                target_col = "target" if "target" in ref_df.columns else "Class" if "Class" in ref_df.columns else None
                if target_col:
                    fraud_rows = ref_df[ref_df[target_col] == 1].head(3)
                    legit_rows = ref_df[ref_df[target_col] == 0].head(7)
                    df_to_score = pd.concat([legit_rows, fraud_rows]).sample(frac=1.0, random_state=42).reset_index(drop=True)
                else:
                    df_to_score = ref_df.head(10).copy()
                drop_cols = [c for c in ["target", "Class", "prediction", "probability"] if c in df_to_score.columns]
                if drop_cols:
                    df_to_score = df_to_score.drop(columns=drop_cols)
                st.info("Loaded 10 sample transactions (containing both legitimate and fraudulent records) from reference dataset.")
            except Exception as e:
                st.warning(f"Could not load sample from reference data: {e}")

        if df_to_score is None:
            # Synthetic batch fallback
            records = []
            for i in range(10):
                row = {f"V{k}": 0.0 for k in range(1, 29)}
                row["Time"] = float(i * 15)
                row["Amount"] = float(25.0 + i * 10.0)
                records.append(row)
            df_to_score = pd.DataFrame(records)
            st.info("Generated 10 synthetic transactions for demonstration.")

    if df_to_score is not None:
        scored_df = None
        batch_latency_ms = 0.0

        if inference_engine.startswith("Kubernetes"):
            # Strict validation: Do not silently switch when Kubernetes is offline or unconfigured
            if not K8S_API_BASE_URL or not k8s_status.get("online"):
                st.error(
                    "🚨 **Kubernetes Microservice is OFFLINE.**\n\n"
                    f"Cannot connect to `{K8S_API_BASE_URL or 'unconfigured'}/predict/batch`: {k8s_status.get('error')}.\n\n"
                    "To use the live cluster, ensure port-forwarding is running:\n"
                    "```bash\nkubectl port-forward svc/nexvia-fraud-api-service 8000:8000\n```\n"
                    "Or manually switch the **Inference Engine** in the sidebar to **Local In-Memory Pipeline**."
                )
            else:
                try:
                    with st.spinner("Scoring batch transactions via Kubernetes microservice..."):
                        start_batch = time.perf_counter()
                        records_payload = df_to_score[EXPECTED_FEATURES].to_dict(orient="records")
                        k8s_resp = predict_via_k8s_batch(records_payload, threshold=threshold, base_url=K8S_API_BASE_URL)
                        scored_df = pd.DataFrame(k8s_resp.get("results", []))
                        batch_latency_ms = (time.perf_counter() - start_batch) * 1000
                except Exception as batch_err:
                    st.error(f"Kubernetes Batch Scoring Error: {batch_err}")
        else:
            # Local In-Memory Pipeline explicitly selected
            if predictor is None:
                st.error("Cannot score batch: Local model artifact is missing from `models/`.")
            else:
                try:
                    with st.spinner("Scoring batch transactions via Local Pipeline..."):
                        start_batch = time.perf_counter()
                        scored_df = predictor.predict(df_to_score, threshold=threshold)
                        batch_latency_ms = (time.perf_counter() - start_batch) * 1000
                except ValueError as val_err:
                    st.error(f"Validation Error: {val_err}")
                except Exception as batch_err:
                    st.error(f"Batch Scoring Error: {batch_err}")

        if scored_df is not None:
            total_tx = len(scored_df)
            fraud_count = int((scored_df["predicted_class"] == 1).sum())
            legit_count = total_tx - fraud_count

            st.markdown("---")
            st.subheader("Batch Screening Summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Scored", f"{total_tx:,}")
            c2.metric("Approved (Legitimate)", f"{legit_count:,}", f"{(legit_count/total_tx)*100:.1f}%")
            c3.metric("Flagged (Fraud Alert)", f"{fraud_count:,}", f"{(fraud_count/total_tx)*100:.1f}%", delta_color="inverse")
            c4.metric("Throughput", f"{batch_latency_ms:.1f} ms", f"{batch_latency_ms/total_tx:.2f} ms/tx")

            st.subheader("Scored Results Table")
            display_cols = ["Time", "Amount", "fraud_probability", "predicted_class", "risk_label"]
            available_cols = [c for c in display_cols if c in scored_df.columns]
            st.dataframe(
                scored_df[available_cols].style.format({
                    "Amount": "${:.2f}",
                    "fraud_probability": "{:.4%}",
                    "Time": "{:.0f}"
                }),
                use_container_width=True
            )

            # CSV Download Button
            csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Scored Results (CSV)",
                data=csv_bytes,
                file_name="nexvia_fraud_scored_results.csv",
                mime="text/csv",
                use_container_width=True
            )


# -----------------------------------------------------------------------------
# Tab 3: Model Benchmark & Audit
# -----------------------------------------------------------------------------
with tab_audit:
    st.subheader("Model Benchmark & Evaluation Audit")
    st.markdown("Comparison between the baseline Logistic Regression model and the production XGBoost classifier.")

    comparison_data = [
        {"Metric": "PR-AUC (Primary Evaluation Metric)", "Logistic Regression Baseline": "0.718971", "XGBoost Classifier": "0.868619", "Improvement": "+0.1496 (+20.8%)"},
        {"Metric": "Fraud Precision (Class 1)", "Logistic Regression Baseline": "6.1%", "XGBoost Classifier": "77.6%", "Improvement": "+71.5% (+1172%)"},
        {"Metric": "Fraud Recall (Class 1)", "Logistic Regression Baseline": "91.8%", "XGBoost Classifier": "84.7%", "Improvement": "-7.1%"},
        {"Metric": "Fraud F1-Score", "Logistic Regression Baseline": "0.1144", "XGBoost Classifier": "0.8098", "Improvement": "+0.6954 (+608%)"},
        {"Metric": "False Positives (Legitimate Flagged)", "Logistic Regression Baseline": "1,386", "XGBoost Classifier": "24", "Improvement": "-1,362 (-98.3%)"},
        {"Metric": "ROC-AUC", "Logistic Regression Baseline": "0.972083", "XGBoost Classifier": "0.979963", "Improvement": "+0.0079"}
    ]
    st.table(pd.DataFrame(comparison_data))

    st.subheader("Saved Evaluation Visualizations")
    col_img1, col_img2 = st.columns(2)

    cm_img_path = OUTPUTS_DIR / "xgboost_confusion_matrix.png"
    pr_img_path = OUTPUTS_DIR / "xgboost_precision_recall_curve.png"

    with col_img1:
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="XGBoost Confusion Matrix (Test Set: 56,962 transactions)")
        else:
            st.caption("Confusion matrix image not found in outputs/.")

    with col_img2:
        if pr_img_path.exists():
            st.image(str(pr_img_path), caption="XGBoost Precision-Recall Curve (PR-AUC = 0.8686)")
        else:
            st.caption("PR curve image not found in outputs/.")


# -----------------------------------------------------------------------------
# Tab 4: Evidently AI Data-Drift Monitoring
# -----------------------------------------------------------------------------
with tab_drift:
    st.subheader("Evidently AI Data-Drift Monitoring")
    st.markdown(
        "Detects distribution drift between baseline reference transactions (`data/reference_data.parquet`) "
        "and production scoring batches using Kolmogorov-Smirnov and Wasserstein distance tests."
    )

    drift_json_path = OUTPUTS_DIR / "drift_metrics.json"
    drift_html_path = OUTPUTS_DIR / "drift_report.html"

    # Display KPI cards if metrics exist
    if drift_json_path.exists():
        try:
            with open(drift_json_path, "r", encoding="utf-8") as f:
                d_metrics = json.load(f)

            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            is_drifted = d_metrics.get("dataset_drift", False)

            with col_d1:
                if is_drifted:
                    st.error("🚨 Status: **DRIFT DETECTED**")
                else:
                    st.success("✅ Status: **DATASET STABLE**")

            with col_d2:
                st.metric("Drifted Features", f"{d_metrics.get('number_of_drifted_columns', 0)} / {d_metrics.get('number_of_columns', 30)}")

            with col_d3:
                st.metric("Drift Share", f"{d_metrics.get('drift_share', 0.0)*100:.1f}%")

            with col_d4:
                st.metric("Evaluation Sample", f"{d_metrics.get('current_rows', 'N/A')} rows")

            if "timestamp" in d_metrics:
                st.caption(f"Last Evaluated: **{d_metrics['timestamp']}** | Test: **{d_metrics.get('stat_test', 'KS')}**")

        except Exception as read_err:
            st.warning(f"Could not read drift metrics: {read_err}")
    else:
        st.info("ℹ️ No drift evaluation metrics recorded yet. Trigger an evaluation below or via the REST API.")

    # Action buttons
    col_act1, col_act2 = st.columns([1, 2])
    with col_act1:
        if st.button("⚡ Run Drift Check (Test Set Sample)"):
            with st.spinner("Executing Evidently drift analysis in isolated environment..."):
                try:
                    from src.drift import drift_monitor
                    sample_eval = None
                    # Load a test sample from raw dataset or reference dataset
                    if DATA_PATH.exists():
                        sample_raw = pd.read_csv(DATA_PATH, nrows=500)
                        sample_eval = sample_raw[EXPECTED_FEATURES]
                    elif REFERENCE_DATA_PATH.exists():
                        sample_raw = pd.read_parquet(REFERENCE_DATA_PATH).tail(500)
                        sample_eval = sample_raw[EXPECTED_FEATURES]

                    if sample_eval is not None:
                        eval_res = drift_monitor.evaluate_drift(current_data=sample_eval)
                        if eval_res.get("status") == "success":
                            st.success("✅ Drift evaluation completed successfully!")
                            st.rerun()
                        else:
                            st.error(f"Evaluation error: {eval_res.get('error', eval_res.get('message'))}")
                    else:
                        st.error("No dataset available to generate sample.")
                except Exception as eval_exc:
                    st.error(f"Failed to run drift evaluation: {eval_exc}")

    # Interactive Evidently HTML report embed
    if drift_html_path.exists():
        st.markdown("---")
        st.subheader("Interactive Evidently AI Drift Report")
        try:
            with open(drift_html_path, "r", encoding="utf-8") as f:
                html_body = f.read()
            components.html(html_body, height=750, scrolling=True)
        except Exception as html_err:
            st.warning(f"Could not display HTML report: {html_err}")
    else:
        st.caption("Interactive HTML report will appear here once generated.")


# -----------------------------------------------------------------------------
# Tab 5: Analyst Case Review & Feedback
# -----------------------------------------------------------------------------
with tab_feedback:
    st.subheader("📋 Analyst Case Review & Feedback Loop")
    st.markdown(
        "Manage alerted transactions, conduct human-in-the-loop review, record verified "
        "investigation outcomes, and persist ground-truth labels for drift analysis and retraining."
    )

    # Summary metric cards
    summary = get_case_summary()
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    col_s1.metric("Total Cases", summary.get("Total", 0))
    col_s2.metric("New Alerts", summary.get("New", 0))
    col_s3.metric("Under Review", summary.get("Under Review", 0))
    col_s4.metric("Confirmed Fraud", summary.get("Confirmed Fraud", 0))
    col_s5.metric("Legitimate", summary.get("Legitimate", 0))

    st.markdown("---")

    col_flt, col_refresh = st.columns([3, 1])
    with col_flt:
        status_options = ["All"] + sorted(list(VALID_STATUSES))
        selected_filter = st.selectbox("Filter Cases by Investigation Status", status_options, index=0)
    with col_refresh:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Cases", use_container_width=True):
            st.rerun()

    filter_arg = None if selected_filter == "All" else selected_filter
    cases_list = get_all_cases(status_filter=filter_arg, limit=100)

    if not cases_list:
        st.info(f"ℹ️ No cases found matching filter: **{selected_filter}**. Test transactions flagged in Tab 1 will appear here.")
    else:
        df_cases = pd.DataFrame(cases_list)
        # Format display columns
        display_cols = [
            "id", "transaction_id", "timestamp", "amount",
            "fraud_probability", "risk_label", "model_decision",
            "case_status", "analyst_notes", "analyst_id", "updated_at"
        ]
        available_cols = [c for c in display_cols if c in df_cases.columns]
        df_display = df_cases[available_cols].copy()
        if "fraud_probability" in df_display.columns:
            df_display["fraud_probability"] = df_display["fraud_probability"].apply(
                lambda p: f"{float(p)*100:.2f}%" if pd.notnull(p) else "N/A"
            )
        if "amount" in df_display.columns:
            df_display["amount"] = df_display["amount"].apply(
                lambda a: f"${float(a):,.2f}" if pd.notnull(a) else "N/A"
            )

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Update Case Status & Notes")

        # Select case to update
        case_options = {
            f"Case #{c['id']} | TX: {c['transaction_id']} | Risk: {c['risk_label']} ({float(c['fraud_probability'])*100:.1f}%) | Current: {c['case_status']}": c
            for c in cases_list
        }
        selected_case_label = st.selectbox("Select Case to Investigate", list(case_options.keys()))
        selected_case = case_options[selected_case_label]

        with st.form("update_case_form"):
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                current_st = selected_case.get("case_status", "New")
                target_status_list = sorted(list(VALID_STATUSES))
                def_idx = target_status_list.index(current_st) if current_st in target_status_list else 0
                new_status = st.selectbox("Updated Case Status", target_status_list, index=def_idx)
            with col_u2:
                analyst_id = st.text_input("Analyst Identifier", value=selected_case.get("analyst_id") or "analyst_1")

            existing_notes = selected_case.get("analyst_notes", "") or ""
            analyst_notes = st.text_area("Investigation Notes & Justification", value=existing_notes, placeholder="Enter reason for fraud confirmation or false positive override...")

            submitted_update = st.form_submit_button("💾 Save Case Decision", use_container_width=True)

            if submitted_update:
                success = update_case_status(
                    case_id=selected_case["id"],
                    case_status=new_status,
                    analyst_notes=analyst_notes,
                    analyst_id=analyst_id
                )
                if success:
                    st.success(f"✅ Case #{selected_case['id']} updated to **{new_status}**.")
                    st.rerun()
                else:
                    st.error("Failed to update case.")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Developed for Real-Time Payment Security")
