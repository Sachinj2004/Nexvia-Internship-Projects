"""
Real-Time Financial Fraud Detection Platform (AIML-01)
FastAPI REST Microservice: src/api.py

Exposes high-throughput, sub-10ms credit card fraud scoring endpoints
backed by the pre-trained XGBoost pipeline (StandardScaler + XGBClassifier).
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    DEFAULT_MODEL_PATH,
    EXPECTED_FEATURES,
    FraudPredictor,
    predict_batch,
    predict_single,
)
from src.feature_store import get_feast_features
from src.drift import drift_monitor

METRICS_PATH = PROJECT_ROOT / "outputs" / "xgboost_metrics.json"


# -----------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# -----------------------------------------------------------------------------
class TransactionPayload(BaseModel):
    """30 required features for credit card transaction scoring."""
    Time: float = Field(..., description="Seconds elapsed since the initial transaction in the dataset")
    V1: float = Field(..., description="PCA anonymized feature V1")
    V2: float = Field(..., description="PCA anonymized feature V2")
    V3: float = Field(..., description="PCA anonymized feature V3")
    V4: float = Field(..., description="PCA anonymized feature V4")
    V5: float = Field(..., description="PCA anonymized feature V5")
    V6: float = Field(..., description="PCA anonymized feature V6")
    V7: float = Field(..., description="PCA anonymized feature V7")
    V8: float = Field(..., description="PCA anonymized feature V8")
    V9: float = Field(..., description="PCA anonymized feature V9")
    V10: float = Field(..., description="PCA anonymized feature V10")
    V11: float = Field(..., description="PCA anonymized feature V11")
    V12: float = Field(..., description="PCA anonymized feature V12")
    V13: float = Field(..., description="PCA anonymized feature V13")
    V14: float = Field(..., description="PCA anonymized feature V14")
    V15: float = Field(..., description="PCA anonymized feature V15")
    V16: float = Field(..., description="PCA anonymized feature V16")
    V17: float = Field(..., description="PCA anonymized feature V17")
    V18: float = Field(..., description="PCA anonymized feature V18")
    V19: float = Field(..., description="PCA anonymized feature V19")
    V20: float = Field(..., description="PCA anonymized feature V20")
    V21: float = Field(..., description="PCA anonymized feature V21")
    V22: float = Field(..., description="PCA anonymized feature V22")
    V23: float = Field(..., description="PCA anonymized feature V23")
    V24: float = Field(..., description="PCA anonymized feature V24")
    V25: float = Field(..., description="PCA anonymized feature V25")
    V26: float = Field(..., description="PCA anonymized feature V26")
    V27: float = Field(..., description="PCA anonymized feature V27")
    V28: float = Field(..., description="PCA anonymized feature V28")
    Amount: float = Field(..., ge=0.0, description="Transaction monetary amount")
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Optional custom decision cutoff threshold (0.0 to 1.0)"
    )
    transaction_id: Optional[int] = Field(
        None, description="Optional transaction identifier for Feast velocity feature enrichment"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Time": 406.0,
                "V1": -2.3122, "V2": 1.9520, "V3": -1.6099, "V4": 3.9979,
                "V5": -0.5222, "V6": -1.4265, "V7": -2.5374, "V8": 1.3917,
                "V9": -2.7701, "V10": -2.7723, "V11": 3.2020, "V12": -2.8999,
                "V13": -0.5952, "V14": -4.2893, "V15": 0.3898, "V16": -1.1407,
                "V17": -2.8301, "V18": -0.0155, "V19": 0.4161, "V20": 0.1269,
                "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202,
                "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1433,
                "Amount": 0.0,
                "threshold": 0.5
            }
        }
    }


class PredictionResponse(BaseModel):
    """Individual transaction prediction response."""
    fraud_probability: float = Field(..., description="Estimated probability of fraud (0.0 to 1.0)")
    predicted_class: int = Field(..., description="0 for Legitimate, 1 for Fraud")
    risk_label: str = Field(..., description="'Legitimate' or 'Fraud'")
    threshold_used: float = Field(..., description="Decision threshold applied")
    status: str = Field(..., description="'APPROVED' or 'FLAGGED_FOR_REVIEW'")
    inference_latency_ms: float = Field(..., description="Execution latency in milliseconds")
    timestamp: str = Field(..., description="UTC ISO timestamp of prediction")
    transaction_id: Optional[int] = Field(None, description="Transaction identifier if supplied")
    enriched_features: Optional[Dict[str, Any]] = Field(
        None, description="Enriched velocity features from Feast online store"
    )


class BatchTransactionPayload(BaseModel):
    """Batch scoring payload containing list of transaction payloads."""
    transactions: List[TransactionPayload] = Field(
        ..., min_length=1, max_length=10000, description="List of transactions to score (max 10,000)"
    )
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Global threshold applied across batch (overrides individual)"
    )


class BatchPredictionResponse(BaseModel):
    """Batch scoring aggregated summary and itemized results."""
    total_transactions: int
    approved_count: int
    flagged_count: int
    fraud_rate_percent: float
    threshold_used: float
    batch_latency_ms: float
    latency_per_tx_ms: float
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health and readiness status response."""
    status: str
    model_loaded: bool
    model_type: str
    artifact_path: str
    uptime_seconds: float
    memory_rss_mb: float
    platform_info: str


class ModelInfoResponse(BaseModel):
    """Model metadata, schema specifications, and evaluation benchmarks."""
    model_name: str
    artifact_path: str
    feature_count: int
    expected_features: List[str]
    default_threshold: float
    benchmarks: Dict[str, Any]


# -----------------------------------------------------------------------------
# Lifespan Management (Warmup and Resource Cache)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler managing startup initialization and model warm-up."""
    app.state.start_time = time.time()
    try:
        predictor = FraudPredictor(model_path=DEFAULT_MODEL_PATH)
        # Model warm-up: score a synthetic baseline to eliminate cold-start latency
        warmup_payload = {col: 0.0 for col in EXPECTED_FEATURES}
        warmup_payload["Time"] = 0.0
        warmup_payload["Amount"] = 1.0
        predictor.predict(warmup_payload)

        app.state.predictor = predictor
        app.state.model_loaded = True
        app.state.load_error = None
    except Exception as exc:
        app.state.predictor = None
        app.state.model_loaded = False
        app.state.load_error = str(exc)

    yield


# -----------------------------------------------------------------------------
# FastAPI Application Setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Nexvia Real-Time Financial Fraud Detection API",
    description=(
        "High-performance REST API for real-time credit card fraud scoring. "
        "Engineered for sub-10ms latency using production XGBoost classifier, "
        "StandardScaler preprocessing, and strict schema validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_active_predictor() -> FraudPredictor:
    """Retrieve loaded predictor or raise 503 if unavailable."""
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        # Fallback attempt if lifespan was bypassed (e.g. direct import in tests)
        try:
            predictor = FraudPredictor(model_path=DEFAULT_MODEL_PATH)
            app.state.predictor = predictor
            app.state.model_loaded = True
            return predictor
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Fraud inference engine is offline: {exc}"
            )
    return predictor


def get_memory_usage_mb() -> float:
    """Get resident set size in megabytes using standard library resource."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, Linux returns kilobytes
        if platform.system() == "Darwin":
            return round(usage / (1024.0 * 1024.0), 2)
        return round(usage / 1024.0, 2)
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["Overview"])
async def root() -> Dict[str, Any]:
    """Root endpoint providing platform description and API navigation links."""
    return {
        "service": "Nexvia Real-Time Financial Fraud Detection Platform (AIML-01)",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "drift_status": "/drift/status",
            "drift_report": "/drift/report",
            "drift_evaluate": "/drift/evaluate"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """
    Kubernetes readiness & liveness probe.
    Returns model availability, system uptime, and memory usage.
    """
    start_time = getattr(app.state, "start_time", time.time())
    uptime = time.time() - start_time
    is_loaded = getattr(app.state, "model_loaded", False)

    # Check if predictor can be resolved
    if not is_loaded:
        try:
            get_active_predictor()
            is_loaded = True
        except Exception:
            is_loaded = False

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_type="XGBoost Classifier (StandardScaler + XGBClassifier)",
        artifact_path=str(DEFAULT_MODEL_PATH),
        uptime_seconds=round(uptime, 2),
        memory_rss_mb=get_memory_usage_mb(),
        platform_info=f"{platform.system()} {platform.machine()}"
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Metadata"])
async def model_info() -> ModelInfoResponse:
    """
    Detailed metadata about the loaded model pipeline, expected schema, and benchmark results.
    """
    benchmarks: Dict[str, Any] = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                benchmarks = json.load(f)
        except Exception:
            benchmarks = {"warning": "Could not read metrics file."}

    return ModelInfoResponse(
        model_name="XGBoost Production Fraud Detection Pipeline",
        artifact_path=str(DEFAULT_MODEL_PATH),
        feature_count=len(EXPECTED_FEATURES),
        expected_features=EXPECTED_FEATURES,
        default_threshold=0.50,
        benchmarks=benchmarks
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def score_transaction(payload: TransactionPayload) -> PredictionResponse:
    """
    Score a single credit card transaction in real time with sub-10ms target latency.
    """
    predictor = get_active_predictor()

    raw_dict = payload.model_dump()
    cutoff_threshold = raw_dict.pop("threshold", None)
    tx_id = raw_dict.pop("transaction_id", None)
    if cutoff_threshold is None:
        cutoff_threshold = predictor.default_threshold

    start_perf = time.perf_counter()
    try:
        result = predictor.predict(raw_dict, threshold=cutoff_threshold)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Schema validation error: {val_err}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}"
        )

    latency_ms = (time.perf_counter() - start_perf) * 1000.0

    # Optional Feast feature store enrichment (does not block or fail prediction)
    enriched_features = None
    if tx_id is not None:
        try:
            enriched_features = get_feast_features(tx_id)
        except Exception:
            enriched_features = None

    # Record transaction to drift monitoring buffer (thread-safe, non-blocking)
    try:
        drift_monitor.record_transaction(raw_dict)
    except Exception:
        pass

    return PredictionResponse(
        fraud_probability=result["fraud_probability"],
        predicted_class=result["predicted_class"],
        risk_label=result["risk_label"],
        threshold_used=result["threshold_used"],
        status=result["status"],
        inference_latency_ms=round(latency_ms, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
        transaction_id=tx_id,
        enriched_features=enriched_features
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def score_batch(payload: BatchTransactionPayload) -> BatchPredictionResponse:
    """
    High-throughput vectorized scoring of a batch of transaction payloads.
    """
    predictor = get_active_predictor()

    if not payload.transactions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transaction batch cannot be empty."
        )

    # Extract payloads and thresholds
    raw_list: List[Dict[str, Any]] = []
    tx_ids: List[Optional[int]] = []
    for item in payload.transactions:
        d = item.model_dump()
        d.pop("threshold", None)
        tx_ids.append(d.pop("transaction_id", None))
        raw_list.append(d)

    global_threshold = payload.threshold if payload.threshold is not None else predictor.default_threshold

    start_perf = time.perf_counter()
    try:
        scored_df = predictor.predict(raw_list, threshold=global_threshold)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch validation error: {val_err}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {exc}"
        )

    total_time_ms = (time.perf_counter() - start_perf) * 1000.0
    total_tx = len(scored_df)
    latency_per_tx = total_time_ms / total_tx if total_tx > 0 else 0.0

    current_time_str = datetime.now(timezone.utc).isoformat()
    results: List[PredictionResponse] = []
    flagged_count = 0

    for idx, row in scored_df.iterrows():
        p_class = int(row["predicted_class"])
        if p_class == 1:
            flagged_count += 1

        row_tx_id = tx_ids[idx] if idx < len(tx_ids) else None
        row_enriched = None
        if row_tx_id is not None:
            try:
                row_enriched = get_feast_features(row_tx_id)
            except Exception:
                row_enriched = None

        results.append(
            PredictionResponse(
                fraud_probability=float(row["fraud_probability"]),
                predicted_class=p_class,
                risk_label=str(row["risk_label"]),
                threshold_used=float(global_threshold),
                status="FLAGGED_FOR_REVIEW" if p_class == 1 else "APPROVED",
                inference_latency_ms=round(latency_per_tx, 3),
                timestamp=current_time_str,
                transaction_id=row_tx_id,
                enriched_features=row_enriched
            )
        )

    approved_count = total_tx - flagged_count
    fraud_rate = (flagged_count / total_tx) * 100.0 if total_tx > 0 else 0.0

    # Record batch to drift monitoring buffer (thread-safe, non-blocking)
    try:
        drift_monitor.record_batch(raw_list)
    except Exception:
        pass

    return BatchPredictionResponse(
        total_transactions=total_tx,
        approved_count=approved_count,
        flagged_count=flagged_count,
        fraud_rate_percent=round(fraud_rate, 2),
        threshold_used=float(global_threshold),
        batch_latency_ms=round(total_time_ms, 3),
        latency_per_tx_ms=round(latency_per_tx, 4),
        results=results
    )


# -----------------------------------------------------------------------------
# Evidently AI Data Drift Monitoring Endpoints
# -----------------------------------------------------------------------------
@app.get("/drift/status", tags=["Model Monitoring"])
async def get_drift_status() -> Dict[str, Any]:
    """
    Return current Evidently AI data-drift metrics, drifted feature counts,
    and sliding window transaction buffer statistics.
    """
    return drift_monitor.get_latest_metrics()


@app.post("/drift/evaluate", tags=["Model Monitoring"])
async def trigger_drift_evaluation(
    sample_size: int = Query(15000, ge=100, le=50000, description="Max reference samples for evaluation")
) -> Dict[str, Any]:
    """
    Trigger on-demand Evidently AI data-drift analysis comparing buffered transactions
    against the baseline reference dataset.
    """
    return drift_monitor.evaluate_drift(sample_size=sample_size)


@app.get("/drift/report", response_class=HTMLResponse, tags=["Model Monitoring"])
async def get_drift_report() -> HTMLResponse:
    """
    Serve the latest interactive Evidently AI HTML data-drift report.
    """
    report_file = drift_monitor.output_html
    if not report_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Drift report HTML not found. Trigger a drift evaluation first via /drift/evaluate."
        )
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load drift report HTML: {exc}"
        )


# -----------------------------------------------------------------------------
# Direct Execution Runner
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting Nexvia Fraud Detection REST API...")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
