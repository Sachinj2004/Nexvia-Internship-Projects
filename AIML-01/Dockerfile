# =============================================================================
# Nexvia Real-Time Financial Fraud Detection Platform (AIML-01)
# Production Containerfile: FastAPI Fraud Detection Microservice
# =============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install minimal system dependencies:
# - libgomp1: OpenMP runtime required by XGBoost and Scikit-Learn on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install strictly required runtime dependencies for FastAPI XGBoost inference
RUN pip install --no-cache-dir \
    fastapi==0.160.0 \
    "uvicorn[standard]==0.54.0" \
    pydantic==2.14.1 \
    pandas==3.1.0 \
    numpy==2.4.6 \
    scikit-learn==1.9.1 \
    joblib==1.6.0 \
    xgboost==3.2.0

# Copy application source code and pre-trained XGBoost model artifact
COPY src/ /app/src/
COPY models/xgboost_pipeline.joblib /app/models/xgboost_pipeline.joblib

# Enforce least privilege: Run as dedicated non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose internal microservice port
EXPOSE 8000

# Container health check testing FastAPI /health using Python standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# Production entry point running high-performance Uvicorn ASGI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
