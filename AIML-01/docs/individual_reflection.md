# Nexvia AIML-01: Technical Retrospective & Engineering Reflection

**Author**: Machine Learning Engineering Intern  
**Project**: Nexvia Real-Time Financial Fraud Detection Platform (AIML-01)  
**Date**: October 2026  

---

## 1. Executive Summary

Building the Nexvia Real-Time Financial Fraud Detection Platform has been an intensive exercise in applied machine learning engineering. The objective was not merely to fit an algorithm to a CSV file, but to design a resilient, sub-10ms target design pipeline (measured at 2.8–4.2ms on local Apple M1) capable of ingestion, feature caching, multi-model scoring, automated drift detection, controlled retraining safeguards, and human-in-the-loop analyst feedback.

This retrospective documents the key engineering challenges encountered, the technical trade-offs evaluated, the architectural decisions implemented, and the genuine lessons learned when theory meets the realities of production systems.

---

## 2. Technical Challenges & Engineering Solutions

### Challenge 1: Extreme Class Imbalance (0.172% Fraud Rate)
* **The Problem**: In a dataset of 284,807 transactions, only 492 are fraudulent. Standard loss functions and naive accuracy metrics fail catastrophically: a dumb model predicting 100% legitimate transactions achieves 99.828% accuracy while letting every fraudulent dollar escape. Moreover, evaluating models using ROC-AUC gives a misleadingly optimistic score because the overwhelming number of true negatives dominates the false positive rate denominator ($FPR = FP / (FP + TN)$).
* **Engineering Solution**:
  1. We selected **Precision-Recall AUC (PR-AUC / Average Precision)** as the primary optimization metric, as it focuses exclusively on the positive class without being inflated by true negatives.
  2. For the XGBoost classifier, we derived the exact imbalance ratio:
     $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{227,451}{394} = 577.29$$
  3. This hyperparameter re-weighted the gradient loss for fraud instances, lifting fraud recall to **84.69%** while maintaining a precision of **77.57%** and cutting false positives from 1,386 down to 24 on the 56,962-transaction test set.

### Challenge 2: Strict Prevention of Data Leakage
* **The Problem**: Preprocessing an entire dataset before splitting introduces subtle, destructive data leakage: global scaling parameters ($\mu, \sigma$) inadvertently encode information about test transaction distributions into the training pipeline.
* **Engineering Solution**:
  * We enforced a strict stratified train/test split ($80/20$, `stratify=y`, `random_state=42`) *before* any transformation was performed.
  * `StandardScaler` was fitted strictly on `X_train` and encapsulated inside a scikit-learn `Pipeline` alongside the model artifact.
  * In the GNN and hybrid modules, the independent scalers (`gnn_scaler.joblib`) were fitted strictly on training subsets and persisted as separate joblib artifacts.

### Challenge 3: Hardware Constraints on an 8 GB Apple Silicon M1
* **The Problem**: Running XGBoost, PyTorch Geometric, Feast, Evidently AI, Kafka simulations, SQLite databases, and Streamlit concurrently on a MacBook Air M1 with 8 GB of shared RAM risks severe memory pressure, swap thrashing, and kernel process termination (`SIGKILL`).
* **Engineering Solution**:
  * **Memory-Efficient Subsampling for GNN**: Rather than materializing a 284,000-node graph (which would require $>16\text{ GB}$ of RAM for dense edge indices), `src/train_gnn.py` constructed a stratified subset of 15,000 transactions that preserved all 492 fraud cases, keeping memory footprint $<400\text{ MB}$.
  * **Zero-Copy Ingestion**: Streaming buffers and drift sliding windows use double-ended queues (`collections.deque(maxlen=2000)`) with fixed memory caps.
  * **Lightweight Local Persistence**: SQLite was selected over heavy external database daemons for both Feast online caching and Analyst case management, maintaining zero operational overhead.

### Challenge 4: macOS ARM64 OpenMP Multi-Runtime Collision (`SIGSEGV 139`)
* **The Problem**: During hybrid inference, loading `xgboost_pipeline.joblib` and `gnn_model.pt` in the same Python process caused an immediate segmentation fault (`SIGSEGV 139`).
* **Root Cause Analysis**: Both PyTorch and XGBoost bundle their own versions of the Intel OpenMP runtime (`libomp.dylib`). On macOS ARM64, when PyTorch initializes its internal threading pool and XGBoost subsequent attempts to initialize its own separate `libomp` instance, dynamic linker runtime collisions cause undefined memory state and fatal thread termination.
* **Engineering Solution**:
  * Set explicit environment guards at the very top of `src/hybrid.py`, `src/predict.py`, and `src/gnn.py` prior to importing third-party libraries:
    ```python
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    ```
  * In addition, `HybridFraudPredictor` encapsulates all GNN execution in defensive `try/except` blocks to ensure that any runtime tensor exception immediately reverts 100% to verified XGBoost tabular inference.

### Challenge 5: The GNN Reality Check (Graph vs. Tabular on Anonymized Data)
* **The Expectation**: Fraud detection literature frequently highlights Graph Neural Networks as the state-of-the-art approach for identifying coordinated fraud syndicates and synthetic identity rings.
* **The Reality**: When we trained a 2-layer `FraudGraphSAGE` model using PyTorch Geometric on a $k$-NN similarity graph, the empirical results were clear:
  * **Standalone XGBoost PR-AUC**: **0.9924** (100% Precision, 98.98% Recall on stratified test split).
  * **Standalone GraphSAGE PR-AUC**: **0.8953** (61.59% Precision, 86.73% Recall).
  * **Hybrid ($\alpha=0.80$)**: **0.9906** PR-AUC (measured in-memory during test split evaluation).
* **Why Tabular Won**: The credit card dataset features ($V_1$ to $V_{28}$) are PCA projections. There are no account numbers, credit card tokens, device fingerprints, IP addresses, or merchant IDs. As a consequence, a $k$-NN graph connects transactions that share similar numerical values in PCA space, which is merely a proxy for Euclidean distance, not real-world fraud rings. Tree-based partitioning with gradient boosting excels at axis-aligned tabular splits on PCA vectors, whereas graph message passing across feature-similarity edges can introduce noise from legitimate transactions that happen to share similar feature magnitudes.
* **Architectural Lesson**: Rather than artificially claiming GNN improved performance, we documented the empirical results honestly, decoupled GNN into `src/hybrid.py`, and made XGBoost the default verified production backbone.

### Challenge 6: Production Gating & Retraining Safeguards
* **The Problem**: Naively connecting a data-drift detector directly to a model retraining script creates dangerous failure modes: retraining loops triggered by transient spikes, model degradation on corrupted incoming data, and unmonitored production overrides.
* **Engineering Solution**:
  * We engineered a standalone `RetrainingController` (`src/retrain.py`) with strict operational gates:
    1. **Cooldown Window**: Minimum 3,600 seconds between retraining events to prevent cascading loops.
    2. **Gated Drift Triggers**: Only activates if `drift_share >= 0.30` or `number_of_drifted_columns >= 8`.
    3. **Candidate Validation**: Candidate models are trained in an isolated path (`models/candidate_pipeline.joblib`) and validated on an untouched test split. Promotion occurs only if candidate PR-AUC meets the minimum threshold ($\ge 0.80$) and matches baseline performance.
    4. **Automatic Rollback**: The active model is backed up to `models/xgboost_pipeline_backup.joblib` before promotion, with automated one-call rollback capability.

---

## 3. Architecture Trade-offs & Decisions

| Decision | Alternative Considered | Chosen Approach | Justification |
| :--- | :--- | :--- | :--- |
| **Inference Orchestration** | Overwrite `src/predict.py` with hybrid logic | Separate module: `src/hybrid.py` | Preserves verified 30-feature baseline contract while providing opt-in hybrid capabilities. |
| **Feature Store Backend** | Redis + PostgreSQL | SQLite online store (`data/online_store.db`) | Eliminates external daemon dependencies for reliable local evaluation on 8 GB RAM. |
| **Drift Monitoring Architecture** | Inline in-process Evidently calls | Subprocess bridge to isolated conda environment | Prevents transitive dependency conflicts between Evidently and PyTorch/scikit-learn wheels. |
| **Analyst Feedback Store** | In-memory session state | Thread-safe SQLite (`data/analyst_feedback.db`) | Persists analyst investigation notes and verified ground truth across application restarts. |

---

## 4. Test Evidence & Quality Assurance

* **Automated Unit Tests**: **79 Passed, 0 Failures, 3 Known Deprecation Warnings** (executed in 36.10s).
  * `tests/test_streaming.py`: 30 tests (Kafka producer, consumer, mock broker, DLQ error boundaries).
  * `tests/test_api.py`: 16 tests (FastAPI routes, schemas, healthcheck, batch inference).
  * `tests/test_feature_store.py`: 9 tests (Feast registry, velocity features, online retrieval).
  * `tests/test_hybrid.py`: 8 tests (Hybrid single/batch scoring, weight blending, fallback mechanisms).
  * `tests/test_drift.py`: 7 tests (Reference schema, buffer mechanics, deliberate drift detection).
  * `tests/test_analyst_feedback.py`: 5 tests (Case recording, status updates, summary aggregation).
  * `tests/test_retrain.py`: 4 tests (Cooldown enforcement, drift decision gating, rollback).

---

## 5. Enterprise Roadmap & Future Enhancements

If this platform were transitioning from internship evaluation to a multi-region tier-1 financial institution, the following enhancements would represent the natural next evolutionary steps:

1. **True Bipartite Entity Graph**:
   Ingest raw transaction graphs connecting `CardholderID` $\leftrightarrow$ `Transaction` $\leftrightarrow$ `MerchantID` $\leftrightarrow$ `DeviceFingerprint`. This would unlock the genuine power of Graph Neural Networks for detecting synchronized card-testing attacks and synthetic identity fraud rings.
2. **Distributed Infrastructure**:
   Replace SQLite with Amazon ElastiCache (Redis) for Feast online feature caching (<1ms retrieval) and Amazon Aurora PostgreSQL for analyst case records.
3. **Live Cluster Orchestration**:
   Deploy the verified Docker image to a managed Kubernetes cluster (EKS/GKE) with horizontal pod autoscaling and automated Canary deployments backed by MLflow model registry promotion gates.
