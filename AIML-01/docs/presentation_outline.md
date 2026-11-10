# Nexvia AIML-01: Final Internship Presentation Deck Outline

**Project**: Real-Time Financial Fraud Detection Platform  
**Presenter**: Machine Learning Engineering Intern  
**Target Duration**: 10 Minutes  
**Audience**: Technical Lead, Project Mentor, and Nexvia Engineering Evaluation Committee  

---

## Presentation Timing Summary

| Slide # | Slide Title | Allocated Time | Cumulative Time |
| :---: | :--- | :---: | :---: |
| 1 | Title, Mission & Executive Summary | 1:00 | 1:00 |
| 2 | Problem Formulation & Hardware Constraints | 1:00 | 2:00 |
| 3 | End-to-End Architecture (6 Core Subsystems) | 1:30 | 3:30 |
| 4 | Resilient Streaming & Feast Feature Store | 1:00 | 4:30 |
| 5 | Model Evolution & Production XGBoost Benchmarks | 1:30 | 6:00 |
| 6 | GNN Exploration & The Hybrid Scoring Reality Check | 1:30 | 7:30 |
| 7 | Continuous Drift Monitoring & Controlled Retraining Safeguards | 1:00 | 8:30 |
| 8 | Human-in-the-Loop Analyst Review & Feedback Loop | 0:30 | 9:00 |
| 9 | Rigorous Verification & Test Evidence (79 Passed Tests) | 0:30 | 9:30 |
| 10 | Technical Reflections & Enterprise Production Roadmap | 0:30 | 10:00 |

---

## Detailed Slide Content & Speaker Notes

### Slide 1: Title, Mission & Executive Summary (1:00)
* **Slide Visuals**:
  * Project Title: Real-Time Financial Fraud Detection Platform (AIML-01).
  * High-level metric highlights: **0.8686 PR-AUC**, **84.7% Fraud Recall**, **98.3% False Positive Reduction**, **2.8–4.2ms Local Latency (Apple M1)**, **79 Unit Tests Passing**.
* **Key Bullet Points**:
  * Designed and deployed an end-to-end payment screening engine for real-time financial security.
  * Solved extreme class imbalance (0.172% fraud) while preserving low operational latency.
  * Integrated streaming ingestion, feature caching, graph relational embeddings, automated drift safeguards, and analyst case review.
* **Speaker Notes**:
  > *"Good morning. Today I am presenting the Real-Time Financial Fraud Detection Platform developed under Nexvia AIML-01. Financial fraud moves in milliseconds, but defending against it requires balancing two opposing forces: catching deceptive anomalies before settlement while avoiding false declines for legitimate cardholders. Over this internship, I built a production-grade ML system that elevates fraud PR-AUC from 0.719 to 0.8686, slashes false positives by 98.3%, and demonstrates 2.8–4.2ms local inference on Apple M1 with automated drift and retraining safeguards."*

---

### Slide 2: Problem Formulation & Hardware Constraints (1:00)
* **Slide Visuals**:
  * Class distribution chart: 99.828% Legitimate vs. 0.172% Fraud (492 cases in 284,807 transactions).
  * Dual challenge callout: **Extreme Imbalance** vs. **Low Resource Footprint (8 GB Apple M1)**.
* **Key Bullet Points**:
  * **Class Imbalance**: High accuracy (99.8%) is trivial and misleading; PR-AUC is the true operational metric.
  * **Privacy Anonymization**: PCA features ($V_1$ to $V_{28}$) conceal raw entity identifiers.
  * **Strict Hardware Constraints**: Zero-swapping memory management on an 8 GB Apple Silicon M1.
* **Speaker Notes**:
  > *"In payments, a model predicting 100% legitimate transactions achieves 99.8% raw accuracy while missing every single fraud event. That is why PR-AUC, precision, and recall are our primary targets. Furthermore, the dataset features are PCA-transformed to protect user privacy, which introduces unique challenges for graph-based fraud ring detection. Finally, every pipeline step was engineered to run memory-efficiently within an 8 GB M1 environment."*

---

### Slide 3: End-to-End Architecture (The 6 Subsystems) (1:30)
* **Slide Visuals**:
  * 6-subgraph architecture diagram:
    1. Ingestion & Streaming (Kafka + DLQ)
    2. Feature Store (Feast SQLite + Parquet)
    3. Multi-Model Inference (XGBoost + GraphSAGE + Hybrid Scorer)
    4. Drift & Retraining (Evidently AI + Retraining Controller + MLflow)
    5. Analyst Review (SQLite + Streamlit Tab 5)
    6. Serving & Packaging (FastAPI + Docker + K8s manifests)
* **Key Bullet Points**:
  * Complete decoupling of concerns: Ingestion $\rightarrow$ Feature Retrieval $\rightarrow$ Inference $\rightarrow$ Drift Logging $\rightarrow$ Human Feedback.
* **Speaker Notes**:
  > *"Here is the 6-subsystem architecture. Payments enter via Kafka or our resilient mock broker. Feast provides rolling velocity metrics for downstream audit. The payload flows to our inference engine, which combines tree-based decision boundaries with graph relational context. Real-time scoring streams directly into an in-memory drift buffer, while any transaction flagged above threshold automatically queues into our analyst review database."*

---

### Slide 4: Resilient Streaming & Feast Feature Store (1:00)
* **Slide Visuals**:
  * Sequence flow: Producer $\rightarrow$ Consumer $\rightarrow$ DLQ on error $\rightarrow$ Feast SQLite Online Feature Cache.
  * Velocity features breakdown: `rolling_tx_count_1h`, `rolling_amount_mean_1h`, `pca_energy_norm`.
* **Key Bullet Points**:
  * **Zero Poison Pills**: Malformed transactions are routed to `payment.transactions.dlq` without stalling streaming.
  * **Schema Safety**: Feast velocity attributes enrich decision payloads without violating the 30-feature baseline contract.
* **Speaker Notes**:
  > *"Our streaming consumer includes explicit dead-letter queuing to ensure that corrupted messages never block real-time payment ingestion. In parallel, Feast maintains entity velocity metrics like 1-hour transaction frequency and spending velocity. To prevent training-serving skew, these enriched features attach to output metadata for compliance and audit rather than corrupting the 30-feature baseline model input."*

---

### Slide 5: Model Evolution & Production XGBoost Benchmarks (1:30)
* **Slide Visuals**:
  * Confusion matrix comparison table: Baseline Logistic Regression vs. Production XGBoost.
  * PR Curve visualization showing PR-AUC jump from 0.7190 to 0.8686.
* **Key Bullet Points**:
  * Baseline (Logistic Regression): 1,386 false alarms for 90 caught frauds (Precision: 6.1%).
  * Production XGBoost (`scale_pos_weight=577.29`): 24 false alarms for 83 caught frauds (Precision: 77.6%, Recall: 84.7%).
  * **Business Impact**: Slashed false declines by 1,362 cases on 56,962 test transactions.
* **Speaker Notes**:
  > *"On the held-out test set of 56,962 transactions, our baseline logistic regression generated 1,386 false alarms—a disaster for customer checkout experience. By optimizing an XGBoost pipeline with tree-based gradient boosting and an exact scale_pos_weight of 577.29, we reduced false positives to just 24, raised precision from 6% to 77.6%, and achieved a production PR-AUC of 0.8686."*

---

### Slide 6: GNN Exploration & The Hybrid Reality Check (1:30)
* **Slide Visuals**:
  * Test set benchmark comparison: Standalone XGBoost (0.9924 PR-AUC) vs. GraphSAGE (0.8953 PR-AUC) vs. Hybrid (0.9906).
  * Diagram of $k$-NN feature graph ($k=5$) with anchor nodes.
* **Key Bullet Points**:
  * **Implementation**: Inductive 2-layer `FraudGraphSAGE` in PyTorch Geometric.
  * **Empirical Reality**: XGBoost outperformed GraphSAGE because PCA features lack real account/device entities.
  * **Architectural Decision**: Hybrid predictor (`src/hybrid.py`) uses configurable weighting ($\alpha=0.80$) with 100% defensive fallback to XGBoost.
* **Speaker Notes**:
  > *"We explored Graph Neural Networks using PyTorch Geometric to detect relational anomalies. However, an honest empirical evaluation on our held-out test split revealed a vital engineering lesson: because the dataset features are PCA-transformed and lack true entity identifiers like card numbers or device hashes, the k-NN graph measures metric-space feature distance rather than real fraud rings. As a result, standalone XGBoost achieved 0.9924 PR-AUC compared to 0.8953 for GraphSAGE. Rather than forcing GNN into the critical path, we built a hybrid module with configurable weighting and 100% automated fallback to XGBoost."*

---

### Slide 7: Continuous Drift Monitoring & Retraining Safeguards (1:00)
* **Slide Visuals**:
  * Drift workflow: 2,000 Tx buffer $\rightarrow$ Isolated conda subprocess $\rightarrow$ KS-Test $\rightarrow$ Retraining Controller.
  * Retraining safeguards: Cooldown timer (3,600s), Drift gate ($\ge 30\%$), Candidate validation, MLflow audit.
* **Key Bullet Points**:
  * **Zero Library Collisions**: Evidently AI runs in an isolated conda environment (`nexvia-monitoring`).
  * **Automated Safeguards**: Retraining is gated by cooldowns, candidate PR-AUC validation, and automated rollback.
* **Speaker Notes**:
  > *"Real-world payment patterns drift constantly. Our drift monitoring buffers live transactions and periodically runs Kolmogorov-Smirnov statistical tests via Evidently AI in an isolated subprocess. When drift exceeds 30%, our Retraining Controller triggers candidate model training. To protect production integrity, candidate models must achieve at least 0.80 PR-AUC on an untouched split before promotion, active models are automatically backed up, and all metrics log to MLflow."*

---

### Slide 8: Human-in-the-Loop Analyst Review & Feedback Loop (0:30)
* **Slide Visuals**:
  * Screenshot/Mockup of Streamlit Tab 5: KPI cards, case queue table, status transition form (`New` $\rightarrow$ `Under Review` $\rightarrow$ `Confirmed Fraud` / `Legitimate`).
* **Key Bullet Points**:
  * Flagged transactions automatically register into `data/analyst_feedback.db`.
  * Analysts record investigation notes and ground-truth decisions to close the active learning loop.
* **Speaker Notes**:
  > *"Model predictions are only as good as the ground truth feeding them. Tab 5 of our Streamlit dashboard provides fraud analysts with an interactive investigation queue. High-risk transactions are automatically captured, investigated, tagged, and persisted in SQLite for future drift and retraining cycles."*

---

### Slide 9: Rigorous Verification & Test Evidence (0:30)
* **Slide Visuals**:
  * Test execution badge: **79 Passed, 0 Failures, 3 Warnings** (36.10s execution).
  * Breakdown: Streaming (30), API (16), Feature Store (9), Hybrid (8), Drift (7), Analyst Feedback (5), Retraining (4).
* **Key Bullet Points**:
  * Unit test coverage spanning every single subsystem with zero regressions.
  * Dockerfile and Kubernetes manifests syntax-verified for deployment readiness.
* **Speaker Notes**:
  > *"Engineering claims require verifiable test evidence. Our automated test suite comprises 79 tests spanning all 6 subsystems—ingestion, feature caching, inference, hybrid fallback, drift monitoring, analyst review, and retraining safeguards. All 79 tests pass with zero failures."*

---

### Slide 10: Conclusion, Technical Takeaways & Future Roadmap (0:30)
* **Slide Visuals**:
  * Summary checklist: Production-Ready XGBoost, Decoupled Hybrid Scorer, Feast Cache, Evidently Drift, Streamlit UI.
  * Future items: Live Kubernetes cluster deployment, multi-entity bipartite graphs with raw transaction metadata.
* **Key Bullet Points**:
  * Completed all 10 project phases without compromising production model artifacts.
  * Platform is ready for presentation and live technical demonstration.
* **Speaker Notes**:
  > *"In summary, the Nexvia Financial Fraud Detection Platform delivers a verified, highly accurate, and resilient machine learning system ready for operational evaluation. Thank you, and I look forward to your questions and demonstrating the live platform."*
