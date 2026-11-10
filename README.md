# Nexvia Technologies AI/ML Engineering Internship Portfolio

**Intern**: Sachin Jadhav  
**Role**: Machine Learning Engineering Intern  
**Internship Tracks**: 
- **AIML-01**: Real-Time Financial Fraud Detection Platform
- **AIML-04**: Autonomous Driving Perception Stack  
**Primary Hardware**: Apple MacBook Air M1 (Apple Silicon arm64, 8 GB Unified Memory, macOS 26.6.2)  
**Evaluation Status**: Final Combined Submission  

---

## Executive Portfolio Summary

This combined repository represents the complete internship deliverables developed for **Nexvia Technologies**. It demonstrates production-oriented machine learning engineering across two demanding domains:

1. **AIML-01 (Financial Machine Learning & Real-Time Systems)**: An end-to-end payment screening engine addressing extreme class imbalance (0.172% fraud) with a cost-sensitive XGBoost classifier, sub-10ms FastAPI microservice, Feast feature caching, Kafka streaming with Dead-Letter Queue fault tolerance, Evidently AI automated drift monitoring, and an interactive 4-tab Streamlit analyst review dashboard.
2. **AIML-04 (Edge Autonomous Perception & Computer Vision)**: A modular autonomous vehicle perception prototype processing forward-facing camera and LiDAR sensors, featuring YOLOv8n obstacle detection, ByteTrack video tracking, deterministic OpenCV lane corridor extraction, MiDaS monocular relative depth estimation with interior-patch obstacle fusion, 3D Camera-LiDAR calibration projection, ONNX Runtime multi-threaded CPU acceleration, and CARLA-compatible ROS 2 Humble bridge nodes.

Across both projects, an automated regression and integration test harness defines **95 individual tests** (79 in AIML-01, 16 in AIML-04). When executed within their respective target environments, both test suites achieved a **100% pass rate** (79/79 passed in 51.91s; 16/16 passed in 0.035s).

---

## Repository Architecture

```text
Nexvia-Internship-Projects/
├── .gitignore                          # Global portfolio exclusion rules
├── README.md                           # This portfolio overview document
│
├── AIML-01/                            # Real-Time Financial Fraud Detection Platform
│   ├── .devcontainer/                  # Dev container definition
│   ├── data/                           # Validated 227k baseline (reference_data.parquet)
│   ├── docs/                           # Architecture, API docs, reflections, presentation outline
│   ├── feature_repo/                   # Feast feature store definitions & sample data generator
│   ├── k8s/                            # Kubernetes manifests (Deployment, Service, HPA, ConfigMap)
│   ├── models/                         # Serialized models & pipelines (.joblib, .pt, .json)
│   ├── outputs/                        # Visual confusion matrices, PR curves, drift reports, metrics JSON
│   ├── src/                            # Complete Python microservice & streaming pipeline
│   ├── tests/                          # 79 automated regression & integration tests (100% passing)
│   ├── Dockerfile                      # Non-root production container definition
│   ├── README.md                       # Dedicated AIML-01 technical documentation
│   ├── app.py                          # 4-tab Streamlit analyst review dashboard
│   ├── main.py                         # CLI execution entry point
│   └── requirements.txt                # Python dependencies
│
└── AIML-04/                            # Autonomous Driving Perception Stack
    ├── ros2_nodes/                     # CARLA-compatible ROS 2 Humble bridge & publisher nodes
    ├── tests/                          # 16 automated local regression tests (100% passing)
    ├── README.md                       # Dedicated AIML-04 technical documentation
    ├── REFLECTION.md                   # 17-topic engineering self-reflection
    ├── PRESENTATION.md                 # 10-slide presentation deck outline
    ├── kitti.yaml                      # Portable dataset configuration
    ├── metrics_eval.json               # KITTI validation benchmark telemetry
    ├── yolov8n.pt                      # Verified PyTorch YOLOv8n detector weights (6.55 MB)
    ├── yolov8n.onnx                    # Exported ONNX Runtime model (12.68 MB)
    ├── sample_traffic_stream.mp4       # Input demonstration traffic stream (5.40 MB)
    ├── tracked_output.mp4              # ByteTrack annotated output video (6.01 MB)
    ├── *.jpg                           # 8 demonstration perception output images
    └── *.py                            # 12 core computer vision modules
```

---

## Project 1: AIML-01 – Real-Time Financial Fraud Detection Platform

### Key Engineering Highlights
- **Cost-Sensitive XGBoost Pipeline**: Optimized for extreme class imbalance (0.172% fraud rate) using exact ratio `scale_pos_weight = 577.29` ($227,451 / 394$).
- **Empirical Performance** (on the 56,962 test transactions):
  - **PR-AUC**: **0.8686** (vs. Logistic Regression Baseline of **0.7190**, a **+0.1496** lift).
  - **Fraud Precision**: **77.57%** (vs. Baseline **6.10%**).
  - **Fraud Recall**: **84.69%** (vs. Baseline **91.84%**).
  - **False Positive Reduction**: Slashed false alarms from **1,386 down to 24** (**98.3% reduction**).
- **Sub-10ms REST Gateway**: Asynchronous FastAPI service delivering **2.8–4.2 ms** local latency on Apple Silicon M1 for `/predict` and `/predict_batch`.
- **Fault-Tolerant Streaming**: Resilient Kafka consumer architecture with automated Dead-Letter Queue (`dead-letter-queue`) isolation for malformed transaction schemas.
- **Feast Online Feature Store**: Low-latency velocity aggregations (1h, 24h) backed by SQLite persistence and in-memory LRU caching.
- **Evidently AI Drift Safeguards**: Continuous Kolmogorov-Smirnov and population drift detection against a 227k baseline (`data/reference_data.parquet`), coupled with an MLflow automated retraining gatekeeper.
- **Analyst Dashboard**: 4-tab Streamlit dashboard (`app.py`) providing single transaction scoring with feature contributions, batch screening, benchmark inspection, and human-in-the-loop analyst feedback logging.

---

## Project 2: AIML-04 – Autonomous Driving Perception Stack

### Key Engineering Highlights
- **Module 1 (Object Detection & Benchmark)**: YOLOv8n evaluated across 1,496 KITTI validation images (`wandb_eval.py`). Benchmark telemetry serialized to `metrics_eval.json`:
  - **Aggregate 4-Class mAP@50**: **0.343** (includes the 0.0 AP penalty for the unmapped Cyclist class).
  - **Mapped 3-Class mAP@50**: **0.458** (across Car, People, and Truck).
  - **Per-Class AP@50**: Car: **0.6558** (0.656), People: **0.395**, Truck: **0.323**, Cyclist: **0.0** (due to unmapped COCO labels).
  - **Aggregate mAP@50-95**: **0.180**, Precision: **0.389**, Recall: **0.344**.
- **Module 2 (Multi-Object Tracking)**: ByteTrack association (`track_video.py`) processing roadway video, maintaining unique track IDs, category persistence, and confidence metrics for vehicles and pedestrians. Output: `tracked_output.mp4`.
- **Module 3 (Lane Corridor Extraction)**: Deterministic OpenCV pipeline (`lane_detection.py`) with Canny edge filtering, Inverse Perspective Mapping (IPM), sliding window polynomial fitting ($x = Ay^2 + By + C$), road curvature radius ($R_{\text{curve}}$), and vehicle center offset calculation. Output: `lane_detection_output.jpg`.
- **Module 4 (Monocular Relative Depth & Spatial Fusion)**: MiDaS DPT relative inverse disparity (`depth_estimation.py`). Central 50% interior-patch sampling of 2D bounding boxes to avoid boundary edge bleed, categorizing obstacles into Near ($\ge 160$), Mid-Range ($90..159$), and Far ($< 90$) proximity tiers (`obstacle_depth_fusion.py`). Output: `fusion_output.jpg`.
- **Module 5 (Camera-LiDAR 3D Fusion)**: Rigid coordinate transformation ($P_2 \cdot R_{\text{rect}}^0 \cdot T_{\text{velo}}^{\text{cam}}$) projecting 3D Velodyne point clouds onto camera coordinates with jet depth colormaps (`camera_lidar_fusion.py`). Output: `camera_lidar_fusion_output.jpg`.
- **Module 6 (Edge Deployment & ROS 2)**: ONNX multi-threaded CPU acceleration (**39.13 ms** / 25.6 FPS) and Apple Silicon MPS GPU benchmarking (**9.38 ms** / 106.6 FPS). Built CARLA-compatible ROS 2 Humble bridge nodes (`ros2_nodes/`).

---

## Verified Performance Scorecard

| Domain | Key Metric | Baseline / Comparison | Verified Value | Improvement / Target |
| :--- | :--- | :---: | :---: | :---: |
| **AIML-01** | **Precision-Recall AUC (PR-AUC)** | 0.7190 (Logistic Reg) | **0.8686** | **+0.1496** |
| **AIML-01** | **Fraud Precision** | 6.10% | **77.57%** | **+71.47%** |
| **AIML-01** | **False Positives (57k test set)** | 1,386 | **24** | **-98.3% False Alarms** |
| **AIML-01** | **Local Inference Latency** | Target: <10 ms | **2.8–4.2 ms** | **SLA Met** |
| **AIML-01** | **Automated Tests** | 0 passed | **79 / 79 passed** | **100% Pass Rate (51.91s)** |
| **AIML-04** | **Aggregate 4-Class mAP@50** | — | **0.343** | **KITTI Validation (1,496 imgs)** |
| **AIML-04** | **Mapped 3-Class mAP@50** | — | **0.458** | **Car, People, Truck** |
| **AIML-04** | **Car AP@50** | — | **0.6558** | **KITTI Validation** |
| **AIML-04** | **MPS GPU Inference Latency** | PyTorch CPU (48 ms) | **9.38 ms (106.6 FPS)** | **Apple M1 Hardware** |
| **AIML-04** | **ONNX Runtime CPU Latency** | PyTorch CPU (48 ms) | **39.13 ms (25.6 FPS)** | **Cross-Platform Ready** |
| **AIML-04** | **Regression Tests** | 0 passed | **16 / 16 passed** | **100% Pass Rate (0.035s)** |

---

## Installation & Environment Setup

Due to differing dependency requirements (PyTorch Geometric & Feast for AIML-01; OpenCV, Ultralytics YOLOv8 & ROS 2 for AIML-04), separate Conda environments are recommended.

### Environment 1: AIML-01 Setup
```bash
conda create -n nexvia-fraud python=3.11 -y
conda activate nexvia-fraud
cd AIML-01
pip install -r requirements.txt
```

### Environment 2: AIML-04 Setup
```bash
conda create -n av-perception python=3.11 -y
conda activate av-perception
cd AIML-04
pip install torch torchvision numpy opencv-python onnx onnxruntime ultralytics timm
```

> [!NOTE]
> **ROS 2 Middleware Notice**: The ROS 2 Humble nodes located in `ros2_nodes/` rely on `rclpy` and standard ROS 2 messaging packages (`sensor_msgs`, `cv_bridge`). These nodes require a separate, compatible ROS 2 Humble installation or dedicated environment (such as a Linux/Ubuntu container or a dedicated Conda `ros2-humble` channel), which is independent of the standard Python pip package installation.

---

## Execution Guide

All commands should be executed from within the respective project directory to preserve relative path resolution.

### Running AIML-01 Modules
```bash
# Navigate to AIML-01
cd AIML-01

# 1. Launch the Streamlit Analyst Review Dashboard (4 tabs)
conda run -n nexvia-fraud streamlit run app.py

# 2. Launch the FastAPI Scoring Microservice
conda run -n nexvia-fraud uvicorn src.api:app --host 0.0.0.0 --port 8000

# 3. Execute the Streaming Consumer Simulation with DLQ
conda run -n nexvia-fraud python src/streaming/consumer.py

# 4. Run the Data Drift Analysis Pipeline
conda run -n nexvia-fraud python src/drift_runner.py
```

### Running AIML-04 Modules
```bash
# Navigate to AIML-04
cd AIML-04

# 1. Execute Multi-Object Tracking across Video Stream
conda run -n av-perception python track_video.py

# 2. Run Lane Corridor Extraction & Curvature Estimation
conda run -n av-perception python lane_detection.py

# 3. Run Monocular Relative Depth Estimation
conda run -n av-perception python depth_estimation.py

# 4. Run Obstacle Bounding Box & Depth Fusion
conda run -n av-perception python obstacle_depth_fusion.py

# 5. Run 3D Camera-LiDAR Calibration Projection
conda run -n av-perception python camera_lidar_fusion.py

# 6. Benchmark YOLOv8n PyTorch vs. ONNX Runtime
conda run -n av-perception python benchmark_yolo_onnx.py
```

---

## Automated Test Suites (95 Defined Tests)

Both projects feature comprehensive test suites validating mathematical calculations, pipeline stability, and edge-case handling. Across both repositories, **95 tests are defined and discovered**:
- **AIML-01**: 79 tests across 7 test suites.
- **AIML-04**: 16 tests in `tests/test_perception_stack.py`.

### Execution Verification Evidence:
```bash
# Project 1: Execute 79 Tests in nexvia-fraud (Verified: 79 passed, 0 failed in 51.91s)
cd AIML-01 && conda run -n nexvia-fraud pytest tests/

# Project 2: Execute 16 Tests in av-perception (Verified: 16 passed, 0 failed in 0.035s)
cd AIML-04 && conda run -n av-perception python -m unittest -v tests/test_perception_stack.py
```

---

## Hardware Limitations & Technical Disclosures

In the spirit of rigorous engineering transparency:
1. **CARLA Closed-Loop Simulation (AIML-04)**: The CARLA UE4 simulator binary is unsupported on macOS Apple Silicon. A partial integration workaround was implemented using CARLA-compatible ROS 2 stream publishers and bridge nodes matching CARLA topic and message schemas, but this does not replace or fully mitigate an actual live closed-loop CARLA simulation.
2. **NVIDIA TensorRT Deployment (AIML-04)**: TensorRT requires NVIDIA CUDA hardware. Performance was benchmarked on cross-platform alternatives: Apple Silicon MPS (9.38 ms) and multi-threaded ONNX Runtime CPU (39.13 ms), rather than claiming equivalent execution of TensorRT.
3. **KITTI Model Fine-Tuning (AIML-04)**: Stock COCO YOLOv8n was evaluated zero-shot on 1,496 KITTI validation images; KITTI conversion and partition pipelines are complete, but model re-training was not executed.
4. **Graph Neural Networks vs. Tabular XGBoost (AIML-01)**: An experimental 2-layer GraphSAGE model was trained on a $k$-NN similarity graph. On the evaluated 15,000-node graph subset, standalone **XGBoost achieved 0.9924 PR-AUC** vs. **GraphSAGE 0.8953 PR-AUC**. Because the dataset features ($V_1$–$V_{28}$) are PCA projections without raw entity identifiers, tree-based partitioning on axis-aligned splits outperformed graph message passing. XGBoost was honestly selected as the verified production backbone, scoring **0.8686 PR-AUC** on the full 57k test set.
5. **Kubernetes Deployment (AIML-01)**: Non-root `Dockerfile` and Kubernetes manifests (`k8s/`) are authored and valid, but live cloud cluster deployment was not executed.

---

## Obtaining Excluded Datasets

To keep the repository within GitHub guidelines and prevent oversized commits, raw datasets are excluded from Git:
- **AIML-01 (Kaggle Credit Card Fraud Dataset)**: Download `creditcard.csv` (144 MB) from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it into `AIML-01/data/raw/creditcard.csv`. Note: The pre-processed 227k baseline (`AIML-01/data/reference_data.parquet`) is already included for drift detection and local testing.
- **AIML-04 (KITTI Vision Benchmark Suite)**: Download the raw Velodyne point clouds, camera calibration matrices, and Left Color Images from [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) and place them into `AIML-04/data/kitti/`.

---

## Original Repositories
- **AIML-01**: [https://github.com/Sachinj2004/Real-Time-Financial-Fraud-Detection.git](https://github.com/Sachinj2004/Real-Time-Financial-Fraud-Detection.git)
- **AIML-04**: [https://github.com/Sachinj2004/Autonomous-Driving-Perception-Stack.git](https://github.com/Sachinj2004/Autonomous-Driving-Perception-Stack.git)
