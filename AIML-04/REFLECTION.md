# Engineering Self-Reflection: Autonomous Driving Perception Stack (AIML-04)

- **Project**: Nexvia Project 2 — Autonomous Driving Perception Stack
- **Repository**: `Autonomous-Driving-Perception-Stack`
- **Hardware Platform**: Apple MacBook Air M1 (Apple Silicon arm64, 8 GB Unified Memory, macOS 26.6.2)
- **Author**: Nexvia AIML Intern
- **Evaluation Status**: Final Technical Reflection

---

## 1. Project Overview and Objectives

The goal of this internship project was to design, implement, and benchmark a modular autonomous driving perception prototype capable of processing forward-facing roadway sensors. Autonomous perception requires solving several vision and spatial reasoning tasks concurrently: identifying obstacles, maintaining track continuity across time, detecting lane boundaries and road geometry, assessing obstacle proximity, and projecting 3D spatial points into camera coordinates.

Rather than relying on a single closed-source black box, the primary objective was to build a modular, explainable, and testable perception prototype on edge hardware (Apple Silicon M1). The stack combines deep learning for obstacle detection and monocular relative depth with deterministic classical computer vision for lane detection, integrates ROS 2 Humble middleware for message passing, exports models to ONNX for cross-platform runtime execution, and evaluates performance against the real-world KITTI Vision Benchmark Suite.

---

## 2. YOLOv8 2D Object Detection Implementation

For 2D obstacle detection, I selected the compact **YOLOv8n** (Nano) architecture (3.2 million parameters, 6.55 MB checkpoint `yolov8n.pt`). Given that the development environment is an Apple Silicon M1 laptop lacking an NVIDIA discrete GPU, I executed PyTorch directly on Apple's **Metal Performance Shaders** (`torch.device("mps")`).

### Key Implementation Details:
- **Standalone Detection Test**: `test_detection.py` loads `yolov8n.pt` and evaluates forward inference across all classes on input imagery (`sample_road.jpg`) using a confidence threshold of $\ge 0.25$, generating `detection_output.jpg`.
- **Traffic Participant Filtering**: Autonomous driving demands filtering non-traffic classes. While `test_detection.py` tests standalone inference broadly, downstream perception modules (`track_video.py`, `onnx_inference.py`, `benchmark_yolo_onnx.py`, `camera_lidar_fusion.py`, and `perception_inference_ipc.py`) explicitly enforce a 6-class traffic safety filter (`RELEVANT_CLASSES`: `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`).
- **Inference Latency**: In isolated model benchmarking over 100 statistical runs (`benchmark_yolo_onnx.py`), YOLOv8n achieved an average forward inference latency of **$9.38\text{ ms}$** on the M1 GPU (equivalent to **$106.6\text{ FPS}$** for the neural network forward pass alone).
- **Scope Distinction**: This $9.38\text{ ms}$ figure measures strictly isolated forward model inference. The complete end-to-end perception pipeline latency is documented in `README.md` as **$\sim 140\text{ ms} - 150\text{ ms}$**, which includes sensor ingestion, image decoding, YOLO detection ($82.64\text{ ms}$), MiDaS depth ($60.18\text{ ms}$), obstacle-depth fusion, IPC transfer, and ROS 2 topic publishing.

---

## 3. ByteTrack Object Tracking

Single-frame object detection lacks temporal memory. To provide temporal continuity and velocity estimation, I integrated the **ByteTrack** multi-object tracking algorithm in `track_video.py`.

### Technical Rationale & Behavior:
- Traditional trackers discard bounding boxes falling below an arbitrary confidence threshold (e.g., $< 0.5$). This causes track fragmentation when vehicles experience momentary occlusions, distance variations, or motion blur.
- ByteTrack uses a two-stage data association approach:
  1. It first matches high-confidence detections with existing track trajectories using Kalman filter motion predictions and spatial bounding box Intersection-over-Union (IoU).
  2. It then matches unmatched tracks against remaining *low-confidence* detections, preserving track identities through momentary visibility dips.
- The pipeline was verified by processing continuous road footage (`sample_traffic_stream.mp4`) and saving the tracking telemetry and centroid trails to `tracked_output.mp4` (6.01 MB decimal / 5.73 MiB).

---

## 4. OpenCV Classical Lane Detection

For lane boundary detection (`lane_detection.py`), I chose deterministic classical computer vision over deep segmentation networks to ensure that geometric calculations remain transparent, verifiable, and computationally lightweight.

### Pipeline Stages:
1. **Color Filtering**: Applied thresholding in the HLS color space to isolate white roadway stripes ($L > 180$) and yellow lane markings ($H \in [15, 35], S > 70, L > 60$).
2. **Edge Extraction**: Combined Gaussian smoothing ($5 \times 5$) and Canny edge detection ($50, 150$) within a trapezoidal Region of Interest (ROI).
3. **Inverse Perspective Mapping (IPM)**: Calculated homography matrices (`cv2.getPerspectiveTransform`) to warp the perspective image into a top-down Bird's-Eye View.
4. **Sliding-Window Polynomial Fitting**: Utilized a 9-stage sliding-window histogram peak search to identify active lane pixels, fitting second-order polynomials ($x = Ay^2 + By + C$) to the left and right boundaries.
5. **Corridor Sanity Checks**: Enforced mathematical sanity rules: boundaries must not cross, and lane separation must fall within valid physical corridor limits ($2.2\text{ m} \le w \le 5.2\text{ m}$).
6. **Estimated Geometric Telemetry**: Calculated the vehicle's lateral offset from lane center and road curvature radius using standard highway pixel-to-meter approximations ($3.7\text{ m} / 700\text{ px}$ width, $30\text{ m} / 720\text{ px}$ forward reach).
- **Outcome & Artifact**: Output visualization verified in `lane_detection_output.jpg`. All measurements are explicitly classified as estimated geometric calculations.

---

## 5. MiDaS Monocular Relative Depth Estimation

To provide depth awareness from a single monocular camera, I integrated the Intel ISL **MiDaS_small** neural network in `depth_estimation.py`.

### Implementation & Critical Disclosures:
- **Relative Disparity vs. Metric Distance**: MiDaS outputs **relative inverse disparity**, where larger pixel values indicate closer objects and smaller values indicate distant background. It does **not** output absolute physical distances in meters.
- **Normalization**: The model output was normalized to a standard 0–255 grayscale range and mapped to an inferno colormap in `depth_output.jpg`.
- **Limitation**: Without stereo cameras, ground-plane calibration, or LiDAR triangulation, monocular depth cannot directly command safety-critical braking in metric meters. It provides qualitative proximity cues.

---

## 6. Obstacle-Depth Proximity Fusion

In `obstacle_depth_fusion.py`, I combined the 2D bounding boxes from YOLOv8n with the dense relative disparity map from MiDaS.

### Engineering Solution — Interior Patch Sampling:
- **Problem**: Averaging disparity over an entire bounding box caused severe depth bleeding. Edge pixels captured background road asphalt, sidewalk curbs, and sky, corrupting the obstacle's depth estimate.
- **Solution**: I designed a **center 50% interior patch sampling method** ($[cx_1, cy_1, cx_2, cy_2]$) and extracted the **median** disparity within that interior region, cleanly rejecting peripheral background noise.
- **Proximity Categorization**: The pipeline categorizes obstacles into three safety tiers:
  - `NEAR` (Disparity $\ge 160$): High collision risk (Red bounding box).
  - `MID-RANGE` (Disparity $90..159$): Caution zone (Yellow bounding box).
  - `FAR` (Disparity $< 90$): Clear corridor (Green bounding box).
- **Artifact**: Verified in `fusion_output.jpg`.

---

## 7. KITTI Dataset Preparation & Zero-Shot Evaluation

To evaluate the perception prototype on real-world autonomous driving data, I prepared the **KITTI Vision Benchmark Suite** (Object Detection):

### Dataset Pipeline:
- **Conversion & Partitioning**: `convert_kitti_to_yolo.py` and `partition_kitti_dataset.py` converted KITTI camera calibrations and 3D labels into normalized YOLO annotations and structured `kitti.yaml`.
- **Data Splits**: Partitioned the 7,481 raw scenes into:
  - **Training Split (80%)**: 5,985 images.
  - **Validation Split (20%)**: 1,496 images containing 7,660 ground-truth bounding boxes across 4 canonical classes defined in `kitti.yaml` (`Car`: 6,230 [81.3%], `People`: 926 [12.1%], `Cyclist`: 306 [4.0%], `Truck`: 198 [2.6%]).
- **Evaluation Engine**: Implemented `wandb_eval.py` to evaluate the 1,496 validation images on Apple Silicon MPS (completed in $44.4\text{ s}$ at $33.7\text{ img/s}$).
- **Measured Metrics (`metrics_eval.json`)**:
  - Overall 4-Class Macro Average: **$\text{mAP@50} = 0.343$**, **$\text{mAP@50-95} = 0.180$**, **$\text{Precision} = 0.389$**, **$\text{Recall} = 0.344$**.
  - Class-Specific Performance:
    - `Car`: $\text{Precision} = 0.706$, $\text{Recall} = 0.603$, $\text{mAP@50} = \mathbf{0.655}$
    - `People`: $\text{Precision} = 0.474$, $\text{Recall} = 0.407$, $\text{mAP@50} = \mathbf{0.395}$
    - `Truck`: $\text{Precision} = 0.375$, $\text{Recall} = 0.366$, $\text{mAP@50} = \mathbf{0.323}$
    - `Cyclist`: $\text{mAP@50} = \mathbf{0.0}$

### Honest Disclosures on Model Evaluation:
1. **No Domain Fine-Tuning**: No model training or fine-tuning was performed on the 5,985 KITTI training images. The evaluation reflects zero-shot inference using stock COCO weights.
2. **Cyclist Class Mapping Mismatch**: Stock COCO YOLOv8n has no dedicated KITTI Cyclist class mapping in the evaluation harness (`COCO_TO_KITTI_MAP`), which lists Cyclist under `UNSUPPORTED_KITTI_CLASSES = [2]`. Consequently, no predictions were mapped to the KITTI Cyclist category during validation, resulting in a benchmark AP of 0.0 for this class. Factoring this 0.0 into the 4-class macro-average reduces overall mAP@50 from 0.458 (across the 3 mapped classes) to 0.343, quantitatively illustrating the necessity of domain-specific fine-tuning.

---

## 8. Camera-LiDAR Fusion Limitations & Ground-Truth Audit Mode

To connect 2D visual detections with 3D physical sensor space, I implemented rigid-body calibration projection in `camera_lidar_fusion.py`:

$$\begin{bmatrix} u \cdot z \\ v \cdot z \\ z \end{bmatrix} = P_2 \cdot R_{0\_rect}^{4\times 4} \cdot Tr_{velo}^{cam} \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

### Honest Technical Disclosures:
- **Raw Point-Cloud Archive Not Downloaded**: The raw KITTI Velodyne point-cloud archive (~29 GB of `.bin` files) was not downloaded to local disk due to storage constraints. Consequently, genuine point-cloud stream fusion was not executed.
- **Ground-Truth Audit Mode**: The script demonstrated the calibration projection mathematics by operating in Ground-Truth Audit Mode. It parsed verified 3D physical coordinates from KITTI `label_2` annotations, projected them onto the camera plane via the calibration matrices, and spatially associated them with 2D YOLO detections using bounding box IoU.
- **Artifact**: Verified in `camera_lidar_fusion_output.jpg`, explicitly labeled as Ground-Truth Audit Mode.

---

## 9. ONNX Export and Performance Comparison

To evaluate edge deployment viability, I exported YOLOv8n to the **Open Neural Network Exchange (ONNX)** format using `export_yolo_onnx.py`:
- Model graph: `yolov8n.onnx` (**12.68 MB decimal / 12.09 MiB**, 12,677,624 bytes, opset 17, dynamic batch axes).
- Standalone inference: `onnx_inference.py` implements custom letterboxing, tensor postprocessing, and Non-Maximum Suppression (NMS) on ONNX Runtime CPU without requiring the Ultralytics framework.

### 100-Iteration Comparative Benchmark (`benchmark_yolo_onnx.py`):
| Runtime Backend | Compute Device | Threads | Mean Latency | Throughput (FPS) | Relative Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Native PyTorch MPS** | Apple M1 Metal GPU | - | **$9.38\text{ ms}$** | **$106.6\text{ FPS}$** | **$4.17\times$ Faster** |
| **ONNX Runtime CPU** | Apple M1 CPU | 4 Intra-Op | **$39.13\text{ ms}$** | **$25.6\text{ FPS}$** | $1.00\times$ (Baseline) |

- **Scope Note**: These measurements reflect isolated forward inference passes. Native Apple Silicon MPS provides a substantial latency advantage over CPU execution, while ONNX CPU provides an engine-independent portable fallback.

---

## 10. ROS 2 Dual-Environment IPC Architecture

One of the most complex systems engineering hurdles was integrating ROS 2 middleware with modern deep learning libraries on macOS.

### The Conflict:
- The ROS 2 Humble distribution (RoboStack) uses Python 3.11.17 and specific C++ library bindings.
- PyTorch 2.14.1 with native Apple Metal MPS support required Python 3.11.18 and conflicting NumPy/Torch dynamic libraries.
- Attempting to load `torch` and `rclpy` in the same Python process resulted in segmentation faults and runtime symbol collisions.

### The Solution — Decoupled IPC Architecture (`ros2_nodes/`):
I designed a decoupled dual-process Inter-Process Communication (IPC) architecture operating over a local Unix domain socket (`/tmp/av_perception_ipc.sock`):
1. **Inference Server (`perception_inference_ipc.py`)**: Runs in the `av-perception` Conda environment on the M1 GPU. It loads YOLOv8n and MiDaS, listens on the IPC socket, processes image byte buffers, and returns structured detection and depth payloads.
2. **ROS 2 Bridge (`perception_ros_bridge.py`)**: Runs in the `ros2-humble` environment. It synchronizes incoming camera and LiDAR streams via `message_filters.ApproximateTimeSynchronizer` ($< 50\text{ ms}$ tolerance; measured mean delta $20.52\text{ ms}$, lifetime max $31.03\text{ ms}$), dispatches frames to a background worker thread, sends image payloads across the IPC socket, and publishes ROS 2 `vision_msgs/Detection2DArray` messages.
3. **Sensor Publishers**: `carla_camera_publisher.py` and `carla_lidar_publisher.py` replay KITTI images and publish synthetic point clouds at $10\text{ Hz}$.

---

## 11. CARLA Limitations on Apple Silicon

### Platform Incompatibility:
- The CARLA Autonomous Driving Simulator (0.9.x) is built on Unreal Engine 4 (UE4). Official production builds require an **x86_64 Linux or Windows host with an NVIDIA discrete GPU** supporting Vulkan or DirectX 12.
- CARLA provides no native pre-compiled macOS Apple Silicon (ARM64) server release, and macOS lacks the NVIDIA Vulkan/CUDA drivers required by the CARLA simulation engine.

### Adaptation & Scope Disclosure:
- To test the perception stack against autonomous vehicle messaging conventions, I implemented **CARLA-compatible ROS 2 stream publishers and bridge nodes** adopting CARLA topic names (`/carla/ego_vehicle/...`) and message definitions.
- **Crucial Disclosure**: This configuration is a **partial integration workaround**. It does **not** constitute a live CARLA simulation, and closed-loop vehicle actuation (throttle/steering commands) was not executed.

---

## 12. TensorRT Limitations on Apple Silicon

### Platform Incompatibility:
- NVIDIA TensorRT is proprietary software engineered specifically for NVIDIA CUDA Streaming Multiprocessors and Tensor Cores (`libnvinfer.so`).
- The Apple MacBook Air M1 uses an ARM64 unified-memory architecture with an integrated Apple Metal GPU. It physically lacks NVIDIA CUDA or Tensor hardware.

### Adaptation & Scope Disclosure:
- Rather than claiming TensorRT execution or mitigation, I evaluated **alternative performance benchmarks** available on this host:
  - Apple Metal Performance Shaders (`torch.device("mps")`) at **$9.38\text{ ms}$**.
  - Multi-threaded ONNX Runtime CPU at **$39.13\text{ ms}$**.
- This established rigorous quantitative baselines on edge hardware while transparently acknowledging platform constraints.

---

## 13. Weights & Biases Local Fallback & Cloud Logging

### Integration Reality & Fallback:
- While `wandb_eval.py` was authored to support experiment logging, the `wandb` library was not pre-installed in the execution environment, and an active cloud API key was not configured.
- Rather than halting evaluation, I implemented an **automated offline fallback mode**. The script evaluated all 1,496 validation images and atomically serialized the complete benchmark metrics, class distributions, and environment metadata to `metrics_eval.json`.
- **Honest Disclosure**: Live cloud synchronization to the online W&B dashboard was not executed; all telemetry was captured and verified through local structured JSON auditing.

---

## 14. Key Engineering Challenges and Solutions

1. **Dual Conda Environment C++ ABI Incompatibility**:
   - *Challenge*: Simultaneous imports of `rclpy` and `torch` caused library collisions and segmentation faults.
   - *Solution*: Built a non-destructive Unix domain socket IPC bridge with length-prefixed binary framing, cleanly isolating deep learning inference from ROS 2 messaging.
2. **Road-Surface Bleed in Monocular Depth Association**:
   - *Challenge*: Full bounding box disparity averaging incorporated background asphalt and road surface, skewing vehicle proximity scores.
   - *Solution*: Developed center 50% interior patch sampling with median disparity filtering, isolating obstacle bulk from background noise.
3. **Perspective Distortion in Road Curvature Fitting**:
   - *Challenge*: Fitting curves in perspective view produced unstable polynomials due to perspective convergence.
   - *Solution*: Calculated homography matrices for top-down Inverse Perspective Mapping, fitting curves in Bird's-Eye View and enforcing non-crossing corridor sanity bounds.
4. **Automated Testing Without External Test Dependencies**:
   - *Challenge*: Verifying complex perception math and data serialization without adding third-party test framework dependencies.
   - *Solution*: Authored `tests/test_perception_stack.py` using Python's standard-library `unittest` framework, validating calibration matrices, IoU logic, depth thresholds, and letterbox unscaling across 16 tests in $0.037\text{ s}$.

---

## 15. Architectural Trade-offs and Lessons Learned

1. **Classical Computer Vision vs. End-to-End Deep Learning**:
   - Classical lane detection provides deterministic explainability, mathematical corridor bounds, and zero training overhead. However, it is sensitive to lighting changes, pavement discoloration, and missing markings. Deep segmentation models (e.g., U-Net or Ultra-Fast-Lane) are more robust to visual artifacts but act as opaque black boxes and demand significantly higher compute.
2. **Monocular Relative Disparity vs. Metric LiDAR Sensing**:
   - Monocular depth estimation using MiDaS requires only a cheap single camera and runs efficiently on edge hardware. However, it provides only relative inverse disparity rather than true metric meters. For safety-critical autonomous braking, genuine metric ranging (via stereo cameras or LiDAR) is non-negotiable.
3. **IPC Socket Decoupling vs. Monolithic Process Architecture**:
   - While inter-process communication across a Unix domain socket introduces serialization and transfer overhead compared to in-process function calls (reflected in an end-to-end full perception pipeline latency of $\sim 140\text{ ms} - 150\text{ ms}$ and an initial cold handshake of $389.8\text{ ms}$ in Stage 3 integration testing), it completely eliminated C++ ABI library conflicts between ROS 2 and PyTorch, enabled independent environment management, and allowed the inference server to remain hot in memory while ROS 2 nodes restarted.
4. **Edge Model Sizing (YOLOv8 Nano vs. Larger Variants)**:
   - Selecting YOLOv8n enabled $106.6\text{ FPS}$ isolated inference on Apple M1, leaving ample compute headroom for tracking and lane detection. However, its small parameter capacity (3.2M parameters) limited recall on distant and partially occluded targets on KITTI (overall recall: $0.344$).

---

## 16. Current Limitations and Future Improvements

### Current Limitations:
- **Zero-Shot Domain Gap**: Off-the-shelf COCO weights were evaluated without fine-tuning on the KITTI dataset, yielding an aggregate mAP@50 of 0.343 and zero mapped predictions for the unmapped Cyclist class.
- **Absence of Raw Point Clouds**: Raw Velodyne `.bin` files were not stored locally, restricting 3D Camera-LiDAR demonstration to Ground-Truth Audit Mode.
- **Lack of Closed-Loop Control**: Perception outputs (bounding boxes, tracks, lane curves) are published to ROS 2 topics, but no lateral/longitudinal vehicle controllers (PID / Pure Pursuit / MPC) consume them to actuate a vehicle.
- **Simulated Sensor Feed**: Replays recorded KITTI frames and synthetic point clouds rather than interacting with a dynamic real-time simulator.

### Recommended Future Work:
1. **KITTI Model Fine-Tuning**: Train YOLOv8n on the 5,985 KITTI training partition with a unified 4-class configuration (`Car`, `People`, `Cyclist`, `Truck`) matching `kitti.yaml` to resolve class mismatches and boost detection accuracy.
2. **Genuine Point-Cloud Stream Ingestion**: Download sample raw Velodyne point clouds and fuse 3D point clusters with 2D bounding boxes using camera intrinsic/extrinsic projection.
3. **Deployment on NVIDIA Jetson Hardware**: Migrate the ROS 2 perception stack to an NVIDIA Jetson Orin platform with native Linux, CUDA, and FP16 TensorRT compilation.
4. **Closed-Loop Vehicle Actuation**: Connect detected lane polynomial offsets and obstacle proximity warnings to a vehicle trajectory planner and lateral PID steering controller.

---

## 17. Personal Learning Outcomes

This project provided invaluable hands-on engineering experience in building an end-to-end perception system:
- **Perception Mathematics**: Gained a deep mathematical understanding of camera projection matrices ($P_2$), coordinate frame rectification ($R_{0\_rect}$), extrinsic rigid-body transformations ($Tr_{velo}^{cam}$), and perspective homography.
- **Edge Acceleration**: Learned how to benchmark and profile neural networks across diverse backends, contrasting Apple Metal Performance Shaders with multi-threaded ONNX Runtime execution.
- **Middleware & Systems Architecture**: Developed practical proficiency in ROS 2 Humble node lifecycle management, QoS profiles, message synchronization, and Unix domain socket IPC design.
- **Empirical Rigor**: Learned that evaluation metrics must be interpreted with domain context—discovering how an unmapped class (`Cyclist`) directly impacts macro-averaged benchmark scores and demonstrates the necessity of domain adaptation.
- **Engineering Transparency**: Understood that in safety-critical robotics, acknowledging hardware boundaries, simulation workarounds, and prototype constraints is just as important as reporting high benchmark numbers.
