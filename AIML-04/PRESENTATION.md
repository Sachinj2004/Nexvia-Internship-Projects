# Nexvia Project 2 (AIML-04): Autonomous Driving Perception Stack
## Final Technical Presentation & Engineering Defense

- **Target Duration**: Approximately 10 Minutes (12 Slides)
- **Target Hardware**: Apple MacBook Air M1 (Apple Silicon arm64, 8 GB Unified Memory, macOS 26.6.2)
- **Status**: Modular Perception Prototype Defense

---

### Slide 1: Title, Executive Summary & Prototype Scope
**Target Duration**: ~45 Seconds

#### Slide Content
- **Project Title**: Autonomous Driving Perception Stack (AIML-04)
- **Subtitle**: A Modular Real-Time Perception Prototype on Apple Silicon
- **Presenter**: Nexvia AIML Intern
- **Hardware Platform**: Apple Silicon M1 (Apple M1 GPU Metal Performance Shaders / CPU)
- **Executive Summary**:
  - Engineered a modular perception prototype unifying 2D obstacle detection, multi-object tracking, classical lane detection, monocular depth heuristics, and 3D camera-LiDAR calibration projection.
  - Isolated neural network inference benchmarked at **9.38 ms** on Apple Silicon MPS ($106.6\text{ FPS}$) and **39.13 ms** on ONNX Runtime CPU ($25.6\text{ FPS}$).
  - Evaluated on all **1,496 KITTI validation images** with offline Weights & Biases experiment telemetry.
  - Automated local regression test suite: **16 of 16 tests passing (100%) in 0.037 seconds**.

#### Referenced Project Artifacts
- Source repository: `Autonomous-Driving-Perception-Stack`
- Benchmark metrics: `metrics_eval.json`
- Test suite: `tests/test_perception_stack.py`

> [!NOTE] Speaker Notes (Target Time: ~45s)
> *"Hello everyone. Today I am presenting our Autonomous Driving Perception Stack, developed for the Nexvia AIML-04 project. We built a modular perception prototype designed to address fundamental autonomous vehicle vision tasks: detecting obstacles, tracking vehicles across video, extracting lane corridors, estimating relative depth, and projecting 3D LiDAR data onto camera coordinates. We developed and evaluated this prototype on an Apple Silicon M1 machine, measuring an isolated model inference speed of 9.38 milliseconds on the GPU and 39 milliseconds on ONNX CPU. All components are validated by an automated local regression test suite where all 16 tests pass. Over the next ten minutes, I will share our pipeline architecture, empirical validation results, and honest technical limitations."*

---

### Slide 2: End-to-End Modular Prototype Architecture
**Target Duration**: ~55 Seconds

#### Slide Content
- **Multi-Modal Perception Head Structure**:
  - **Sensors Input**: Forward monocular camera frames ($1242 \times 375$ KITTI / $1280 \times 720$ video) and KITTI calibration matrices.
  - **Obstacle Perception**: YOLOv8n ($640 \times 640$) $\rightarrow$ ByteTrack two-stage data association across consecutive frames.
  - **Lane Corridor Perception**: OpenCV HLS color filtering + Canny edge + IPM Bird's-Eye View + sliding-window 2nd-order polynomial fitting.
  - **Depth & Proximity Reasoning**: MiDaS_small monocular depth $\rightarrow$ center 50% interior patch median $\rightarrow$ 3-tier proximity categorization (`NEAR`, `MID-RANGE`, `FAR`).
  - **Calibration Projection**: Rigid-body geometric transformation ($P_2 \cdot R_{0\_rect} \cdot Tr_{velo}^{cam}$) mapping 3D coordinates to camera pixels, utilizing a Ground-Truth Audit fallback when raw Velodyne `.bin` scans are absent locally.
  - **Middleware**: CARLA-compatible ROS 2 stream publishers and IPC bridge nodes.

```
[Raw RGB Video / KITTI Camera]
   │
   ├──> [YOLOv8n Object Detection] ──> [ByteTrack Multi-Object Tracker] ──> Persistent Tracks
   │                                                 │
   ├──> [OpenCV Lane Detection] ───────────────────┼──> Curvature & Offset (Estimated)
   │    (HLS + Canny + IPM + Polynomial Fit)         │
   │                                                 ▼
   ├──> [MiDaS Monocular Depth] ─────────> [Obstacle Distance Fusion] ──> Proximity Warnings
   │                                     (Center 50% Interior Patch)     (NEAR / MID / FAR)
   │
[KITTI Calibration Matrices] ───────────> [Calibration Projection] ────> Spatial Association
                                         (Ground-Truth Audit Mode)
```

> [!NOTE] Speaker Notes (Target Time: ~55s)
> *"Here is the architecture of our modular perception prototype. Rather than building a single monolithic model, we structured perception into specialized, decoupled stages. Incoming camera frames feed simultaneously into two parallel paths: deep neural networks for obstacle detection and monocular relative depth, and deterministic classical computer vision for lane detection. Bounding boxes are tracked over time using ByteTrack. Obstacle boxes are then fused with our depth map using an interior patch sampling method that avoids road-surface bleed. For sensor fusion, our codebase implements calibration-based projection matrices to map 3D spatial coordinates into 2D camera coordinates, utilizing a Ground-Truth Audit fallback when raw Velodyne scans are not stored locally."*

---

### Slide 3: Dataset Engineering & KITTI Benchmark Partitioning
**Target Duration**: ~45 Seconds

#### Slide Content
- **Benchmark Dataset**: KITTI Vision Benchmark Suite (Object Detection).
- **Partitioning Statistics**:
  - Total Raw Dataset: **7,481 scenes** with camera images, calibration files, and 3D labels.
  - Training Partition (80%): **5,985 images** converted to normalized YOLO format.
  - Validation Partition (20%): **1,496 images** converted to normalized YOLO format.
  - Validation Ground Truth: **7,660 annotated bounding boxes** across 4 canonical classes.
- **Validation Class Distribution**:
  - `Car`: 6,230 annotations ($81.3\%$)
  - `People`: 926 annotations ($12.1\%$)
  - `Cyclist`: 306 annotations ($4.0\%$)
  - `Truck`: 198 annotations ($2.6\%$)
- **Reproducible Pipeline**: `convert_kitti_to_yolo.py`, `partition_kitti_dataset.py`, `kitti.yaml`.

| Split | Images | Bounding Boxes | Car | People | Cyclist | Truck |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train (80%)** | 5,985 | ~30,600 | - | - | - | - |
| **Val (20%)** | 1,496 | 7,660 | 6,230 (81.3%) | 926 (12.1%) | 306 (4.0%) | 198 (2.6%) |
| **Total** | **7,481** | **~38,260** | - | - | - | - |

> [!NOTE] Speaker Notes (Target Time: ~45s)
> *"For empirical validation, we prepared the real-world KITTI benchmark. We wrote automated scripts to convert KITTI's camera calibration and 3D labels into normalized bounding box coordinates, and split the 7,481 samples into an 80-20 partition: 5,985 training images and 1,496 validation images. Our validation set contains 7,660 ground-truth boxes. In line with European driving conditions, passenger cars make up about 81% of the annotations, while pedestrians, cyclists, and trucks make up the remaining 19%."*

---

### Slide 4: 2D Obstacle Detection & Apple Silicon MPS Acceleration
**Target Duration**: ~50 Seconds

#### Slide Content
- **Model**: YOLOv8n (Nano architecture, 3.2M parameters, 6.55 MB weight file).
- **Execution Engine**: Apple Silicon MPS (`torch.device("mps")`) via Apple Metal GPU shaders.
- **Isolated Inference Benchmarks**:
  - **Native Apple Silicon MPS GPU**: **9.38 ms** (equivalent to $106.6\text{ FPS}$).
  - *Scope Clarification*: 106.6 FPS refers strictly to the isolated neural network inference pass, not the full multi-sensor pipeline.
  - **ONNX Runtime CPU**: **39.13 ms** (equivalent to $25.6\text{ FPS}$).
  - **Speedup**: Native Apple Silicon MPS GPU is **4.17x faster** than multi-threaded CPU inference.
- **Traffic Safety Class Filtering**: Restricts detections to `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck` with confidence threshold $\ge 0.25$.
- **Artifacts**: `test_detection.py`, verified output image `detection_output.jpg`.

> [!NOTE] Speaker Notes (Target Time: ~50s)
> *"For 2D obstacle detection, we used the compact YOLOv8 Nano model. Because our test environment is an Apple Silicon M1 laptop without an NVIDIA GPU, we ran PyTorch directly on Apple's Metal Performance Shaders GPU backend. In isolated inference benchmarking, YOLOv8n ran in 9.38 milliseconds per frame—which translates to over 100 frames per second for the model pass alone. We filter the raw detections to retain only safety-critical traffic participants like cars, pedestrians, cyclists, and trucks, ensuring downstream modules only process relevant road obstacles."*

---

### Slide 5: Multi-Object Tracking with ByteTrack
**Target Duration**: ~45 Seconds

#### Slide Content
- **Algorithm**: ByteTrack (Two-stage association using bounding box overlap and Kalman filtering).
- **Core Benefit**: Associates both high-confidence and low-confidence detection boxes across consecutive video frames to prevent track fragmentation.
- **Implementation Highlights**:
  - Evaluated on continuous road driving video via `track_video.py`.
  - Maintains persistent track IDs and draws dynamic centroid trajectory trails.
  - Successfully tracks vehicles through temporary occlusions and motion blur.
- **Artifact**: Verified output video `tracked_output.mp4` (5.7 MB, 100+ frames processed).

> [!NOTE] Speaker Notes (Target Time: ~45s)
> *"Single-frame object detection doesn't tell us how objects move over time. To solve this, we integrated the ByteTrack tracking algorithm. Unlike older trackers that discard low-confidence detections, ByteTrack keeps lower-confidence boxes and uses Kalman filtering to bridge temporary occlusions—for instance, when a pedestrian walks behind a pole or a car experiences brief motion blur. We validated this by processing road video and saving the tracked output with unique track IDs to tracked_output.mp4."*

---

### Slide 6: Classical Computer Vision Lane Detection & Estimated Geometry
**Target Duration**: ~50 Seconds

#### Slide Content
- **Approach**: Deterministic classical computer vision using OpenCV and NumPy (transparent, explainable heuristics).
- **Pipeline Stages**:
  1. *Color Filtering*: HLS color space filtering for white ($L > 180$) and yellow ($H \in [15, 35], S > 70, L > 60$).
  2. *Edge Extraction*: Gaussian smoothing ($5 \times 5$) + Canny edge detector ($50, 150$).
  3. *Trapezoidal ROI*: Restricts search area to the forward road corridor.
  4. *Inverse Perspective Mapping (IPM)*: Transforms perspective image into a top-down Bird's-Eye View.
  5. *Sliding-Window Polynomial Fit*: 9 vertical search windows tracking histogram peaks, fitting 2nd-order curves: $x = Ay^2 + By + C$.
  6. *Corridor Sanity Validation*: Verifies non-crossing boundaries and enforces plausible lane separation ($2.2\text{ m} \le w \le 5.2\text{ m}$).
  7. *Estimated Geometric Measurements*: Computes estimated radius of curvature and lateral vehicle offset based on standard highway lane assumptions ($3.7\text{ m} / 700\text{ px}$ width, $30\text{ m} / 720\text{ px}$ forward span).
- **Artifact**: `lane_detection.py`, verified output image `lane_detection_output.jpg`.

> [!NOTE] Speaker Notes (Target Time: ~50s)
> *"For lane detection, we implemented a classical computer vision pipeline so that every geometric measurement is fully explainable. We filter the road image in HLS color space to isolate white and yellow markings, apply Canny edge detection, and transform the road into a top-down Bird's-Eye View using Inverse Perspective Mapping. A sliding-window search finds the lane boundaries and fits second-order polynomials. The code checks that the boundaries do not cross and that the corridor width is plausible. Finally, using an assumed 3.7-meter highway lane width conversion, it computes estimated road curvature and the vehicle's lateral offset from lane center."*

---

### Slide 7: Monocular Relative Depth & Obstacle Distance Fusion
**Target Duration**: ~50 Seconds

#### Slide Content
- **Depth Model**: Intel ISL `MiDaS_small` executed on Apple Silicon MPS.
- **Metric Nature**: Monocular depth produces **relative inverse disparity** (higher value = closer to camera), NOT absolute physical meters.
- **Center 50% Interior Patch Sampling**:
  - Full-box averaging contaminated depth estimates with background asphalt, buildings, and sky.
  - Solution: Samples the median depth strictly from the center 50% interior patch ($[cx_1, cy_1, cx_2, cy_2]$), cleanly rejecting edge bleeding.
- **Three-Tier Safety Categorization**:
  - `NEAR` (Disparity $\ge 160$): Immediate proximity warning (Red box).
  - `MID-RANGE` (Disparity $90..159$): Caution (Yellow box).
  - `FAR` (Disparity $< 90$): Clear corridor (Green box).
- **Artifact**: `obstacle_depth_fusion.py`, verified output `fusion_output.jpg`.

> [!NOTE] Speaker Notes (Target Time: ~50s)
> *"To provide depth awareness from a single camera, we integrated the MiDaS small model. It is important to emphasize that monocular depth produces relative inverse disparity—meaning larger numbers mean closer objects—not exact metric distances in meters. A common challenge in depth fusion is that averaging an entire bounding box blends in background road or sky. We solved this by sampling the median depth only from the center 50% interior patch of each detection box. The prototype then tags each obstacle into one of three color-coded proximity zones: Near, Mid-Range, or Far."*

---

### Slide 8: 3D Camera-LiDAR Calibration Projection & Spatial Association
**Target Duration**: ~50 Seconds

#### Slide Content
- **Methodology**: Rigid-body geometric transformation mapping 3D coordinates $(X, Y, Z)$ into 2D camera pixel coordinates $(u, v)$ via calibration matrices:
  $$\begin{bmatrix} u \cdot z \\ v \cdot z \\ z \end{bmatrix} = P_2 \cdot R_{0\_rect}^{4\times 4} \cdot Tr_{velo}^{cam} \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$
- **Implementation & Data Disclosure**:
  - The ~29 GB raw KITTI Velodyne `.bin` point-cloud archive was **not downloaded** to local disk.
  - The codebase implements full point-cloud parsing and projection logic for genuine Velodyne files when present.
  - In execution, the script ran in **Ground-Truth Audit Mode**, projecting verified 3D physical sensor coordinates from KITTI `label_2` annotations onto the camera image and spatially associating them with 2D YOLO bounding boxes.
- **Output Artifact**: `camera_lidar_fusion.py`, verified visualization `camera_lidar_fusion_output.jpg` (explicitly labeled as Ground-Truth Audit Mode).

> [!NOTE] Speaker Notes (Target Time: ~50s)
> *"To connect 2D camera vision with 3D space, we implemented calibration-based Camera-LiDAR projection using KITTI calibration matrices. By applying the camera projection, rectification, and extrinsic transformation matrices, our code maps 3D spatial points directly onto camera pixels. Because the 29-gigabyte raw Velodyne point-cloud archive was not downloaded locally, live point-cloud fusion was not executed. Instead, our script demonstrated calibration projection in Ground-Truth Audit Mode, projecting verified physical 3D sensor measurements from KITTI annotations onto the image and linking them to 2D detection boxes."*

---

### Slide 9: Edge Benchmarks: Native PyTorch MPS vs. ONNX Runtime CPU
**Target Duration**: ~45 Seconds

#### Slide Content
- **Model Export**: Exported YOLOv8n to Open Neural Network Exchange format (`yolov8n.onnx`, 12.09 MB, opset 17).
- **Measured Isolated Inference Benchmarks (100 Statistical Runs)**:
  - **Native PyTorch MPS (Apple Silicon M1 GPU)**: **9.38 ms** average latency ($106.6\text{ FPS}$).
  - **ONNX Runtime CPU (4 Intra-Op Threads)**: **39.13 ms** average latency ($25.6\text{ FPS}$).
  - **Speedup**: Native Apple Silicon MPS GPU execution is **4.17x faster** than multi-threaded CPU inference.
- **Scope Note**: These measurements represent isolated model inference benchmarking; complete perception pipeline throughput involves additional computer vision and tracking steps.
- **Deployment Script**: `onnx_inference.py` provides standalone CPU execution with custom letterboxing and NMS, independent of the Ultralytics package.
- **Artifacts**: `export_yolo_onnx.py`, `yolov8n.onnx`, `onnx_inference.py`, `benchmark_yolo_onnx.py`.

> [!NOTE] Speaker Notes (Target Time: ~45s)
> *"To evaluate edge viability, we exported YOLOv8n to ONNX format and tested both CPU and GPU backends across 100 isolated benchmark runs. On our Apple M1 hardware, native PyTorch running on Metal GPU shaders achieved 9.38 milliseconds per frame in isolated inference, while ONNX Runtime running on CPU with four threads took 39.13 milliseconds. While the GPU backend is over four times faster, both benchmarks establish quantitative baseline performance for isolated model inference on edge hardware."*

---

### Slide 10: CARLA-Compatible ROS 2 Stream Publishers & Middleware
**Target Duration**: ~45 Seconds

#### Slide Content
- **Middleware Prototype**: ROS 2 Humble nodes communicating over asynchronous IPC topics.
- **Implemented Node Architecture (`ros2_nodes/`)**:
  - `carla_camera_publisher.py`: Replays offline KITTI frames and publishes CARLA-compatible standard messages on `/carla/ego_vehicle/camera/rgb/image_raw` and camera info.
  - `carla_lidar_publisher.py`: Publishes CARLA-schema `sensor_msgs/PointCloud2` messages.
  - `perception_ros_bridge.py`: Subscribes to camera and LiDAR topics, maintaining synchronized frame queues.
  - `perception_inference_ipc.py`: Consumes synchronized data and publishes detected bounding boxes, lane polynomials, and proximity tags.
- **Explicit Technical Disclosure**:
  - These nodes are **CARLA-compatible stream publishers and bridge nodes** built to match CARLA's topic and message schemas.
  - **An actual CARLA Unreal Engine simulation server was not run** due to Apple Silicon macOS platform constraints.

> [!NOTE] Speaker Notes (Target Time: ~45s)
> *"For vehicle middleware, we implemented four ROS 2 Humble nodes in the ros2_nodes directory. To support future simulator integration, our camera and LiDAR publisher nodes emulate CARLA's standard topic names and message formats using offline replay. Our bridge node synchronizes incoming image and LiDAR messages, and our inference node publishes the resulting bounding boxes and lane curves. I want to clearly state that this is a CARLA-compatible stream prototype—an actual CARLA Unreal Engine server was not executed because CARLA does not natively support macOS on Apple Silicon."*

---

### Slide 11: Empirical Evaluation & Weights & Biases Telemetry
**Target Duration**: ~60 Seconds

#### Slide Content
- **Evaluation Engine**: Custom offline evaluation harness (`wandb_eval.py`) evaluated across all **1,496 KITTI validation images**.
- **Execution Performance**: Full validation completed in **44.4 seconds** on Apple Silicon MPS (aggregate throughput: **33.7 FPS** at batch size 4).
- **Verified Benchmark Metrics (`metrics_eval.json`)**:
  - **Overall 4-Class Macro Average**: $\text{mAP@50} = \mathbf{0.343}$, $\text{mAP@50-95} = \mathbf{0.180}$, $\text{Precision} = \mathbf{0.389}$, $\text{Recall} = \mathbf{0.344}$.
  - `Car`: $\text{Precision} = 0.706$, $\text{Recall} = 0.603$, $\text{mAP@50} = \mathbf{0.655}$, $\text{mAP@50-95} = \mathbf{0.366}$
  - `People`: $\text{Precision} = 0.474$, $\text{Recall} = 0.407$, $\text{mAP@50} = \mathbf{0.395}$, $\text{mAP@50-95} = \mathbf{0.169}$
  - `Truck`: $\text{Precision} = 0.375$, $\text{Recall} = 0.366$, $\text{mAP@50} = \mathbf{0.323}$, $\text{mAP@50-95} = \mathbf{0.184}$
  - `Cyclist`: $\text{mAP@50} = \mathbf{0.0}$ (unsupported by COCO-pretrained YOLOv8n mapping).
- **Class-Mapping Mismatch Analysis**:
  - Stock COCO YOLOv8n has no dedicated KITTI Cyclist class mapping.
  - KITTI defines `Cyclist` as a distinct primary ground-truth class with 306 validation annotations.
  - Because the off-the-shelf COCO model has no dedicated Cyclist output mapping, its benchmark AP is recorded as 0.0.
  - Factoring Cyclist's 0.0 AP into the 4-class macro-average reduces overall aggregate mAP@50 from 0.458 (evaluated 3 classes) to 0.343, quantitatively illustrating the necessity of dataset-specific fine-tuning.

> [!NOTE] Speaker Notes (Target Time: ~60s)
> *"We evaluated our detection model on all 1,496 KITTI validation images using our evaluation script, wandb_eval.py. The entire validation ran in 44.4 seconds on the Apple GPU. On passenger cars, the model achieved a 0.706 precision and 0.655 mAP@50. On pedestrians and trucks, it scored 0.395 and 0.323 mAP@50. However, our evaluation documented an important class-mapping mismatch: stock COCO YOLOv8n has no dedicated KITTI Cyclist class mapping. KITTI defines Cyclist as a distinct category with 306 validation boxes, but because our off-the-shelf model mapping did not output a unified Cyclist class, its benchmark score is recorded as 0.0. Factoring that into the 4-class macro-average brings overall mAP@50 to 0.343, clearly documenting the need for domain-specific fine-tuning."*

---

### Slide 12: Compliance Audit, Limitations, Workarounds & Roadmap
**Target Duration**: ~60 Seconds

#### Slide Content
- **Transparent Assignment Compliance Audit**:
  1. *KITTI Fine-Tuning*: Evaluated stock COCO YOLOv8n; conversion and partition pipelines are complete, but model re-training on KITTI was not performed.
  2. *CARLA Simulation*: CARLA UE4 server is unsupported on Apple Silicon macOS; a **partial integration workaround** was implemented using CARLA-compatible ROS 2 stream publishers and bridge nodes matching CARLA topic and message schemas, but this does not replace or fully mitigate an actual live CARLA closed-loop simulation.
  3. *NVIDIA TensorRT*: Blocked by macOS non-CUDA hardware platform; evaluated using **alternative performance benchmarks** on Apple Silicon MPS (9.38 ms) and multi-threaded ONNX Runtime CPU (39.13 ms), rather than claiming mitigation or equivalent execution of TensorRT.
  4. *Closed-Loop Vehicle Control*: Prototype produces perception outputs (tracks, lane curves, proximity alerts); vehicle actuation (throttle/steering commands) was not implemented.
- **Automated Local Regression Test Suite**:
  - Created [`tests/test_perception_stack.py`](tests/test_perception_stack.py).
  - Built using Python's standard-library `unittest` framework.
  - **16 of 16 tests passing (100%) in 0.037 seconds**, validating calibration math, IoU calculations, depth heuristics, IPM matrices, letterbox unscaling math, and atomic metrics serialization.
- **Future Engineering Roadmap**:
  - Fine-tune YOLOv8 on the 5,985 KITTI training partition with a unified Cyclist class.
  - Deploy onto an NVIDIA Jetson platform with FP16 TensorRT compilation.
  - Connect perception outputs to a lateral PID/MPC steering controller.

> [!NOTE] Speaker Notes (Target Time: ~60s)
> *"To conclude, we emphasize rigorous engineering transparency. We performed an honest compliance audit against assignment requirements. Due to our macOS Apple Silicon hardware, a live CARLA simulator and NVIDIA TensorRT could not be run. For CARLA, we built CARLA-compatible ROS 2 stream publishers as a partial integration workaround, while acknowledging this does not replace a live closed-loop simulation. For TensorRT, we evaluated alternative performance benchmarks on Apple MPS at 9.38 milliseconds and ONNX CPU at 39 milliseconds. In addition, we built an automated local regression test suite using Python's standard-library unittest framework, where all 16 tests pass in 37 milliseconds, verifying our calibration math, depth heuristics, and data safety. In the future, fine-tuning on KITTI to recognize cyclists and deploying on NVIDIA Jetson hardware are the natural next steps. Thank you, and I welcome any questions."*
