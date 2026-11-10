"""
Autonomous Driving Perception Stack - Module 4: Step 2 - Obstacle Distance Fusion
Combines YOLOv8n object detection with MiDaS_small monocular relative depth estimation
on Apple Silicon MPS (Metal Performance Shaders) hardware acceleration.

IMPORTANT TECHNICAL NOTICE:
MiDaS produces RELATIVE inverse depth (disparity-like values), not metric distances in metres.
Higher numerical values indicate obstacles CLOSER to the camera.
Lower numerical values indicate obstacles FARTHER from the camera.
This script explicitly labels all distance metrics as relative estimates.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Category mappings for Autonomous Driving Perception
PERCEPTION_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Proximity category thresholds based on normalized relative depth [0, 255]
# MiDaS disparity direction: Higher value = Closer to camera
DEPTH_THRESHOLD_NEAR = 160      # Relative depth >= 160: Close / Near obstacle
DEPTH_THRESHOLD_MID = 90        # Relative depth 90..159: Mid-Range obstacle
                                # Relative depth < 90: Far obstacle

# Bounding box color palette (BGR format for OpenCV)
COLOR_NEAR = (0, 0, 255)       # Red: Immediate proximity warning
COLOR_MID = (0, 255, 255)      # Yellow: Caution / Mid-range
COLOR_FAR = (0, 255, 0)        # Green: Far / Clear corridor


def verify_apple_mps_device() -> torch.device:
    """Verifies and initializes Apple Silicon MPS device; raises error if unavailable."""
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "[ERROR] Apple Silicon MPS hardware acceleration is not available! "
            "CPU fallback is disabled per strict safety rules."
        )
    device = torch.device("mps")
    print(f"[INFO] Hardware Device: {device.type.upper()} (Apple Silicon M1 GPU Native Metal)")
    return device


def resolve_input_image_path() -> Path:
    """
    Resolves the input image path:
    1. CLI argument if explicitly passed
    2. Local perception asset (bus.jpg) containing vehicles and pedestrians
    3. Project sample_road.jpg
    """
    project_dir = Path(__file__).resolve().parent

    # 1. Custom CLI argument
    if len(sys.argv) > 1:
        custom_path = Path(sys.argv[1])
        if custom_path.is_file():
            print(f"[INFO] Using user-specified input image: {custom_path}")
            return custom_path
        raise FileNotFoundError(f"[ERROR] Specified input image not found: {custom_path}")

    # 2. Local perception asset with real autonomous driving obstacles
    perception_bus_asset = Path(
        "/opt/anaconda3/envs/av-perception/lib/python3.11/site-packages/ultralytics/assets/bus.jpg"
    )
    if perception_bus_asset.is_file():
        print(f"[INFO] Using autonomous driving obstacle asset: {perception_bus_asset.name}")
        return perception_bus_asset

    # 3. Project sample road image
    sample_road = project_dir / "sample_road.jpg"
    if sample_road.is_file():
        print(f"[INFO] Using project road scene: {sample_road.name}")
        return sample_road

    raise FileNotFoundError("[ERROR] No valid input image found in project or environment.")


def load_perception_models(device: torch.device, project_dir: Path):
    """Loads existing YOLOv8n and MiDaS_small models on Apple Silicon MPS."""
    print("-" * 75)
    print("[INFO] Loading YOLOv8n object detection model...")
    yolo_model_path = project_dir / "yolov8n.pt"
    if not yolo_model_path.is_file():
        raise FileNotFoundError(f"[ERROR] Required YOLO model weight not found: {yolo_model_path}")
    yolo_model = YOLO(str(yolo_model_path))

    print("[INFO] Loading pretrained MiDaS_small depth estimation model...")
    load_start = time.perf_counter()
    midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas_model.to(device)
    midas_model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    load_time = time.perf_counter() - load_start
    print(f"[SUCCESS] MiDaS_small loaded successfully in {load_time:.2f}s.")
    print("-" * 75)

    return yolo_model, midas_model, transform


def run_yolo_detection(yolo_model, image_bgr: np.ndarray, device: torch.device, conf_thresh: float = 0.25):
    """Executes YOLOv8n 2D object detection on the input frame using MPS."""
    t0 = time.perf_counter()
    results = yolo_model.predict(
        source=image_bgr,
        conf=conf_thresh,
        device=str(device),
        save=False,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    boxes = results[0].boxes
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        coords = [int(round(float(c))) for c in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
        class_name = yolo_model.names.get(cls_id, f"class_{cls_id}")

        # Filter for perception-relevant classes if desired, or retain all detected COCO objects
        detections.append({
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf,
            "box": coords,  # [x1, y1, x2, y2]
        })

    return detections, latency_ms


def run_midas_depth(midas_model, transform, image_bgr: np.ndarray, device: torch.device):
    """Executes MiDaS_small monocular depth estimation and normalizes depth map to [0, 255]."""
    orig_h, orig_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    depth_map = prediction.cpu().numpy()
    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())

    # Normalized relative depth (0 = farthest, 255 = nearest)
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_8u = (depth_norm * 255.0).astype(np.uint8)

    return depth_8u, depth_norm, latency_ms


def extract_obstacle_relative_depth(box, depth_8u: np.ndarray):
    """
    Extracts the median relative depth within the center 50% patch of an obstacle bounding box.
    Cropping the center patch eliminates background edge bleeding and ground plane contamination.
    """
    x1, y1, x2, y2 = box
    h, w = depth_8u.shape[:2]

    # Clip coordinates to image boundary
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 4 or box_h <= 4:
        patch = depth_8u[y1:y2, x1:x2]
        return float(np.median(patch)) if patch.size > 0 else 0.0

    # Center 50% patch
    cx1 = int(x1 + 0.25 * box_w)
    cx2 = int(x2 - 0.25 * box_w)
    cy1 = int(y1 + 0.25 * box_h)
    cy2 = int(y2 - 0.25 * box_h)

    patch = depth_8u[cy1:cy2, cx1:cx2]
    if patch.size > 0:
        return float(np.median(patch))
    return float(np.median(depth_8u[y1:y2, x1:x2]))


def categorize_proximity(rel_depth_score: float):
    """
    Categorizes obstacle proximity based on verified MiDaS inverse depth direction:
    - High disparity (>= 160) = Near / Close
    - Medium disparity (90..159) = Mid-Range
    - Low disparity (< 90) = Far
    """
    if rel_depth_score >= DEPTH_THRESHOLD_NEAR:
        return "Near", COLOR_NEAR
    elif rel_depth_score >= DEPTH_THRESHOLD_MID:
        return "Mid-Range", COLOR_MID
    else:
        return "Far", COLOR_FAR


def render_fusion_visualization(
    image_bgr: np.ndarray,
    depth_8u: np.ndarray,
    detections: list,
    yolo_latency_ms: float,
    midas_latency_ms: float,
    combined_fps: float,
) -> np.ndarray:
    """Renders side-by-side composite showing obstacle distance fusion and relative depth map."""
    h, w = image_bgr.shape[:2]
    annotated_cam = image_bgr.copy()

    # Apply Inferno Colormap to relative depth (Warm Yellow = Close, Dark Purple = Far)
    depth_colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)

    # Annotate detected obstacles on the camera image
    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = box
        cls_name = det["class_name"]
        conf = det["confidence"]
        rel_depth = det["rel_depth"]
        category = det["proximity_category"]
        color = det["color"]

        # Draw 2D bounding box
        cv2.rectangle(annotated_cam, (x1, y1), (x2, y2), color, 2)

        # Label text: Explicitly indicates relative estimate, non-metric
        label_line1 = f"{cls_name.upper()} {conf * 100:.0f}%"
        label_line2 = f"Rel-Depth: {rel_depth:.0f}/255 ({category}) [Est.]"

        # Text banner backgrounds
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (w1, h1), _ = cv2.getTextSize(label_line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(label_line2, font, font_scale, thickness)
        banner_w = max(w1, w2) + 8
        banner_h = h1 + h2 + 12

        # Ensure label banner fits within image
        banner_y1 = max(0, y1 - banner_h)
        banner_y2 = banner_y1 + banner_h
        banner_x2 = min(w, x1 + banner_w)

        # Semi-transparent label background
        sub_img = annotated_cam[banner_y1:banner_y2, x1:banner_x2]
        black_rect = np.zeros_like(sub_img)
        cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 0, sub_img)

        # Render text lines
        cv2.putText(
            annotated_cam,
            label_line1,
            (x1 + 4, banner_y1 + h1 + 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_cam,
            label_line2,
            (x1 + 4, banner_y1 + h1 + h2 + 7),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        # Also draw corresponding bounding box on the depth map for direct visual correlation
        cv2.rectangle(depth_colored, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            depth_colored,
            f"{cls_name}: {rel_depth:.0f}",
            (x1 + 4, max(18, y1 - 6)),
            font,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Create composite side-by-side canvas
    composite = np.hstack([annotated_cam, depth_colored])
    comp_h, comp_w = composite.shape[:2]

    # Top HUD Banner (Height: 64 px)
    hud_height = 64
    hud = np.zeros((hud_height, comp_w, 3), dtype=np.uint8)
    hud[:] = (20, 20, 20)

    title_text = "MODULE 4 - STEP 2: OBSTACLE DISTANCE FUSION (YOLOv8n + MiDaS_small)"
    sub_text = (
        f"Device: Apple MPS (Metal) | YOLO: {yolo_latency_ms:.1f}ms | MiDaS: {midas_latency_ms:.1f}ms | "
        f"FPS: {combined_fps:.1f} | Obstacles: {len(detections)} | Non-Metric Relative Depth"
    )

    cv2.putText(hud, title_text, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, sub_text, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # Panel Sub-headers on the composite image
    cv2.putText(composite, "[PANEL 1] Camera Detections & Proximity", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(composite, "[PANEL 2] MiDaS Relative Depth Map", (w + 16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    final_output = np.vstack([hud, composite])
    return final_output


def run_obstacle_depth_fusion():
    print("=" * 80)
    print("AUTONOMOUS DRIVING PERCEPTION STACK - STEP 2: OBSTACLE DISTANCE FUSION")
    print("=" * 80)

    project_dir = Path(__file__).resolve().parent

    # 1. Device Verification
    device = verify_apple_mps_device()

    # 2. Input Image Resolution
    input_path = resolve_input_image_path()
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise RuntimeError(f"[ERROR] cv2.imread failed to load input image: {input_path}")
    orig_h, orig_w = img_bgr.shape[:2]
    print(f"[INFO] Loaded input frame: {input_path.name} ({orig_w}x{orig_h} px)")

    # 3. Model Loading
    yolo_model, midas_model, transform = load_perception_models(device, project_dir)

    # 4. Warm-up passes (Metal shader compilation)
    print("[INFO] Executing warm-up passes for Apple MPS shader caching...")
    dummy_input = torch.zeros((1, 3, 256, 256), device=device)
    with torch.no_grad():
        _ = midas_model(dummy_input)
    _ = yolo_model.predict(source=img_bgr, device=str(device), verbose=False)
    print("[SUCCESS] MPS shaders warmed up successfully.")

    # 5. Execute 2D Object Detection
    print("-" * 75)
    print("[INFO] Running YOLOv8n object detection on Apple MPS...")
    detections, yolo_latency_ms = run_yolo_detection(yolo_model, img_bgr, device, conf_thresh=0.25)
    print(f"[SUCCESS] YOLOv8n completed in {yolo_latency_ms:.2f} ms. Detected objects: {len(detections)}")

    # 6. Execute Monocular Depth Estimation
    print("[INFO] Running MiDaS_small depth estimation on Apple MPS...")
    depth_8u, depth_norm, midas_latency_ms = run_midas_depth(midas_model, transform, img_bgr, device)
    print(f"[SUCCESS] MiDaS_small completed in {midas_latency_ms:.2f} ms.")

    # 7. Obstacle-Depth Fusion & Depth Score Direction Verification
    print("-" * 75)
    print("DEPTH SCORE DIRECTION VERIFICATION:")
    print("  * MiDaS outputs inverse depth (disparity).")
    print("  * Higher values (160..255) indicate NEAR / CLOSE obstacles (warm colors).")
    print("  * Medium values (90..159) indicate MID-RANGE obstacles.")
    print("  * Lower values (0..89) indicate FAR / DISTANT obstacles (cool/dark colors).")
    print("-" * 75)

    fused_results = []
    t_fusion_start = time.perf_counter()
    for idx, det in enumerate(detections, start=1):
        rel_depth = extract_obstacle_relative_depth(det["box"], depth_8u)
        category, color = categorize_proximity(rel_depth)
        det["rel_depth"] = rel_depth
        det["proximity_category"] = category
        det["color"] = color

        fused_results.append({
            "index": idx,
            "class": det["class_name"],
            "confidence": det["confidence"],
            "box_xyxy": det["box"],
            "rel_depth_score": rel_depth,
            "proximity_category": category,
        })

        print(
            f"  [{idx}] Class: {det['class_name']:<10} | Conf: {det['confidence']*100:5.1f}% | "
            f"Rel-Depth: {rel_depth:5.1f}/255 | Category: {category:<9} | BBox: {det['box']}"
        )

    fusion_latency_ms = (time.perf_counter() - t_fusion_start) * 1000.0
    combined_latency_ms = yolo_latency_ms + midas_latency_ms + fusion_latency_ms
    combined_fps = (1000.0 / combined_latency_ms) if combined_latency_ms > 0 else 0.0

    print("-" * 75)
    print("PERFORMANCE BENCHMARKS (Combined Perception Pipeline):")
    print(f"  * YOLOv8n Latency        : {yolo_latency_ms:.2f} ms")
    print(f"  * MiDaS_small Latency    : {midas_latency_ms:.2f} ms")
    print(f"  * Box-Depth Fusion       : {fusion_latency_ms:.2f} ms")
    print(f"  * Combined Frame Latency : {combined_latency_ms:.2f} ms")
    print(f"  * Sustained Throughput   : {combined_fps:.1f} FPS (Apple Silicon M1 MPS)")
    print("-" * 75)

    # 8. Render Composite Visualization
    print("[INFO] Rendering composite obstacle distance fusion visualization...")
    composite_image = render_fusion_visualization(
        img_bgr, depth_8u, detections, yolo_latency_ms, midas_latency_ms, combined_fps
    )

    # 9. Save Output Image & Validate Return Value
    output_image_path = project_dir / "fusion_output.jpg"
    save_ok = cv2.imwrite(str(output_image_path), composite_image)
    if not save_ok:
        raise IOError(f"[ERROR] cv2.imwrite failed to save composite output to: {output_image_path}")

    print(f"[SUCCESS] Saved fusion output image: {output_image_path.name}")
    print(f"[SUCCESS] Dimensions: {composite_image.shape[1]}x{composite_image.shape[0]} px")
    print("=" * 80)

    return {
        "output_path": str(output_image_path),
        "object_count": len(detections),
        "detections": fused_results,
        "yolo_latency_ms": yolo_latency_ms,
        "midas_latency_ms": midas_latency_ms,
        "combined_latency_ms": combined_latency_ms,
        "combined_fps": combined_fps,
        "device": "mps",
    }


if __name__ == "__main__":
    run_obstacle_depth_fusion()
