#!/usr/bin/env python3
"""
camera_lidar_fusion.py
Module: 3D Camera-LiDAR Extrinsic Calibration Projection & 3D Sensor Evaluation.

Features:
  1. Genuine LiDAR Mode (when .bin point cloud file exists):
     - Ingests raw float32 [x, y, z, reflectance] Velodyne scans with malformed file validation.
     - Projects 3D LiDAR points into 2D camera coordinates using KITTI calibration matrices
       (P2 @ R0_rect_4x4 @ Tr_velo_to_cam_4x4).
     - Filters points behind camera (Z <= 0.1m) and out of image bounds.
     - Spatially associates enclosed points within YOLOv8 2D detection boxes.
     - Computes per-obstacle metric distance (median depth) and point counts.
  2. Ground-Truth Audit Mode (when .bin file is missing):
     - Uses verified KITTI label_2 3D physical sensor locations (measured by Velodyne during data capture).
     - Associates 3D physical ground truth with YOLO 2D detections.
     - Explicitly labels output as 'Ground-Truth Audit Mode (Velodyne .bin absent)', NEVER claiming point-cloud fusion.
  3. Robustness:
     - Strict validation for image loading, calibration matrices, bounding boxes, and malformed files.
"""

import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Autonomous driving road obstacle classes
RELEVANT_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

CONF_THRESHOLD = 0.25


def parse_kitti_calib(calib_file: Path) -> Dict[str, np.ndarray]:
    """
    Parses KITTI calibration text file for P2, R0_rect, and Tr_velo_to_cam.
    Validates shapes and finiteness.
    """
    if not calib_file.is_file():
        raise FileNotFoundError(f"[ERROR] Calibration file missing: {calib_file}")

    calib_data = {}
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            key = parts[0].rstrip(":")
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError as e:
                raise ValueError(f"[ERROR] Malformed numeric value in calib file {calib_file}: {e}")

            if key == "P2":
                if len(values) != 12:
                    raise ValueError(f"[ERROR] P2 matrix in {calib_file} must have 12 values, found {len(values)}")
                calib_data["P2"] = np.array(values, dtype=np.float64).reshape((3, 4))
            elif key == "R0_rect":
                if len(values) != 9:
                    raise ValueError(f"[ERROR] R0_rect in {calib_file} must have 9 values, found {len(values)}")
                calib_data["R0_rect"] = np.array(values, dtype=np.float64).reshape((3, 3))
            elif key == "Tr_velo_to_cam":
                if len(values) != 12:
                    raise ValueError(f"[ERROR] Tr_velo_to_cam in {calib_file} must have 12 values, found {len(values)}")
                calib_data["Tr_velo_to_cam"] = np.array(values, dtype=np.float64).reshape((3, 4))

    required_keys = ["P2", "R0_rect", "Tr_velo_to_cam"]
    for k in required_keys:
        if k not in calib_data:
            raise ValueError(f"[ERROR] Missing required calibration matrix '{k}' in {calib_file}")
        if not np.all(np.isfinite(calib_data[k])):
            raise ValueError(f"[ERROR] Non-finite values detected in calibration matrix '{k}' in {calib_file}")

    return calib_data


def parse_kitti_ground_truth_3d(label_file: Path) -> List[Dict[str, Any]]:
    """
    Parses KITTI label_2 text file extracting 2D bounding boxes and
    true 3D physical sensor locations (x, y, z in camera coordinates in meters).
    """
    if not label_file.is_file():
        return []

    objects = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "DontCare":
                continue
            cls_type = parts[0]
            try:
                # 2D Bounding Box: [x1, y1, x2, y2]
                bbox_2d = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                # 3D Dimensions: height, width, length
                dim_3d = [float(parts[8]), float(parts[9]), float(parts[10])]
                # 3D Location in camera coords (X, Y, Z in meters)
                loc_3d = [float(parts[11]), float(parts[12]), float(parts[13])]
            except (ValueError, IndexError):
                continue

            if not all(math.isfinite(v) for v in bbox_2d + dim_3d + loc_3d):
                continue

            metric_dist = float(np.linalg.norm(loc_3d))
            objects.append({
                "type": cls_type,
                "bbox_2d": bbox_2d,
                "dim_3d": dim_3d,
                "loc_3d": loc_3d,
                "metric_distance_m": round(metric_dist, 2),
                "depth_z_m": round(loc_3d[2], 2)
            })
    return objects


def build_projection_matrix(calib_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs 3x4 combined projection matrix M_proj = P2 * R0_rect_4x4 * Tr_velo_to_cam_4x4."""
    p2 = calib_data["P2"]
    r0_rect = np.eye(4, dtype=np.float64)
    r0_rect[:3, :3] = calib_data["R0_rect"]

    tr_velo_to_cam = np.eye(4, dtype=np.float64)
    tr_velo_to_cam[:3, :4] = calib_data["Tr_velo_to_cam"]

    extrinsic_rect = r0_rect @ tr_velo_to_cam
    proj_matrix = p2 @ extrinsic_rect
    return proj_matrix, extrinsic_rect


def load_velodyne_points(bin_path: Path) -> Optional[np.ndarray]:
    """
    Loads genuine KITTI binary float32 Velodyne point cloud (stride = 4: x, y, z, reflectance).
    Validates file size (must be non-empty multiple of 16 bytes) and finiteness.
    """
    if not bin_path.is_file():
        return None

    size_bytes = bin_path.stat().st_size
    if size_bytes == 0 or size_bytes % 16 != 0:
        print(f"[WARN] Malformed Velodyne file size ({size_bytes} bytes). Must be multiple of 16. Skipping.")
        return None

    try:
        pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        if not np.all(np.isfinite(pts)):
            print("[WARN] Non-finite coordinates in Velodyne scan. Filtering...")
            pts = pts[np.all(np.isfinite(pts), axis=1)]
        return pts.astype(np.float64)
    except Exception as e:
        print(f"[WARN] Failed to parse Velodyne scan {bin_path}: {e}")
        return None


def calculate_box_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates 2D IoU between [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return float(inter_area / union_area) if union_area > 0.0 else 0.0


def run_camera_lidar_fusion(stem: str = "000000") -> Dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    kitti_dir = project_dir / "data" / "kitti" / "training"
    img_path = kitti_dir / "image_2" / f"{stem}.png"
    calib_path = kitti_dir / "calib" / f"{stem}.txt"
    label_path = kitti_dir / "label_2" / f"{stem}.txt"
    velo_path = kitti_dir / "velodyne" / f"{stem}.bin"
    model_path = project_dir / "yolov8n.pt"
    output_path = project_dir / "camera_lidar_fusion_output.jpg"

    print("=" * 80)
    print("3D CAMERA-LIDAR CALIBRATION PROJECTION & SENSOR EVALUATION")
    print("=" * 80)

    # 1. Output safety check: abort if output file already exists
    if output_path.exists():
        raise FileExistsError(f"[ABORT] Output file {output_path.name} already exists! Halting per safety rules.")

    # 2. Validate input image
    if not img_path.is_file():
        raise FileNotFoundError(f"[ERROR] Camera image missing: {img_path}")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError(f"[ERROR] cv2.imread failed to load valid image from: {img_path}")
    h, w = img_bgr.shape[:2]
    print(f"[INFO] Frame Identifier         : {stem}")
    print(f"[INFO] Camera Image Loaded      : {img_path.name} ({w} x {h} px, 3 channels)")

    # 3. Parse and validate calibration matrices
    calib_data = parse_kitti_calib(calib_path)
    proj_matrix, extrinsic_rect = build_projection_matrix(calib_data)
    print(f"[INFO] Calibration Matrices     : P2 (3x4), R0_rect (3x3), Tr_velo_to_cam (3x4) verified.")

    # 4. YOLOv8n 2D object detection on Apple Silicon MPS
    if not model_path.is_file():
        raise FileNotFoundError(f"[ERROR] YOLO model checkpoint missing: {model_path}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Object Detection Engine  : YOLOv8n on {device.upper()}")
    model = YOLO(str(model_path))
    results = model.predict(source=img_bgr, conf=CONF_THRESHOLD, device=device, verbose=False, save=False)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in RELEVANT_CLASSES:
            continue
        conf = float(box.conf[0].item())
        x1_raw, y1_raw, x2_raw, y2_raw = [float(c) for c in box.xyxy[0].tolist()]

        # Validate bounding box before and after clipping
        if x2_raw <= x1_raw or y2_raw <= y1_raw:
            continue

        x1 = max(0, min(int(round(x1_raw)), w - 1))
        y1 = max(0, min(int(round(y1_raw)), h - 1))
        x2 = max(0, min(int(round(x2_raw)), w))
        y2 = max(0, min(int(round(y2_raw)), h))

        if (x2 - x1) > 4 and (y2 - y1) > 4:
            detections.append({
                "class_id": cls_id,
                "class_name": RELEVANT_CLASSES[cls_id],
                "conf": conf,
                "bbox": [x1, y1, x2, y2]
            })
    print(f"[INFO] Filtered 2D Detections   : {len(detections)} obstacles ({[d['class_name'] for d in detections]})")

    # 5. Check for genuine Velodyne LiDAR point cloud
    pts_velo = load_velodyne_points(velo_path)
    vis_img = img_bgr.copy()

    evaluation_report = {
        "frame": stem,
        "image_dimensions": f"{w}x{h}",
        "calib_status": "VALID",
        "detections_count": len(detections),
        "detections": []
    }

    if pts_velo is not None:
        # === MODE 1: GENUINE VELODYNE LIDAR FUSION ===
        print(f"\n[MODE] GENUINE CAMERA-LIDAR FUSION ACTIVE ({len(pts_velo):,} Velodyne points loaded)")
        evaluation_report["mode"] = "GENUINE_LIDAR_FUSION"

        n_pts = pts_velo.shape[0]
        pts_homo = np.hstack([pts_velo[:, :3], np.ones((n_pts, 1))])
        pts_cam = (extrinsic_rect @ pts_homo.T).T

        # Filter points behind camera plane (Z > 0.1m)
        front_mask = pts_cam[:, 2] > 0.1
        pts_2d_homo = (proj_matrix @ pts_homo[front_mask].T).T
        u = pts_2d_homo[:, 0] / pts_2d_homo[:, 2]
        v = pts_2d_homo[:, 1] / pts_2d_homo[:, 2]
        depths = np.linalg.norm(pts_cam[front_mask, :3], axis=1)

        # Filter within image boundaries
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u_valid, v_valid, d_valid = u[in_bounds], v[in_bounds], depths[in_bounds]
        print(f"[INFO] Projected in-bounds LiDAR points: {len(u_valid):,}")

        # Render projected points with distance color coding
        for ux, vy, d in zip(u_valid, v_valid, d_valid):
            norm_d = min(1.0, max(0.0, d / 45.0))
            color_b = int(norm_d * 255)
            color_r = int((1.0 - norm_d) * 255)
            cv2.circle(vis_img, (int(round(ux)), int(round(vy))), 1, (color_b, 255, color_r), -1)

        # Associate points with 2D bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            box_mask = (u_valid >= x1) & (u_valid <= x2) & (v_valid >= y1) & (v_valid <= y2)
            pts_in_box = d_valid[box_mask]

            if len(pts_in_box) > 0:
                med_dist = float(np.median(pts_in_box))
                min_dist = float(np.min(pts_in_box))
                pts_count = len(pts_in_box)
                dist_label = f"{med_dist:.1f}m ({pts_count} pts)"
            else:
                med_dist = None
                min_dist = None
                pts_count = 0
                dist_label = "0 pts"

            det_entry = {
                "class": det["class_name"],
                "confidence": round(det["conf"], 3),
                "bbox": det["bbox"],
                "associated_lidar_points": pts_count,
                "metric_distance_m": round(med_dist, 2) if med_dist else None,
                "min_distance_m": round(min_dist, 2) if min_dist else None
            }
            evaluation_report["detections"].append(det_entry)
            print(f"  * {det['class_name']:<10} | Conf: {det['conf']:.2f} | Box: {det['bbox']} | Distance: {dist_label}")

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, f"{det['class_name']} {dist_label}", (x1, max(y1 - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        hud_banner = f"Genuine 3D Camera-LiDAR Fusion | Frame {stem}.png | Ingested: {velo_path.name}"
        cv2.putText(vis_img, hud_banner, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    else:
        # === MODE 2: GROUND-TRUTH AUDIT MODE ===
        print(f"\n[STATUS] Velodyne file missing: {velo_path.name}")
        print("[NOTICE] Operating in GROUND-TRUTH AUDIT MODE.")
        print("[NOTICE] Point-cloud fusion is NOT claimed. Evaluating 3D physical sensor ground-truth from label_2.")
        evaluation_report["mode"] = "GROUND_TRUTH_AUDIT_MODE"
        evaluation_report["point_cloud_fusion_claimed"] = False

        gt_objects = parse_kitti_ground_truth_3d(label_path)
        print(f"[INFO] Physical Sensor 3D Ground-Truth Objects in Frame: {len(gt_objects)}")

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            best_iou = 0.0
            best_gt = None

            for gt in gt_objects:
                iou = calculate_box_iou(det["bbox"], gt["bbox_2d"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None and best_iou >= 0.40:
                true_z = best_gt["depth_z_m"]
                true_dist = best_gt["metric_distance_m"]
                label_text = f"{det['class_name']} (GT: {true_z:.1f}m)"
                det_entry = {
                    "class": det["class_name"],
                    "confidence": round(det["conf"], 3),
                    "bbox": det["bbox"],
                    "matched_gt_type": best_gt["type"],
                    "matched_gt_iou": round(best_iou, 3),
                    "sensor_ground_truth_z_depth_m": true_z,
                    "sensor_ground_truth_3d_dist_m": true_dist
                }
            else:
                label_text = f"{det['class_name']} (GT: N/A)"
                det_entry = {
                    "class": det["class_name"],
                    "confidence": round(det["conf"], 3),
                    "bbox": det["bbox"],
                    "matched_gt_type": None,
                    "matched_gt_iou": 0.0,
                    "sensor_ground_truth_z_depth_m": None,
                    "sensor_ground_truth_3d_dist_m": None
                }

            evaluation_report["detections"].append(det_entry)
            print(
                f"  * {det['class_name']:<10} | Conf: {det['conf']:.2f} | Box: {det['bbox']} | "
                f"Ground-Truth Depth Z: {det_entry['sensor_ground_truth_z_depth_m']}m (IoU: {det_entry['matched_gt_iou']})"
            )

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis_img, label_text, (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        hud_title = f"Ground-Truth Audit Mode (Frame {stem}.png) | Calibration Verified"
        hud_subtitle = "NOTICE: Velodyne .bin absent from dataset. Evaluating physical sensor 3D ground-truth."
        cv2.putText(vis_img, hud_title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(vis_img, hud_subtitle, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

    # 6. Save annotated visual artifact
    cv2.imwrite(str(output_path), vis_img)
    print(f"\n[SUCCESS] Output artifact saved to: {output_path.name}")
    print("=" * 80)

    return evaluation_report


if __name__ == "__main__":
    frame_arg = sys.argv[1] if len(sys.argv) > 1 else "000000"
    run_camera_lidar_fusion(stem=frame_arg)
