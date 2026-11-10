"""
Autonomous Driving Perception Stack - Module 2: Multi-Object Tracking using ByteTrack
Performs multi-object tracking with vehicle & people counting on Apple Silicon MPS.
"""

import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Category mappings for Autonomous Driving Perception
VEHICLE_CLASS_IDS = {
    2: "car",
    5: "bus",
    7: "truck",
    3: "motorcycle",
    1: "bicycle"
}
PEOPLE_CLASS_IDS = {
    0: "person"
}
ALL_PERCEPTION_CLASSES = list(VEHICLE_CLASS_IDS.keys()) + list(PEOPLE_CLASS_IDS.keys())


def generate_synthetic_traffic_stream(
    output_video_path: Path,
    num_frames: int = 60,
    fps: int = 30
) -> Path:
    """
    Generates a synthetic traffic video stream from the local bus.jpg asset.
    Simulates simple camera pan via affine translation of a single static image.
    """
    asset_path = Path("/opt/anaconda3/envs/av-perception/lib/python3.11/site-packages/ultralytics/assets/bus.jpg")
    if not asset_path.exists():
        raise FileNotFoundError(f"Local image asset not found at: {asset_path}")

    base_frame = cv2.imread(str(asset_path))
    if base_frame is None:
        raise RuntimeError(f"cv2.imread failed to load asset at: {asset_path}")

    h, w = base_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for synthetic stream at: {output_video_path}")

    print(f"[INFO] Generating synthetic video stream ({num_frames} frames @ {fps} FPS)...")
    try:
        for i in range(num_frames):
            dx = float(np.sin(i / 10.0) * 8.0 + (i * 0.5))
            dy = float(np.cos(i / 15.0) * 3.0)
            scale = 1.0 + 0.02 * np.sin(i / 12.0)
            center = (w / 2.0, h / 2.0)

            M = cv2.getRotationMatrix2D(center, 0.0, scale)
            M[0, 2] += dx
            M[1, 2] += dy

            frame = cv2.warpAffine(base_frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            writer.write(frame)
    finally:
        writer.release()

    print(f"[SUCCESS] Synthetic traffic video saved to: {output_video_path} ({num_frames} frames)")
    return output_video_path


def run_bytetrack_video(
    video_source: Path = None,
    output_video_path: Path = None,
    conf_thresh: float = 0.1,
    device: str = "mps"
):
    print("=" * 75)
    print("MODULE 2: MULTI-OBJECT TRACKING & VEHICLE / PEOPLE PERCEPTION")
    print("=" * 75)

    # 1. Device check
    if device == "mps" and not torch.backends.mps.is_available():
        print("[WARN] Apple MPS unavailable; falling back to CPU.")
        device = "cpu"
    else:
        print(f"[INFO] Compute Acceleration: {device.upper()} (Apple Silicon M1)")

    # 2. Paths setup
    project_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    if video_source is None or not video_source.exists():
        video_source = project_dir / "sample_traffic_stream.mp4"
        if not video_source.exists():
            generate_synthetic_traffic_stream(video_source, num_frames=60, fps=30)

    if output_video_path is None:
        output_video_path = project_dir / "tracked_output.mp4"

    # 3. Model setup - strict local model verification (no downloads allowed)
    model_path = project_dir / "yolov8n.pt"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Required local YOLO weights not found at: {model_path}. "
            "Execution halted to prevent automatic model download."
        )
    print(f"[INFO] Verified local YOLO detector: {model_path}")
    model = YOLO(str(model_path))

    # 4. Video reader & writer setup
    cap = cv2.VideoCapture(str(video_source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for output video at: {output_video_path}")

    print(f"[INFO] Input stream: {video_source.name} | Res: {width}x{height} | Frames: {total_frames} | Target FPS: {fps}")
    print(f"[INFO] Tracker: ByteTrack (bytetrack.yaml) | Detector Conf: {conf_thresh}")
    print(f"[NOTE] 'Inference FPS' measures model.track() GPU execution latency only, excluding I/O and drawing.")
    print("-" * 75)

    # Data structures for tracking & analytics
    track_history = defaultdict(lambda: deque(maxlen=30))
    track_stats = defaultdict(lambda: {"category": "", "class": "", "conf_sum": 0.0, "frames": 0, "first": 0, "last": 0})

    unique_vehicles = {
        "car": set(),
        "bus": set(),
        "truck": set(),
        "motorcycle": set(),
        "bicycle": set()
    }
    unique_people = set()

    frame_idx = 0
    latencies = []

    color_palette = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
        (52, 69, 255), (100, 115, 255), (0, 24, 236), (132, 56, 255)
    ]

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_start = time.perf_counter()

            # ByteTrack tracking with persistent ID state across sequential frames
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_thresh,
                classes=ALL_PERCEPTION_CLASSES,
                device=device,
                verbose=False
            )

            t_end = time.perf_counter()
            inference_latency_ms = (t_end - t_start) * 1000.0
            latencies.append(inference_latency_ms)

            res = results[0]
            boxes = res.boxes

            annotated_frame = frame.copy()

            frame_vehicle_breakdown = {
                "car": 0,
                "bus": 0,
                "truck": 0,
                "motorcycle": 0,
                "bicycle": 0
            }
            frame_people_count = 0

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                coords = boxes.xyxy.cpu().tolist()

                for box_coords, track_id, cls_id, conf in zip(coords, track_ids, classes, confs):
                    raw_class_name = model.names[cls_id]

                    # Separate Vehicles vs People
                    if cls_id in VEHICLE_CLASS_IDS:
                        vehicle_type = VEHICLE_CLASS_IDS[cls_id]
                        category = "vehicle"
                        frame_vehicle_breakdown[vehicle_type] += 1
                        unique_vehicles[vehicle_type].add(track_id)
                    elif cls_id in PEOPLE_CLASS_IDS:
                        category = "people"
                        frame_people_count += 1
                        unique_people.add(track_id)
                    else:
                        category = "other"

                    # Update cumulative stats
                    stat = track_stats[track_id]
                    stat["category"] = category
                    stat["class"] = raw_class_name
                    stat["conf_sum"] += conf
                    stat["frames"] += 1
                    if stat["first"] == 0:
                        stat["first"] = frame_idx
                    stat["last"] = frame_idx

                    # Geometry & motion trail
                    x1, y1, x2, y2 = map(int, box_coords)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    track_history[track_id].append((cx, cy))

                    color = color_palette[track_id % len(color_palette)]

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Label banner: [ID:X] Class (Confidence%)
                    label = f"ID:{track_id} {raw_class_name} {conf*100:.0f}%"
                    (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - lbl_h - 8)), (x1 + lbl_w + 4, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

                    # Draw past centroid trail
                    pts = track_history[track_id]
                    for p_idx in range(1, len(pts)):
                        thickness = int(np.sqrt(30 / float(p_idx + 1)) * 1.5) + 1
                        cv2.line(annotated_frame, pts[p_idx - 1], pts[p_idx], color, thickness)

                    cv2.circle(annotated_frame, (cx, cy), 4, color, -1)

            total_frame_vehicles = sum(frame_vehicle_breakdown.values())
            inference_fps = (1000.0 / inference_latency_ms) if inference_latency_ms > 0 else 0.0

            # Perception HUD
            cv2.rectangle(annotated_frame, (10, 10), (740, 95), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (10, 10), (740, 95), (70, 70, 70), 1)

            active_total = total_frame_vehicles + frame_people_count
            hud_l1 = f"Frame: {frame_idx:02d}/{total_frames} | Active Tracks: {active_total} | Device: {device.upper()}"
            cv2.putText(annotated_frame, hud_l1, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)

            cars_cnt = frame_vehicle_breakdown["car"]
            bus_cnt = frame_vehicle_breakdown["bus"]
            truck_cnt = frame_vehicle_breakdown["truck"]
            moto_cnt = frame_vehicle_breakdown["motorcycle"]
            bike_cnt = frame_vehicle_breakdown["bicycle"]
            hud_l2 = f"Vehicles: {total_frame_vehicles} (Cars:{cars_cnt} Bus:{bus_cnt} Truck:{truck_cnt} Moto:{moto_cnt} Bike:{bike_cnt}) | People: {frame_people_count}"
            cv2.putText(annotated_frame, hud_l2, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

            hud_l3 = f"Inference Latency: {inference_latency_ms:.1f} ms | Inference FPS: {inference_fps:.1f}"
            cv2.putText(annotated_frame, hud_l3, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

            out_writer.write(annotated_frame)

            if frame_idx % 15 == 0 or frame_idx == total_frames:
                print(
                    f"Frame [{frame_idx:02d}/{total_frames}] | "
                    f"Inference Latency: {inference_latency_ms:5.1f} ms | "
                    f"Inference FPS: {inference_fps:4.1f} | "
                    f"Vehicles: {total_frame_vehicles} | "
                    f"People: {frame_people_count}"
                )
    finally:
        cap.release()
        out_writer.release()

    # 5. Compute global statistics (safe for zero-frame edge case)
    if latencies:
        avg_latency = float(np.mean(latencies))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))
        avg_inference_fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
    else:
        avg_latency = 0.0
        min_latency = 0.0
        max_latency = 0.0
        avg_inference_fps = 0.0

    all_unique_vehicles = set().union(*unique_vehicles.values())

    print()
    print("=" * 75)
    print("AUTONOMOUS DRIVING PERCEPTION & TRACKING SUMMARY")
    print("=" * 75)
    print(f"Total Frames Processed        : {frame_idx}")
    print(f"Average Inference Latency     : {avg_latency:.2f} ms")
    print(f"Latency Range (Min / Max)     : {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f"Average Inference FPS         : {avg_inference_fps:.1f} FPS (model.track execution only)")
    print("-" * 75)
    print("PERCEPTION ENTITY TOTALS (CUMULATIVE UNIQUE TRACKS):")
    print(f"  * Total Unique Vehicles     : {len(all_unique_vehicles)}")
    print(f"      - Cars                  : {len(unique_vehicles['car'])}")
    print(f"      - Buses                 : {len(unique_vehicles['bus'])}")
    print(f"      - Trucks                : {len(unique_vehicles['truck'])}")
    print(f"      - Motorcycles           : {len(unique_vehicles['motorcycle'])}")
    print(f"      - Bicycles              : {len(unique_vehicles['bicycle'])}")
    print(f"  * Total Unique People       : {len(unique_people)}")
    print("-" * 75)
    print(f"{'Track ID':<10} | {'Category':<12} | {'Class':<12} | {'Avg Conf':<10} | {'Lifespan':<18} | {'Persistence'}")
    print("-" * 75)
    for tid in sorted(track_stats.keys()):
        s = track_stats[tid]
        avg_c = s["conf_sum"] / max(1, s["frames"])
        persistence = (s["frames"] / frame_idx * 100.0) if frame_idx > 0 else 0.0
        print(f"ID #{tid:<7} | {s['category']:<12} | {s['class']:<12} | {avg_c*100:5.1f}%    | {s['frames']:2d} frames ({s['first']:02d}-{s['last']:02d}) | {persistence:5.1f}%")
    print("-" * 75)
    print(f"[SUCCESS] Exported tracked video to: {output_video_path}")

    return {
        "frames_processed": frame_idx,
        "avg_latency_ms": avg_latency,
        "avg_inference_fps": avg_inference_fps,
        "unique_vehicles": len(all_unique_vehicles),
        "unique_people": len(unique_people),
        "output_video": str(output_video_path)
    }


if __name__ == "__main__":
    run_bytetrack_video()
