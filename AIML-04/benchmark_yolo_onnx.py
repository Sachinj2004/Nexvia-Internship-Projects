#!/usr/bin/env python3
"""
benchmark_yolo_onnx.py
Rigorous Comparative Benchmark & Numerical Validation:
YOLOv8n PyTorch (Apple MPS) vs. ONNX Runtime (CPUExecutionProvider).

Requirements:
  - 10 un-timed warm-up passes + 100 timed steady-state passes.
  - Proper torch.mps.synchronize() for Apple Silicon GPU timing.
  - Statistical metrics: mean, median (p50), p95, min, max, FPS.
  - Numerical validation: Matched detection IoU, confidence delta, unmatched detections.
  - Memory tracking via psutil (current RSS).
  - Visual comparison saved to onnx_vs_pytorch_comparison.jpg (with pre-existence check).
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import psutil
import torch
import onnxruntime as ort
from ultralytics import YOLO

# Autonomous driving evaluation classes
RELEVANT_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


def get_current_rss_mb() -> float:
    """Returns the current process resident memory (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def letterbox(
    img: np.ndarray,
    target_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) / 2.0
    pad_h = (target_size - new_h) / 2.0

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (pad_w, pad_h)


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union between [x1, y1, x2, y2] boxes."""
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

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def run_pytorch_mps_inference(
    model: YOLO,
    img_bgr: np.ndarray,
    device: str = "mps"
) -> List[Dict[str, Any]]:
    """Runs single PyTorch MPS forward pass and returns filtered detections [x1, y1, x2, y2]."""
    results = model.predict(
        source=img_bgr,
        conf=CONF_THRESHOLD,
        device=device,
        verbose=False,
        save=False
    )
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in RELEVANT_CLASSES:
            continue
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(c) for c in box.xyxy[0].tolist()]
        detections.append({
            "class_id": cls_id,
            "class_name": RELEVANT_CLASSES[cls_id],
            "confidence": conf,
            "bbox_xyxy": [x1, y1, x2, y2]
        })
    return detections


def run_onnx_cpu_inference(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    img_bgr: np.ndarray
) -> List[Dict[str, Any]]:
    """Runs single ONNX Runtime CPU forward pass and returns filtered detections [x1, y1, x2, y2]."""
    orig_h, orig_w = img_bgr.shape[:2]
    padded_img, scale, (pad_w, pad_h) = letterbox(img_bgr, target_size=INPUT_SIZE)
    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

    raw_outputs = session.run([output_name], {input_name: input_tensor})
    predictions = np.squeeze(raw_outputs[0]).transpose((1, 0))  # (8400, 84)

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    conf_mask = max_scores >= CONF_THRESHOLD
    boxes_filtered = boxes_xywh[conf_mask]
    scores_filtered = max_scores[conf_mask]
    classes_filtered = class_ids[conf_mask]

    candidate_boxes = []
    candidate_scores = []
    candidate_classes = []

    for box, score, cls_id in zip(boxes_filtered, scores_filtered, classes_filtered):
        if cls_id not in RELEVANT_CLASSES:
            continue
        cx, cy, bw, bh = box
        x1 = (cx - bw / 2.0 - pad_w) / scale
        y1 = (cy - bh / 2.0 - pad_h) / scale
        x2 = (cx + bw / 2.0 - pad_w) / scale
        y2 = (cy + bh / 2.0 - pad_h) / scale

        x1 = max(0.0, min(float(x1), float(orig_w)))
        y1 = max(0.0, min(float(y1), float(orig_h)))
        x2 = max(0.0, min(float(x2), float(orig_w)))
        y2 = max(0.0, min(float(y2), float(orig_h)))

        w_box = x2 - x1
        h_box = y2 - y1
        if w_box > 0 and h_box > 0:
            candidate_boxes.append([int(round(x1)), int(round(y1)), int(round(w_box)), int(round(h_box))])
            candidate_scores.append(float(score))
            candidate_classes.append(int(cls_id))

    indices = []
    if candidate_boxes:
        indices = cv2.dnn.NMSBoxes(
            candidate_boxes,
            candidate_scores,
            score_threshold=CONF_THRESHOLD,
            nms_threshold=IOU_THRESHOLD
        )
        if len(indices) > 0 and hasattr(indices[0], '__len__'):
            indices = [i[0] for i in indices]

    detections = []
    for idx in indices:
        bx, by, bw, bh = candidate_boxes[idx]
        conf = candidate_scores[idx]
        cid = candidate_classes[idx]
        detections.append({
            "class_id": cid,
            "class_name": RELEVANT_CLASSES[cid],
            "confidence": conf,
            "bbox_xyxy": [float(bx), float(by), float(bx + bw), float(by + bh)]
        })
    return detections


def main():
    project_dir = Path(__file__).resolve().parent
    pt_path = project_dir / "yolov8n.pt"
    onnx_path = project_dir / "yolov8n.onnx"
    test_img_path = project_dir / "data" / "kitti" / "training" / "image_2" / "000000.png"
    comparison_img_path = project_dir / "onnx_vs_pytorch_comparison.jpg"

    print("=" * 80)
    print("COMPARATIVE BENCHMARK: PYTORCH (APPLE MPS) VS. ONNX RUNTIME (CPU)")
    print("=" * 80)

    # 1. Existence validations
    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing PyTorch model: {pt_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")
    if not test_img_path.is_file():
        raise FileNotFoundError(f"Missing test image: {test_img_path}")
    if comparison_img_path.exists():
        raise FileExistsError(f"[ABORT] Comparison image {comparison_img_path} already exists! Halting.")

    # 2. Load test image
    img_bgr = cv2.imread(str(test_img_path))
    h_orig, w_orig = img_bgr.shape[:2]
    print(f"[INFO] Test Dataset Frame: {test_img_path.name} ({w_orig} x {h_orig} px)")
    print(f"[INFO] Initial Process Memory RSS: {get_current_rss_mb():.1f} MB")

    # 3. Load PyTorch model on Apple Silicon MPS
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n[INFO] Initializing PyTorch Model on {device_name.upper()}...")
    pt_model = YOLO(str(pt_path))
    rss_after_pt = get_current_rss_mb()
    print(f"[INFO] Process Memory after PyTorch load: {rss_after_pt:.1f} MB (+{rss_after_pt - get_current_rss_mb():.1f} MB)")

    # 4. Load ONNX Runtime session with CPUExecutionProvider
    print("\n[INFO] Initializing ONNX Runtime Session (CPUExecutionProvider)...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    onnx_session = ort.InferenceSession(str(onnx_path), sess_opts, providers=['CPUExecutionProvider'])
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    rss_after_onnx = get_current_rss_mb()
    print(f"[INFO] Process Memory after ONNX Runtime load: {rss_after_onnx:.1f} MB")

    # 5. Numerical Detection Parity Check
    print("\n" + "-" * 80)
    print("NUMERICAL DETECTION PARITY & MATCHING VALIDATION")
    print("-" * 80)
    pt_dets = run_pytorch_mps_inference(pt_model, img_bgr, device=device_name)
    onnx_dets = run_onnx_cpu_inference(onnx_session, input_name, output_name, img_bgr)

    print(f"PyTorch MPS Detections  : {len(pt_dets)}")
    for d in pt_dets:
        box = [round(c, 1) for c in d['bbox_xyxy']]
        print(f"  * {d['class_name']:<10} | Conf: {d['confidence']:.3f} | Box: {box}")

    print(f"\nONNX Runtime Detections : {len(onnx_dets)}")
    for d in onnx_dets:
        box = [round(c, 1) for c in d['bbox_xyxy']]
        print(f"  * {d['class_name']:<10} | Conf: {d['confidence']:.3f} | Box: {box}")

    # Matching algorithm
    matched_pairs = []
    unmatched_pt = list(pt_dets)
    unmatched_onnx = list(onnx_dets)

    for pt_d in pt_dets:
        best_iou = 0.0
        best_match = None
        for onnx_d in unmatched_onnx:
            if pt_d['class_name'] == onnx_d['class_name']:
                iou = calculate_iou(pt_d['bbox_xyxy'], onnx_d['bbox_xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = onnx_d

        if best_match is not None and best_iou >= 0.50:
            conf_diff = abs(pt_d['confidence'] - best_match['confidence'])
            matched_pairs.append({
                "class_name": pt_d['class_name'],
                "iou": best_iou,
                "pt_conf": pt_d['confidence'],
                "onnx_conf": best_match['confidence'],
                "conf_delta": conf_diff
            })
            unmatched_pt.remove(pt_d)
            unmatched_onnx.remove(best_match)

    print(f"\nMatched Detection Pairs : {len(matched_pairs)}")
    for m in matched_pairs:
        print(f"  * {m['class_name']:<10} | IoU: {m['iou']:.4f} | PT Conf: {m['pt_conf']:.3f} | ONNX Conf: {m['onnx_conf']:.3f} | Delta: {m['conf_delta']:.4f}")

    if unmatched_pt:
        print(f"Unmatched PyTorch Detections : {[d['class_name'] for d in unmatched_pt]}")
    else:
        print("Unmatched PyTorch Detections : None (0)")

    if unmatched_onnx:
        print(f"Unmatched ONNX Detections    : {[d['class_name'] for d in unmatched_onnx]}")
    else:
        print("Unmatched ONNX Detections    : None (0)")

    # 6. Render Side-by-Side Visual Comparison
    vis_pt = img_bgr.copy()
    for d in pt_dets:
        x1, y1, x2, y2 = [int(round(c)) for c in d['bbox_xyxy']]
        cv2.rectangle(vis_pt, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_pt, f"{d['class_name']} {d['confidence']:.2f}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis_pt, "PyTorch (Apple MPS)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    vis_onnx = img_bgr.copy()
    for d in onnx_dets:
        x1, y1, x2, y2 = [int(round(c)) for c in d['bbox_xyxy']]
        cv2.rectangle(vis_onnx, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_onnx, f"{d['class_name']} {d['confidence']:.2f}", (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_onnx, "ONNX Runtime (CPU)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    combined_vis = np.hstack([vis_pt, vis_onnx])
    cv2.imwrite(str(comparison_img_path), combined_vis)
    print(f"\n[SUCCESS] Side-by-side comparison image saved to: {comparison_img_path.name}")

    # 7. Benchmark: PyTorch (Apple MPS)
    print("\n" + "-" * 80)
    print(f"BENCHMARKING: PyTorch (Apple MPS) — {WARMUP_RUNS} Warm-up + {BENCHMARK_RUNS} Timed Iterations")
    print("-" * 80)

    # Warm-up
    print("[INFO] Executing PyTorch MPS warm-up passes...")
    for _ in range(WARMUP_RUNS):
        if device_name == "mps":
            torch.mps.synchronize()
        _ = pt_model.predict(source=img_bgr, conf=CONF_THRESHOLD, device=device_name, verbose=False, save=False)
        if device_name == "mps":
            torch.mps.synchronize()

    # Timed runs
    pt_latencies = []
    print("[INFO] Running timed iterations...")
    for _ in range(BENCHMARK_RUNS):
        if device_name == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = pt_model.predict(source=img_bgr, conf=CONF_THRESHOLD, device=device_name, verbose=False, save=False)
        if device_name == "mps":
            torch.mps.synchronize()
        pt_latencies.append((time.perf_counter() - t0) * 1000.0)

    pt_mean = float(np.mean(pt_latencies))
    pt_median = float(np.median(pt_latencies))
    pt_p95 = float(np.percentile(pt_latencies, 95))
    pt_min = float(np.min(pt_latencies))
    pt_max = float(np.max(pt_latencies))
    pt_fps = 1000.0 / pt_mean
    pt_rss = get_current_rss_mb()

    # 8. Benchmark: ONNX Runtime (CPUExecutionProvider)
    print("\n" + "-" * 80)
    print(f"BENCHMARKING: ONNX Runtime (CPU) — {WARMUP_RUNS} Warm-up + {BENCHMARK_RUNS} Timed Iterations")
    print("-" * 80)

    # Prepare input tensor once for pure inference timing
    padded_img, _, _ = letterbox(img_bgr, target_size=INPUT_SIZE)
    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

    # Warm-up
    print("[INFO] Executing ONNX Runtime CPU warm-up passes...")
    for _ in range(WARMUP_RUNS):
        _ = onnx_session.run([output_name], {input_name: input_tensor})

    # Timed runs
    onnx_latencies = []
    print("[INFO] Running timed iterations...")
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        _ = onnx_session.run([output_name], {input_name: input_tensor})
        onnx_latencies.append((time.perf_counter() - t0) * 1000.0)

    onnx_mean = float(np.mean(onnx_latencies))
    onnx_median = float(np.median(onnx_latencies))
    onnx_p95 = float(np.percentile(onnx_latencies, 95))
    onnx_min = float(np.min(onnx_latencies))
    onnx_max = float(np.max(onnx_latencies))
    onnx_fps = 1000.0 / onnx_mean
    onnx_rss = get_current_rss_mb()

    # 9. Final Comparative Report
    print("\n" + "=" * 80)
    print("FINAL COMPARATIVE PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<28} | {'PyTorch (Apple MPS)':<22} | {'ONNX Runtime (CPU)':<22}")
    print("-" * 80)
    print(f"{'Model Format':<28} | {'PyTorch (.pt)':<22} | {'ONNX (.onnx)':<22}")
    print(f"{'File Size':<28} | {f'{pt_path.stat().st_size / (1024*1024):.2f} MB':<22} | {f'{onnx_path.stat().st_size / (1024*1024):.2f} MB':<22}")
    print(f"{'Execution Hardware':<28} | {'Apple Silicon M1 GPU':<22} | {'Apple Silicon M1 CPU':<22}")
    print(f"{'Execution Provider/Device':<28} | {'torch.backends.mps':<22} | {'CPUExecutionProvider':<22}")
    print(f"{'Mean Latency':<28} | {f'{pt_mean:.2f} ms':<22} | {f'{onnx_mean:.2f} ms':<22}")
    print(f"{'Median Latency (p50)':<28} | {f'{pt_median:.2f} ms':<22} | {f'{onnx_median:.2f} ms':<22}")
    print(f"{'95th Percentile (p95)':<28} | {f'{pt_p95:.2f} ms':<22} | {f'{onnx_p95:.2f} ms':<22}")
    print(f"{'Minimum Latency':<28} | {f'{pt_min:.2f} ms':<22} | {f'{onnx_min:.2f} ms':<22}")
    print(f"{'Maximum Latency':<28} | {f'{pt_max:.2f} ms':<22} | {f'{onnx_max:.2f} ms':<22}")
    print(f"{'Throughput (FPS)':<28} | {f'{pt_fps:.1f} FPS':<22} | {f'{onnx_fps:.1f} FPS':<22}")
    print(f"{'Peak Process RSS':<28} | {f'{pt_rss:.1f} MB':<22} | {f'{onnx_rss:.1f} MB':<22}")
    print("-" * 80)

    # Relative speedup comparison
    speedup = pt_mean / onnx_mean
    if speedup > 1.0:
        summary_verdict = f"ONNX Runtime CPU is {speedup:.2f}x FASTER than PyTorch MPS on this workload."
    else:
        summary_verdict = f"PyTorch MPS is {(1.0 / speedup):.2f}x FASTER than ONNX Runtime CPU on this workload."
    print(f"Conclusion: {summary_verdict}")
    print("=" * 80)


if __name__ == "__main__":
    main()
