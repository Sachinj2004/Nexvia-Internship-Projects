#!/usr/bin/env python3
"""
onnx_inference.py
Standalone ONNX Runtime inference pipeline for YOLOv8n.
Uses strictly CPUExecutionProvider, letterbox preprocessing, and custom NMS.
Filters for autonomous driving classes: person, bicycle, car, motorcycle, bus, truck.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort

# Filtered autonomous driving classes (COCO IDs)
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


def letterbox(
    img: np.ndarray,
    target_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resizes image to target_size x target_size with aspect-ratio preservation and padding.
    Returns: (padded_image, scale_ratio, (pad_left, pad_top))
    """
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


def run_onnx_inference(
    image_path: Path = None,
    output_path: Path = None
) -> Dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "yolov8n.onnx"

    if image_path is None:
        image_path = project_dir / "sample_road.jpg"
    if output_path is None:
        output_path = project_dir / "onnx_detection_output.jpg"

    print("=" * 70)
    print("STANDALONE ONNX RUNTIME INFERENCE (YOLOv8n)")
    print("=" * 70)

    # 1. Verify model and image existence
    if not model_path.is_file():
        raise FileNotFoundError(f"[ERROR] ONNX model missing: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"[ERROR] Input image missing: {image_path}")

    # 2. Check if output image already exists
    if output_path.exists():
        raise FileExistsError(
            f"[ABORT] Output image {output_path} already exists! Halting per safety condition."
        )

    # 3. Initialize ONNX Runtime InferenceSession with CPUExecutionProvider
    providers = ['CPUExecutionProvider']
    print(f"[INFO] Initializing ONNX Runtime session with providers: {providers}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), session_options, providers=providers)

    active_provider = session.get_providers()[0]
    print(f"[INFO] Active Execution Provider : {active_provider}")

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    input_name = input_meta.name
    output_name = output_meta.name
    print(f"[INFO] Input Tensor Name         : {input_name} (Shape: {input_meta.shape})")
    print(f"[INFO] Output Tensor Name        : {output_name} (Shape: {output_meta.shape})")

    # 4. Load input image
    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        raise ValueError(f"[ERROR] cv2.imread failed on: {image_path}")
    orig_h, orig_w = orig_bgr.shape[:2]
    print(f"[INFO] Input Image               : {image_path.name} ({orig_w} x {orig_h} px)")

    # 5. Letterbox Preprocessing
    padded_img, scale, (pad_w, pad_h) = letterbox(orig_bgr, target_size=INPUT_SIZE)
    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 640, 640)

    # 6. Timed Inference Pass
    print("[INFO] Executing ONNX Runtime inference...")
    t0 = time.perf_counter()
    raw_outputs = session.run([output_name], {input_name: input_tensor})
    t_inference_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[INFO] Inference Latency         : {t_inference_ms:.2f} ms")

    # 7. Output Tensor Parsing & Class Filtering
    # raw_outputs[0] shape: (1, 84, 8400) -> transpose to (8400, 84)
    predictions = np.squeeze(raw_outputs[0]).transpose((1, 0))  # (8400, 84)

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]  # (8400, 80)

    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # Filter by confidence threshold
    conf_mask = max_scores >= CONF_THRESHOLD
    boxes_filtered = boxes_xywh[conf_mask]
    scores_filtered = max_scores[conf_mask]
    classes_filtered = class_ids[conf_mask]

    # Convert xywh in letterbox coords to xywh in original image coords for cv2.dnn.NMSBoxes
    candidate_boxes: List[List[int]] = []
    candidate_scores: List[float] = []
    candidate_classes: List[int] = []

    for box, score, cls_id in zip(boxes_filtered, scores_filtered, classes_filtered):
        if cls_id not in RELEVANT_CLASSES:
            continue

        cx, cy, bw, bh = box
        # Scale back from letterbox to original frame coordinates
        x1 = (cx - bw / 2.0 - pad_w) / scale
        y1 = (cy - bh / 2.0 - pad_h) / scale
        x2 = (cx + bw / 2.0 - pad_w) / scale
        y2 = (cy + bh / 2.0 - pad_h) / scale

        # Clip to image boundaries
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

    # 8. Non-Maximum Suppression (NMS)
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

    final_detections = []
    vis_img = orig_bgr.copy()

    for idx in indices:
        bx, by, bw, bh = candidate_boxes[idx]
        conf = candidate_scores[idx]
        cid = candidate_classes[idx]
        cname = RELEVANT_CLASSES[cid]

        final_detections.append({
            "class_id": cid,
            "class_name": cname,
            "confidence": round(conf, 3),
            "bbox_xywh": [bx, by, bw, bh]
        })

        # Draw visualization box
        color = (0, 255, 0)
        cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), color, 2)
        label = f"{cname} {conf:.2f}"
        cv2.putText(vis_img, label, (bx, max(by - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 9. Save annotated image with existence verification
    cv2.imwrite(str(output_path), vis_img)
    print(f"[SUCCESS] Annotated output saved to: {output_path}")

    # 10. Summary report
    print("-" * 70)
    print(f"Detected Object Count   : {len(final_detections)}")
    detected_class_names = [d["class_name"] for d in final_detections]
    print(f"Detected Class Names    : {detected_class_names if detected_class_names else 'None'}")
    print(f"Inference Latency       : {t_inference_ms:.2f} ms")
    print(f"Output Image Dimensions : {vis_img.shape[1]} x {vis_img.shape[0]} px (3 channels)")
    print(f"Execution Provider      : {active_provider}")
    print("=" * 70)

    return {
        "count": len(final_detections),
        "classes": detected_class_names,
        "latency_ms": t_inference_ms,
        "dimensions": f"{vis_img.shape[1]}x{vis_img.shape[0]}",
        "provider": active_provider,
        "detections": final_detections
    }


if __name__ == "__main__":
    img_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_onnx_inference(image_path=img_arg)
