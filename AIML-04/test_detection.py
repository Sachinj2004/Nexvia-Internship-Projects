"""
Autonomous Driving Perception Stack - Module 1: 2D Object Detection Test
Performs real-time object detection using YOLOv8n with Apple Silicon MPS acceleration.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def create_synthetic_road_scene(output_path: Path) -> Path:
    """Creates a synthetic camera frame with simulated vehicles and road markings for offline testing."""
    height, width = 640, 640
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Road surface (dark grey)
    cv2.rectangle(img, (0, 300), (width, height), (50, 50, 50), -1)

    # Sky (light blue/grey)
    cv2.rectangle(img, (0, 0), (width, 300), (200, 180, 140), -1)

    # Lane markings (yellow center line, white side lines)
    cv2.line(img, (320, 320), (320, height), (0, 255, 255), 4)  # center yellow
    cv2.line(img, (80, 320), (0, height), (255, 255, 255), 4)    # left white
    cv2.line(img, (560, 320), (width, height), (255, 255, 255), 4)  # right white

    # Draw simulated vehicle (car shape)
    # Body
    cv2.rectangle(img, (240, 380), (400, 480), (180, 40, 40), -1)
    # Roof/cabin
    cv2.rectangle(img, (270, 340), (370, 380), (140, 30, 30), -1)
    # Wheels
    cv2.circle(img, (270, 480), 20, (20, 20, 20), -1)
    cv2.circle(img, (370, 480), 20, (20, 20, 20), -1)
    # Rear brake lights
    cv2.rectangle(img, (245, 395), (275, 415), (0, 0, 230), -1)
    cv2.rectangle(img, (365, 395), (395, 415), (0, 0, 230), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Generated synthetic road test scene: {output_path}")
    return output_path


def run_detection(
    model_path: str = "yolov8n.pt",
    image_path: str = None,
    conf_threshold: float = 0.25,
    device: str = "mps"
):
    print("=" * 60)
    print("AUTONOMOUS DRIVING PERCEPTION STACK - YOLO TEST")
    print("=" * 60)

    # 1. Device check
    if device == "mps" and not torch.backends.mps.is_available():
        print("[WARN] Apple MPS is not available; falling back to CPU.")
        device = "cpu"
    else:
        print(f"[INFO] Using hardware acceleration device: {device.upper()} (Apple Silicon M1)")

    # 2. Image resolution
    script_dir = Path(__file__).resolve().parent
    if image_path is None or not Path(image_path).exists():
        default_img = script_dir / "sample_road.jpg"
        if not default_img.exists():
            create_synthetic_road_scene(default_img)
        image_path = str(default_img)

    print(f"[INFO] Input image path: {image_path}")

    # 3. Model loading
    print(f"[INFO] Loading YOLO model: {model_path} ...")
    model = YOLO(model_path)

    # 4. Inference timing
    print("[INFO] Running perception inference...")
    start_time = time.perf_counter()
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        device=device,
        save=False,
        verbose=False
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    print(f"[INFO] Inference latency: {latency_ms:.2f} ms (~{1000 / latency_ms:.1f} FPS)")

    # 5. Extract and print detections
    result = results[0]
    boxes = result.boxes
    num_detections = len(boxes)

    print("-" * 60)
    print(f"Detections Found: {num_detections}")
    print("-" * 60)

    detections_summary = []
    for idx, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        class_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]

        detections_summary.append({
            "id": idx + 1,
            "class": class_name,
            "confidence": conf,
            "bbox_xyxy": coords
        })

        print(f"[{idx+1}] Class: {class_name:<15} | Conf: {conf*100:5.1f}% | Box (xyxy): {coords}")

    # 6. Save annotated visualization
    output_path = script_dir / "detection_output.jpg"
    annotated_frame = result.plot()
    cv2.imwrite(str(output_path), annotated_frame)
    print("-" * 60)
    print(f"[SUCCESS] Annotated detection image saved to: {output_path}")

    return {
        "num_detections": num_detections,
        "latency_ms": latency_ms,
        "detections": detections_summary,
        "output_image": str(output_path)
    }


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_detection(image_path=img_arg)
