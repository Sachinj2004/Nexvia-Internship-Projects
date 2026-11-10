"""
Autonomous Driving Perception Stack - Module 4: Monocular Depth Estimation
Performs relative depth estimation using MiDaS_small with Apple Silicon MPS acceleration.
Generates a side-by-side visualization comparing the camera view with the relative depth map.
"""

import time
from pathlib import Path
import cv2
import numpy as np
import torch


def run_depth_estimation(input_image_path: Path, output_image_path: Path):
    print("=" * 75)
    print("MODULE 4: MONOCULAR DEPTH ESTIMATION (MiDaS_small)")
    print("=" * 75)

    # 1. Strict input file existence check
    if not input_image_path.is_file():
        raise FileNotFoundError(f"[ERROR] Required input image not found: {input_image_path}")

    # 2. Strict Apple MPS device check
    if not torch.backends.mps.is_available():
        raise RuntimeError("[ERROR] Apple MPS hardware acceleration is not available! Halting as requested.")

    device = torch.device("mps")
    print(f"[INFO] Hardware Acceleration Device: {device.type.upper()} (Apple Silicon M1 GPU)")

    # 3. Load input image
    img_bgr = cv2.imread(str(input_image_path))
    if img_bgr is None:
        raise RuntimeError(f"[ERROR] cv2.imread failed to load: {input_image_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    print(f"[INFO] Input Image: {input_image_path.name} ({orig_w}x{orig_h} px)")

    # 4. Load pretrained MiDaS_small model from PyTorch Hub
    print("[INFO] Loading pretrained MiDaS_small model (~80 MB)...")
    load_start = time.perf_counter()
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.to(device)
    midas.eval()

    # Load input preprocessing transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    load_time = time.perf_counter() - load_start
    print(f"[SUCCESS] Model loaded successfully in {load_time:.2f} seconds.")

    # 5. Preprocess frame
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    # 6. Cold-start / warm-up pass (compiles Metal MPS kernels)
    print("[INFO] Executing initial warm-up pass for Metal shader compilation...")
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = midas(input_batch)
        cold_latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[INFO] Cold-start latency (with JIT compile): {cold_latency_ms:.1f} ms")

    # 7. Timed steady-state inference passes
    num_runs = 5
    latencies = []
    with torch.no_grad():
        for i in range(num_runs):
            t_start = time.perf_counter()
            prediction = midas(input_batch)

            # Resize depth map to original camera resolution using bicubic interpolation
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)

    avg_latency_ms = float(np.mean(latencies))
    steady_fps = (1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

    print(f"[INFO] Steady-State Inference Latency: {avg_latency_ms:.2f} ms")
    print(f"[INFO] Steady-State Inference FPS    : {steady_fps:.1f} FPS")

    # 8. Extract depth map & normalize
    depth_map = prediction.cpu().numpy()
    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())

    # Normalized relative depth (0 = far, 255 = near)
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_8u = (depth_normalized * 255.0).astype(np.uint8)

    # 9. Apply automotive colormap (Inferno: warm yellow/orange = near, dark purple = far)
    depth_colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)

    # 10. Construct Side-by-Side Visualization Canvas
    # Header banner (80 px height)
    header_h = 80
    total_w = orig_w * 2
    total_h = orig_h + header_h

    composite = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    # Header background
    cv2.rectangle(composite, (0, 0), (total_w, header_h), (20, 20, 20), -1)
    cv2.line(composite, (0, header_h), (total_w, header_h), (70, 70, 70), 2)

    # Header text
    cv2.putText(
        composite,
        "MODULE 4: MONOCULAR DEPTH ESTIMATION (MiDaS_small)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2,
    )
    hud_sub = f"Device: MPS (Apple M1) | Latency: {avg_latency_ms:.1f} ms | FPS: {steady_fps:.1f} | Note: Relative Depth Estimate (Non-Metric)"
    cv2.putText(
        composite,
        hud_sub,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
    )

    # Insert panels
    composite[header_h:header_h + orig_h, 0:orig_w] = img_bgr
    composite[header_h:header_h + orig_h, orig_w:total_w] = depth_colored

    # Dividing line between left and right views
    cv2.line(composite, (orig_w, header_h), (orig_w, total_h), (100, 100, 100), 2)

    # Labels for individual panels
    # Left panel label
    cv2.rectangle(composite, (10, header_h + 10), (280, header_h + 42), (0, 0, 0), -1)
    cv2.putText(
        composite,
        f"Camera: {input_image_path.name}",
        (18, header_h + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    # Right panel label
    cv2.rectangle(composite, (orig_w + 10, header_h + 10), (orig_w + 460, header_h + 42), (0, 0, 0), -1)
    cv2.putText(
        composite,
        "Relative Depth Map (Warm=Near, Dark=Far)",
        (orig_w + 18, header_h + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

    # 11. Save output image & validate return value
    save_success = cv2.imwrite(str(output_image_path), composite)
    if not save_success:
        raise IOError(f"[ERROR] cv2.imwrite failed to save output image to: {output_image_path}")

    print("-" * 75)
    print(f"[SUCCESS] Depth visualization saved to: {output_image_path}")
    print(f"[SUCCESS] Composite Dimensions: {total_w}x{total_h} px")
    print("=" * 75)

    return {
        "model": "MiDaS_small",
        "device": device.type.upper(),
        "cold_latency_ms": cold_latency_ms,
        "avg_latency_ms": avg_latency_ms,
        "fps": steady_fps,
        "output_path": str(output_image_path),
        "dimensions": f"{total_w}x{total_h}",
    }


if __name__ == "__main__":
    proj_dir = Path(__file__).resolve().parent
    in_img = proj_dir / "sample_road.jpg"
    out_img = proj_dir / "depth_output.jpg"
    run_depth_estimation(in_img, out_img)
