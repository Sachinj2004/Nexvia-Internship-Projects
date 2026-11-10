#!/usr/bin/env python3
"""
PerceptionInferenceService
Dedicated deep learning inference server running under av-perception.

Capabilities:
  - YOLOv8n (yolov8n.pt): 2D object detection on Apple Silicon MPS GPU.
    * Strictly filtered by RELEVANT_CLASSES (person, bicycle, car, motorcycle, bus, truck).
  - MiDaS_small: Monocular relative inverse disparity estimation on Apple Silicon MPS GPU.
    * Verified local PyTorch cache loading (~/.cache/torch/hub); zero automatic downloads.
  - Accurate MPS Hardware Timing: Synchronizes Apple Silicon Metal shaders with
    torch.mps.synchronize() before and after inference for true GPU duration measurement.
  - Startup MPS Shader Warm-up: Executes dummy forward pass on startup before opening socket.
  - Robust Depth Processing: Rejects NaN/Inf outputs, handles constant/degenerate depth safely.
  - Obstacle-Depth Fusion: Extracts median relative depth from center 50% patch.
  - Proximity Categorization: NEAR (>= 160), MEDIUM (90..159), FAR (< 90).
    * CRITICAL NOTICE: Scores are frame-relative heuristic indicators, NOT metric distances in meters.
  - Hardened UNIX Domain Socket Lifecycle:
    * Safe socket cleanup: tests connection with probe; only unlinks on ECONNREFUSED.
    * Tracks socket ownership; unlinks socket on exit ONLY if owned by this process.
    * Controlled signal handling (SIGINT, SIGTERM) requests clean loop exit.
    * Initializes conn = None per accept cycle; handles accept errors gracefully.
    * Bounded payload validation: 0 < H <= 2160, 0 < W <= 3840, C == 3, length == H * W * 3.
    * Maximum 50 detected obstacles per response; all numeric values validated finite.
"""

import os
import sys
import time
import math
import socket
import struct
import signal
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

SOCKET_PATH = '/tmp/av_perception_ipc.sock'

# Protocol bounds
MAX_WIDTH = 3840
MAX_HEIGHT = 2160
MAX_PAYLOAD_BYTES = 25 * 1024 * 1024  # 25 MB max frame buffer
MAX_RESP_BYTES = 5 * 1024 * 1024     # 5 MB max response
MAX_OBSTACLES_PER_FRAME = 50
IPC_IO_TIMEOUT_SEC = 10.0            # Request/response transfer timeout

# Proximity classification thresholds on normalized relative disparity [0, 255]
DEPTH_THRESHOLD_NEAR = 160
DEPTH_THRESHOLD_MID = 90

# Autonomous driving relevant classes (COCO IDs)
RELEVANT_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

server_running = True
socket_owned = False


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Guarantees reception of exactly num_bytes from stream socket or raises ConnectionError."""
    buf = bytearray(num_bytes)
    view = memoryview(buf)
    received = 0
    while received < num_bytes:
        chunk_size = sock.recv_into(view[received:], num_bytes - received)
        if chunk_size == 0:
            raise ConnectionError(f'Socket closed unexpectedly after {received}/{num_bytes} bytes.')
        received += chunk_size
    return bytes(buf)


def safe_socket_cleanup():
    """
    Safely probes existing socket file before binding.
    CRITICAL HARDENING:
      - If connect() succeeds: Active server detected -> DO NOT UNLINK. Abort immediately.
      - If connect() times out: Server may be busy -> DO NOT UNLINK. Abort.
      - If PermissionError occurs: Permission denied -> DO NOT UNLINK. Abort.
      - ONLY if ConnectionRefusedError occurs: Safe to unlink dead/stale socket file.
    """
    if not os.path.exists(SOCKET_PATH):
        return

    test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    test_sock.settimeout(1.0)
    try:
        test_sock.connect(SOCKET_PATH)
        test_sock.close()
        raise RuntimeError(
            f'An active inference server is already listening on {SOCKET_PATH}. '
            f'Refusing to overwrite active socket. Stop the existing instance first.'
        )
    except ConnectionRefusedError:
        print(f'[av-perception] Verified stale socket (ConnectionRefused). Safely unlinking {SOCKET_PATH}.')
        try:
            os.unlink(SOCKET_PATH)
        except OSError as e:
            raise RuntimeError(f'Failed to unlink stale socket: {e}')
    except socket.timeout:
        raise RuntimeError(
            f'Socket connect to {SOCKET_PATH} timed out. '
            f'An active server may be busy. Refusing to unlink.'
        )
    except PermissionError as e:
        raise RuntimeError(f'Permission denied accessing socket {SOCKET_PATH}: {e}. Refusing to unlink.')
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f'Unexpected error probing socket {SOCKET_PATH} ({e}). Refusing to unlink.')
    finally:
        test_sock.close()


def resolve_yolo_weights() -> Path:
    """Resolves yolov8n.pt across multiple search paths to ensure robust loading."""
    candidates = [
        Path(__file__).resolve().parent / 'yolov8n.pt',
        Path(__file__).resolve().parent.parent / 'yolov8n.pt',
        Path.cwd() / 'yolov8n.pt',
        Path.home() / 'my intern' / 'Autonomous-Driving-Perception-Stack' / 'yolov8n.pt'
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f'[ERROR] yolov8n.pt missing in project hierarchy. Searched: {[str(x) for x in candidates]}. '
        f'Model downloads are blocked by strict safety rules.'
    )


def verify_and_load_models(device: torch.device):
    """
    Verifies that YOLOv8n and MiDaS_small weights exist locally before loading.
    Aborts immediately if models are missing, preventing unapproved downloads.
    """
    # 1. YOLOv8n check
    yolo_weights = resolve_yolo_weights()
    print(f'[av-perception] Loading YOLOv8n from {yolo_weights} on {device}...')
    yolo_model = YOLO(str(yolo_weights))

    # 2. MiDaS local cache check
    hub_dir = Path.home() / '.cache' / 'torch' / 'hub'
    midas_repo = hub_dir / 'intel-isl_MiDaS_master'
    midas_ckpt = hub_dir / 'checkpoints' / 'midas_v21_small_256.pt'

    if not midas_repo.is_dir() or not midas_ckpt.is_file():
        raise FileNotFoundError(
            f'[ERROR] MiDaS repo ({midas_repo}) or checkpoint ({midas_ckpt}) missing in local PyTorch cache! '
            f'Automatic internet downloads are prohibited. Verify cached assets before running.'
        )

    print(f'[av-perception] Loading verified cached MiDaS_small model from {midas_ckpt}...')
    midas_model = torch.hub.load(
        str(midas_repo),
        'MiDaS_small',
        source='local',
        pretrained=False,
        trust_repo=True
    )
    state_dict = torch.load(str(midas_ckpt), map_location=device)
    midas_model.load_state_dict(state_dict)
    midas_model.to(device)
    midas_model.eval()

    midas_transforms = torch.hub.load(str(midas_repo), 'transforms', source='local', trust_repo=True)
    transform = midas_transforms.small_transform

    # 3. Metal MPS Shader Warm-up: pre-compile GPU kernels using dummy image BEFORE binding socket
    print('[av-perception] Pre-compiling Apple Silicon Metal MPS shaders with dummy warm-up frame...')
    dummy_frame = np.zeros((375, 1242, 3), dtype=np.uint8)
    _ = yolo_model.predict(source=dummy_frame, conf=0.25, device=str(device), verbose=False, save=False)
    dummy_input = transform(cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)).to(device)
    with torch.no_grad():
        _ = midas_model(dummy_input)
    if device.type == 'mps':
        torch.mps.synchronize()
    print('[av-perception] MPS shader warm-up complete. Models ready for steady-state inference.')

    return yolo_model, midas_model, transform


def extract_box_relative_depth(bbox: List[int], depth_8u: np.ndarray) -> float:
    """Extracts median relative depth from center 50% patch to eliminate background contamination."""
    x1, y1, x2, y2 = bbox
    h, w = depth_8u.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 4 or bh <= 4:
        patch = depth_8u[y1:y2, x1:x2]
        return float(np.median(patch)) if patch.size > 0 else 0.0

    cx1 = int(x1 + 0.25 * bw)
    cx2 = int(x2 - 0.25 * bw)
    cy1 = int(y1 + 0.25 * bh)
    cy2 = int(y2 - 0.25 * bh)

    patch = depth_8u[cy1:cy2, cx1:cx2]
    return float(np.median(patch)) if patch.size > 0 else float(np.median(depth_8u[y1:y2, x1:x2]))


def categorize_proximity(rel_depth: float) -> str:
    """Heuristic proximity classification based on frame-relative inverse disparity score."""
    if rel_depth >= DEPTH_THRESHOLD_NEAR:
        return 'NEAR'
    elif rel_depth >= DEPTH_THRESHOLD_MID:
        return 'MEDIUM'
    else:
        return 'FAR'


def main():
    global server_running, socket_owned

    def sig_handler(signum, frame):
        global server_running
        print(f'[av-perception] Received signal ({signum}). Requesting graceful shutdown...')
        server_running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[av-perception] Hardware Device: {device.type.upper()}')

    # Load and warm up models BEFORE touching the socket
    yolo_model, midas_model, transform = verify_and_load_models(device)

    # Clean stale socket only after probe verification
    safe_socket_cleanup()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    socket_owned = True
    server.listen(1)
    server.settimeout(1.0)  # Non-blocking accept checks shutdown signal every 1.0s
    print(f'[av-perception] IPC Server listening on {SOCKET_PATH}...')

    current_conn: Optional[socket.socket] = None

    try:
        while server_running:
            try:
                current_conn, _ = server.accept()
            except socket.timeout:
                continue
            except Exception as e:
                if server_running:
                    print(f'[av-perception] Socket accept error: {e}')
                    time.sleep(0.1)
                continue

            print('[av-perception] Client bridge connected.')
            current_conn.settimeout(IPC_IO_TIMEOUT_SEC)

            try:
                while server_running:
                    # 1. Receive header (20 bytes: 3x uint32 [h, w, c] + 1x uint64 [length])
                    try:
                        hdr_bytes = recv_exact(current_conn, 20)
                    except (ConnectionError, socket.timeout):
                        break

                    h, w, c, length = struct.unpack('!IIIQ', hdr_bytes)

                    # 2. Strict dimension and payload bounds validation
                    if not (0 < h <= MAX_HEIGHT):
                        raise ValueError(f'Invalid height: {h} (limit: {MAX_HEIGHT})')
                    if not (0 < w <= MAX_WIDTH):
                        raise ValueError(f'Invalid width: {w} (limit: {MAX_WIDTH})')
                    if c != 3:
                        raise ValueError(f'Invalid channels: {c} (must be exactly 3)')
                    if length != (h * w * c):
                        raise ValueError(f'Payload length mismatch: {length} != {h * w * c}')
                    if length > MAX_PAYLOAD_BYTES:
                        raise ValueError(f'Payload exceeds maximum allowed: {length} > {MAX_PAYLOAD_BYTES}')

                    # 3. Receive exact raw image bytes
                    raw_bytes = recv_exact(current_conn, length)
                    frame_bgr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, c))

                    # --- 4. YOLOv8 Detection with Synchronized MPS Timing ---
                    if device.type == 'mps':
                        torch.mps.synchronize()
                    t0_det = time.perf_counter()
                    results = yolo_model.predict(
                        source=frame_bgr,
                        conf=0.25,
                        device=str(device),
                        verbose=False,
                        save=False
                    )
                    if device.type == 'mps':
                        torch.mps.synchronize()
                    t_det_ms = (time.perf_counter() - t0_det) * 1000.0

                    # --- 5. MiDaS Monocular Relative Depth with Synchronized MPS Timing ---
                    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    input_batch = transform(img_rgb).to(device)

                    if device.type == 'mps':
                        torch.mps.synchronize()
                    t0_depth = time.perf_counter()
                    with torch.no_grad():
                        depth_pred = midas_model(input_batch)
                        depth_pred = torch.nn.functional.interpolate(
                            depth_pred.unsqueeze(1),
                            size=(h, w),
                            mode='bicubic',
                            align_corners=False
                        ).squeeze()
                    if device.type == 'mps':
                        torch.mps.synchronize()
                    t_depth_ms = (time.perf_counter() - t0_depth) * 1000.0

                    depth_np = depth_pred.cpu().numpy()

                    # Robust depth output validation
                    if not np.all(np.isfinite(depth_np)):
                        raise ValueError('Non-finite values (NaN/Inf) in depth estimation map.')

                    d_min = float(depth_np.min())
                    d_max = float(depth_np.max())
                    d_range = d_max - d_min

                    if d_range < 1e-6:
                        depth_8u = np.zeros_like(depth_np, dtype=np.uint8)
                    else:
                        depth_norm = (depth_np - d_min) / (d_range + 1e-8)
                        depth_8u = (depth_norm * 255.0).astype(np.uint8)

                    # --- 6. Obstacle-Depth Fusion Filtered by RELEVANT_CLASSES ---
                    obstacles = []
                    for box in results[0].boxes:
                        if len(obstacles) >= MAX_OBSTACLES_PER_FRAME:
                            break

                        cls_id = int(box.cls[0].item())
                        if cls_id not in RELEVANT_CLASSES:
                            continue

                        conf = float(box.conf[0].item())
                        cls_name = RELEVANT_CLASSES[cls_id]
                        x1, y1, x2, y2 = [int(round(coord)) for coord in box.xyxy[0].tolist()]

                        if not all(math.isfinite(val) for val in (x1, y1, x2, y2, conf)):
                            continue

                        rel_depth_score = extract_box_relative_depth([x1, y1, x2, y2], depth_8u)
                        proximity_label = categorize_proximity(rel_depth_score)

                        obstacles.append({
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'confidence': round(conf, 3),
                            'bbox': [x1, y1, x2, y2],
                            'rel_depth': round(rel_depth_score, 1),
                            'proximity': proximity_label
                        })

                    # --- 7. Serialize Response ---
                    resp_dict = {
                        'det_latency_ms': round(t_det_ms, 2) if math.isfinite(t_det_ms) else 0.0,
                        'depth_latency_ms': round(t_depth_ms, 2) if math.isfinite(t_depth_ms) else 0.0,
                        'obstacles': obstacles
                    }
                    resp_payload = json.dumps(resp_dict).encode('utf-8')

                    if len(resp_payload) > MAX_RESP_BYTES:
                        raise ValueError('Response payload exceeds maximum allowed size.')

                    current_conn.sendall(struct.pack('!I', len(resp_payload)) + resp_payload)

            except Exception as e:
                print(f'[av-perception] Client communication exception: {e}')
            finally:
                if current_conn is not None:
                    try:
                        current_conn.close()
                    except Exception:
                        pass
                    current_conn = None
                print('[av-perception] Client disconnected. Ready for next connection.')

    finally:
        print('[av-perception] Shutting down listener...')
        try:
            server.close()
        except Exception:
            pass

        if socket_owned and os.path.exists(SOCKET_PATH):
            try:
                os.unlink(SOCKET_PATH)
                print(f'[av-perception] Socket {SOCKET_PATH} cleanly removed.')
            except OSError as e:
                print(f'[av-perception] Error removing socket: {e}')


if __name__ == '__main__':
    main()
