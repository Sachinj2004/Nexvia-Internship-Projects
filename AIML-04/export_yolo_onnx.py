#!/usr/bin/env python3
"""
export_yolo_onnx.py
Exports existing YOLOv8n PyTorch checkpoint to ONNX format.
Validates the exported graph integrity and verifies checksum preservation.
"""

import sys
import hashlib
from pathlib import Path
import onnx
from ultralytics import YOLO

EXPECTED_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_model():
    project_dir = Path(__file__).resolve().parent
    pt_path = project_dir / "yolov8n.pt"
    onnx_path = project_dir / "yolov8n.onnx"

    print("=" * 70)
    print("YOLOV8N ONNX EXPORT & INTEGRITY VALIDATION")
    print("=" * 70)

    # 1. Verify source PyTorch model existence & checksum
    if not pt_path.is_file():
        raise FileNotFoundError(f"[ERROR] Required PyTorch model not found: {pt_path}")

    sha_before = compute_sha256(pt_path)
    print(f"[INFO] Source PyTorch Model: {pt_path.name}")
    print(f"[INFO] Size: {pt_path.stat().st_size:,} bytes ({pt_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[INFO] SHA-256 (Pre-export): {sha_before}")

    if sha_before != EXPECTED_SHA256:
        raise ValueError(f"[ERROR] Checksum mismatch! Expected {EXPECTED_SHA256}, got {sha_before}")

    # 2. Pre-export check: ensure yolov8n.onnx does not already exist
    if onnx_path.exists():
        raise FileExistsError(
            f"[ABORT] {onnx_path} already exists! Halting per safety condition."
        )

    # 3. Perform ONNX export
    print("\n[INFO] Initializing YOLOv8n from PyTorch weights...")
    model = YOLO(str(pt_path))

    print("[INFO] Exporting to ONNX format...")
    print("       - Input size    : 640 x 640")
    print("       - ONNX opset    : 17")
    print("       - Dynamic batch : True")
    print("       - Precision     : FP32 (half=False)")
    print("       - Simplify      : False")

    exported_path = model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        dynamic=True,
        half=False,
        simplify=False
    )

    print(f"[SUCCESS] Export completed. Output: {exported_path}")

    if not onnx_path.is_file():
        raise FileNotFoundError(f"[ERROR] Exported ONNX file missing at expected location: {onnx_path}")

    # 4. Post-export verification of original PyTorch model integrity
    sha_after = compute_sha256(pt_path)
    print(f"\n[INFO] SHA-256 (Post-export): {sha_after}")
    if sha_before != sha_after:
        raise RuntimeError("[CRITICAL] Original PyTorch model was altered during export!")
    print("[SUCCESS] Verified yolov8n.pt was NOT modified, overwritten, or corrupted.")

    # 5. Validate exported ONNX model structural integrity
    print(f"\n[INFO] Validating exported ONNX graph via onnx.checker.check_model()...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] ONNX structural integrity check PASSED!")

    # 6. Extract input and output tensor metadata
    graph = onnx_model.graph
    print("\n=== Model Tensor Specification ===")
    print(f"Graph Name: {graph.name}")

    print("\nInputs:")
    for inp in graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param or "dynamic"
                 for dim in inp.type.tensor_type.shape.dim]
        elem_type = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"  - Name: {inp.name} | Type: {elem_type} | Shape: {shape}")

    print("\nOutputs:")
    for out in graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param or "dynamic"
                 for dim in out.type.tensor_type.shape.dim]
        elem_type = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"  - Name: {out.name} | Type: {elem_type} | Shape: {shape}")

    # 7. Model file size comparison
    pt_size_mb = pt_path.stat().st_size / (1024 * 1024)
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print("\n=== File Footprint Comparison ===")
    print(f"  PyTorch Checkpoint (.pt) : {pt_size_mb:.2f} MB ({pt_path.stat().st_size:,} bytes)")
    print(f"  Exported ONNX Model (.onnx): {onnx_size_mb:.2f} MB ({onnx_path.stat().st_size:,} bytes)")
    print("=" * 70)


if __name__ == "__main__":
    export_model()
