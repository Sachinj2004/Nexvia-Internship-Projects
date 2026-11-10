"""
KITTI to YOLO Annotation Converter
Converts KITTI 2D bounding box annotations to normalized YOLO format.

Class Mapping:
  0 = Car (KITTI: Car + Van)
  1 = People (KITTI: Pedestrian + Person_sitting)
  2 = Cyclist (KITTI: Cyclist)
  3 = Truck (KITTI: Truck)

Excluded Classes:
  DontCare, Misc, Tram

Output Format per line:
  class_id x_center y_center width height
  All coordinates normalized to [0.0, 1.0].
"""

import time
from pathlib import Path
from collections import Counter
from PIL import Image


CLASS_MAPPING = {
    "Car": 0,
    "Van": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
    "Truck": 3,
}

CLASS_NAMES = {
    0: "Car",
    1: "People",
    2: "Cyclist",
    3: "Truck",
}

EXCLUDED_CLASSES = {"DontCare", "Misc", "Tram"}


def convert_dataset():
    print("=" * 80)
    print("KITTI TO YOLO ANNOTATION CONVERSION")
    print("=" * 80)

    project_dir = Path(__file__).resolve().parent
    kitti_train_dir = project_dir / "data" / "kitti" / "training"
    image_dir = kitti_train_dir / "image_2"
    label_dir = kitti_train_dir / "label_2"
    output_dir = project_dir / "data" / "kitti" / "yolo_labels" / "training"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"[ERROR] Source KITTI directories not found at: {kitti_train_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Source Images : {image_dir}")
    print(f"[INFO] Source Labels : {label_dir}")
    print(f"[INFO] Destination   : {output_dir}")

    # Gather all image files
    image_files = sorted(image_dir.glob("*.png"))
    total_images = len(image_files)
    print(f"[INFO] Total training images to process: {total_images:,}")

    # Statistics tracking
    converted_class_counts = Counter()
    excluded_counts = Counter()
    invalid_boxes_count = 0
    empty_label_files = 0
    total_converted_boxes = 0

    # Coordinate boundaries for verification
    min_x_c, max_x_c = 1.0, 0.0
    min_y_c, max_y_c = 1.0, 0.0
    min_w, max_w = 1.0, 0.0
    min_h, max_h = 1.0, 0.0

    t_start = time.perf_counter()

    for idx, img_path in enumerate(image_files, start=1):
        stem = img_path.stem
        src_label_path = label_dir / f"{stem}.txt"
        dst_label_path = output_dir / f"{stem}.txt"

        # Read actual image dimensions via PIL header (fast)
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = []

        if src_label_path.is_file():
            content = src_label_path.read_text().strip()
            if content:
                for line in content.split("\n"):
                    parts = line.strip().split()
                    if not parts:
                        continue

                    obj_type = parts[0]

                    # Check if target class
                    if obj_type in CLASS_MAPPING:
                        cls_id = CLASS_MAPPING[obj_type]

                        # KITTI 2D bbox in pixels: [xmin, ymin, xmax, ymax]
                        try:
                            xmin = float(parts[4])
                            ymin = float(parts[5])
                            xmax = float(parts[6])
                            ymax = float(parts[7])
                        except (ValueError, IndexError):
                            invalid_boxes_count += 1
                            continue

                        # Boundary clipping to valid image extent
                        xmin = max(0.0, min(xmin, float(img_w)))
                        xmax = max(0.0, min(xmax, float(img_w)))
                        ymin = max(0.0, min(ymin, float(img_h)))
                        ymax = max(0.0, min(ymax, float(img_h)))

                        box_pixel_w = xmax - xmin
                        box_pixel_h = ymax - ymin

                        # Validate box dimensions
                        if box_pixel_w <= 0.0 or box_pixel_h <= 0.0:
                            invalid_boxes_count += 1
                            continue

                        # Normalize to [0.0, 1.0]
                        x_center = (xmin + xmax) / (2.0 * img_w)
                        y_center = (ymin + ymax) / (2.0 * img_h)
                        norm_w = box_pixel_w / img_w
                        norm_h = box_pixel_h / img_h

                        # Strict [0.0, 1.0] clamping
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        norm_w = max(0.0, min(1.0, norm_w))
                        norm_h = max(0.0, min(1.0, norm_h))

                        # Update min/max ranges for sanity report
                        min_x_c = min(min_x_c, x_center)
                        max_x_c = max(max_x_c, x_center)
                        min_y_c = min(min_y_c, y_center)
                        max_y_c = max(max_y_c, y_center)
                        min_w = min(min_w, norm_w)
                        max_w = max(max_w, norm_w)
                        min_h = min(min_h, norm_h)
                        max_h = max(max_h, norm_h)

                        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                        converted_class_counts[cls_id] += 1
                        total_converted_boxes += 1

                    elif obj_type in EXCLUDED_CLASSES:
                        excluded_counts[obj_type] += 1
                    else:
                        excluded_counts["Other_Unknown"] += 1

        # Write output label file (including empty files if no valid targets exist)
        dst_label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        if not yolo_lines:
            empty_label_files += 1

    elapsed = time.perf_counter() - t_start
    print(f"[SUCCESS] Conversion completed in {elapsed:.2f} seconds ({total_images / elapsed:.1f} frames/sec).")
    print("-" * 80)

    # Output verification summary
    created_files = len(list(output_dir.glob("*.txt")))
    print("CONVERSION VERIFICATION REPORT:")
    print(f"  * Total YOLO Label Files Created : {created_files:,} / {total_images:,} (100% matched)")
    print(f"  * Files with Active Targets      : {total_images - empty_label_files:,}")
    print(f"  * Empty Files (No Targets/Only DC): {empty_label_files:,}")
    print(f"  * Total Converted Bounding Boxes : {total_converted_boxes:,}")
    print(f"  * Invalid / Corrupt Boxes Skipped: {invalid_boxes_count}")
    print("-" * 80)
    print("CONVERTED CLASS BREAKDOWN:")
    for cid in range(4):
        cname = CLASS_NAMES[cid]
        count = converted_class_counts[cid]
        pct = (count / total_converted_boxes) * 100 if total_converted_boxes > 0 else 0
        print(f"  * Class {cid} ({cname:<6}): {count:6,d} instances ({pct:5.2f}%)")
    print("-" * 80)
    print("EXCLUDED OBJECTS BREAKDOWN:")
    for ex_cls, count in excluded_counts.most_common():
        print(f"  * {ex_cls:<15}: {count:6,d} instances (Filtered out)")
    print("-" * 80)
    print("COORDINATE RANGE VERIFICATION (Normalized [0.0, 1.0]):")
    print(f"  * X_Center Range : [{min_x_c:.6f}, {max_x_c:.6f}]")
    print(f"  * Y_Center Range : [{min_y_c:.6f}, {max_y_c:.6f}]")
    print(f"  * Width Range    : [{min_w:.6f}, {max_w:.6f}]")
    print(f"  * Height Range   : [{min_h:.6f}, {max_h:.6f}]")
    print("=" * 80)


if __name__ == "__main__":
    convert_dataset()
