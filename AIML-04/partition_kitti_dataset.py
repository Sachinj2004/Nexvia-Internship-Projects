"""
KITTI Dataset Partitioning and YOLOv8 Configuration
Creates an 80/20 train-validation split (Seed 42) using symbolic links
to avoid duplicating the 5.8 GB image dataset.
Generates train.txt, val.txt, and kitti.yaml with the approved 4-class mapping.
"""

import os
import random
import time
from pathlib import Path
from collections import Counter
import yaml
from ultralytics.data.utils import check_det_dataset


def partition_and_configure():
    print("=" * 80)
    print("KITTI DATASET PARTITIONING & YOLOV8 CONFIGURATION")
    print("=" * 80)

    project_dir = Path(__file__).resolve().parent
    kitti_root = project_dir / "data" / "kitti"
    src_images_dir = kitti_root / "training" / "image_2"
    src_labels_dir = kitti_root / "yolo_labels" / "training"

    if not src_images_dir.is_dir() or not src_labels_dir.is_dir():
        raise FileNotFoundError(f"[ERROR] Source images or labels missing under {kitti_root}")

    # 1. Reproducible split with seed 42
    all_stems = sorted([f.stem for f in src_images_dir.glob("*.png")])
    total_samples = len(all_stems)
    assert total_samples == 7481, f"Expected 7481 samples, found {total_samples}"

    rng = random.Random(42)
    shuffled_stems = list(all_stems)
    rng.shuffle(shuffled_stems)

    train_count = 5985
    val_count = 1496
    train_stems = sorted(shuffled_stems[:train_count])
    val_stems = sorted(shuffled_stems[train_count:])

    print(f"[INFO] Total Dataset Samples   : {total_samples:,}")
    print(f"[INFO] Training Split (80%)    : {len(train_stems):,} samples")
    print(f"[INFO] Validation Split (20%)  : {len(val_stems):,} samples")
    print(f"[INFO] Random Seed             : 42 (Reproducible)")
    print("-" * 80)

    # 2. Setup standard YOLO directories
    splits_dir = kitti_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    images_train_dir = kitti_root / "images" / "train"
    images_val_dir = kitti_root / "images" / "val"
    labels_train_dir = kitti_root / "labels" / "train"
    labels_val_dir = kitti_root / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Create splits text files (train.txt and val.txt)
    train_txt_path = splits_dir / "train.txt"
    val_txt_path = splits_dir / "val.txt"

    train_lines = []
    val_lines = []

    print("[INFO] Creating standard Ultralytics directory symlinks (0 bytes duplication)...")
    t0 = time.perf_counter()

    # Process training symlinks
    for stem in train_stems:
        img_src = (src_images_dir / f"{stem}.png").resolve()
        lbl_src = (src_labels_dir / f"{stem}.txt").resolve()

        img_dst = images_train_dir / f"{stem}.png"
        lbl_dst = labels_train_dir / f"{stem}.txt"

        if not (img_dst.exists() or img_dst.is_symlink()):
            os.symlink(img_src, img_dst)
        if not (lbl_dst.exists() or lbl_dst.is_symlink()):
            os.symlink(lbl_src, lbl_dst)

        train_lines.append(f"./images/train/{stem}.png")

    # Process validation symlinks
    for stem in val_stems:
        img_src = (src_images_dir / f"{stem}.png").resolve()
        lbl_src = (src_labels_dir / f"{stem}.txt").resolve()

        img_dst = images_val_dir / f"{stem}.png"
        lbl_dst = labels_val_dir / f"{stem}.txt"

        if not (img_dst.exists() or img_dst.is_symlink()):
            os.symlink(img_src, img_dst)
        if not (lbl_dst.exists() or lbl_dst.is_symlink()):
            os.symlink(lbl_src, lbl_dst)

        val_lines.append(f"./images/val/{stem}.png")

    train_txt_path.write_text("\n".join(train_lines) + "\n")
    val_txt_path.write_text("\n".join(val_lines) + "\n")

    elapsed_symlinks = time.perf_counter() - t0
    print(f"[SUCCESS] Created 14,962 symlinks and split text files in {elapsed_symlinks:.2f} seconds.")
    print("-" * 80)

    # 4. Create kitti.yaml configuration
    kitti_yaml_path = project_dir / "kitti.yaml"
    yaml_content = {
        "path": str(kitti_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "Car",
            1: "People",
            2: "Cyclist",
            3: "Truck",
        },
    }

    with open(kitti_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"[SUCCESS] Created YOLOv8 dataset configuration: {kitti_yaml_path}")
    print("-" * 80)

    # 5. Class Distribution in Splits
    train_classes = Counter()
    val_classes = Counter()

    for s in train_stems:
        c = (src_labels_dir / f"{s}.txt").read_text().strip()
        if c:
            for line in c.split("\n"):
                train_classes[int(line.split()[0])] += 1

    for s in val_stems:
        c = (src_labels_dir / f"{s}.txt").read_text().strip()
        if c:
            for line in c.split("\n"):
                val_classes[int(line.split()[0])] += 1

    print("SPLIT CLASS DISTRIBUTION ANALYSIS:")
    class_names = {0: "Car", 1: "People", 2: "Cyclist", 3: "Truck"}
    for cid in range(4):
        tc = train_classes[cid]
        vc = val_classes[cid]
        total_c = tc + vc
        print(
            f"  Class {cid} ({class_names[cid]:<7}): "
            f"Train = {tc:5,d} ({tc/sum(train_classes.values())*100:5.2f}%) | "
            f"Val = {vc:5,d} ({vc/sum(val_classes.values())*100:5.2f}%) | "
            f"Total = {total_c:5,d}"
        )
    print("-" * 80)

    # 6. Ultralytics Dataset Validation Check
    print("[INFO] Running Ultralytics check_det_dataset validation on kitti.yaml...")
    val_result = check_det_dataset(str(kitti_yaml_path))
    print(f"[SUCCESS] Dataset check PASSED!")
    print(f"  * Dataset root : {val_result['path']}")
    print(f"  * Train set    : {val_result['train']}")
    print(f"  * Val set      : {val_result['val']}")
    print(f"  * Classes      : {val_result['names']}")
    print("=" * 80)


if __name__ == "__main__":
    partition_and_configure()
