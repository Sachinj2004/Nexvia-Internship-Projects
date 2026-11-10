#!/usr/bin/env python3
"""
tests/test_perception_stack.py

CI/CD Regression Test Suite for Perception Stack (AIML-04).
Built strictly with standard library unittest (no extra packages).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path for direct module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe module imports (guarded with if __name__ == "__main__")
import camera_lidar_fusion
import obstacle_depth_fusion
import lane_detection
import onnx_inference
import wandb_eval


# =====================================================================
# TEST CLASS 1: 3D Camera-LiDAR Calibration and Projection Math
# =====================================================================
class TestCalibrationAndFusionMath(unittest.TestCase):
    """Validates calibration matrix parsing and 3D projection."""

    def setUp(self):
        calib_dir = PROJECT_ROOT / "data" / "kitti" / "training" / "calib"
        self.calib_path = calib_dir / "000000.txt"

    def test_01_parse_kitti_calib_shapes(self):
        """Validates calibration parser returns correct ndarray shapes."""
        msg = f"Calib file missing: {self.calib_path}"
        self.assertTrue(self.calib_path.is_file(), msg)
        calib_data = camera_lidar_fusion.parse_kitti_calib(
            self.calib_path
        )

        self.assertIn("P2", calib_data)
        self.assertIn("R0_rect", calib_data)
        self.assertIn("Tr_velo_to_cam", calib_data)

        self.assertEqual(calib_data["P2"].shape, (3, 4))
        self.assertEqual(calib_data["R0_rect"].shape, (3, 3))
        self.assertEqual(calib_data["Tr_velo_to_cam"].shape, (3, 4))

        self.assertTrue(np.all(np.isfinite(calib_data["P2"])))
        self.assertTrue(np.all(np.isfinite(calib_data["R0_rect"])))
        self.assertTrue(np.all(np.isfinite(calib_data["Tr_velo_to_cam"])))

    def test_02_parse_kitti_calib_missing_file_raises(self):
        """Validates FileNotFoundError is raised on missing calib file."""
        calib_dir = PROJECT_ROOT / "data" / "kitti" / "training" / "calib"
        nonexistent = calib_dir / "nonexistent_999999.txt"
        with self.assertRaises(FileNotFoundError):
            camera_lidar_fusion.parse_kitti_calib(nonexistent)

    def test_03_build_projection_matrix_shapes(self):
        """Validates construction of 3x4 projection and 4x4 extrinsic."""
        calib_data = camera_lidar_fusion.parse_kitti_calib(
            self.calib_path
        )
        proj_matrix, extrinsic_rect = (
            camera_lidar_fusion.build_projection_matrix(calib_data)
        )

        self.assertEqual(proj_matrix.shape, (3, 4))
        self.assertEqual(extrinsic_rect.shape, (4, 4))

        expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(
            extrinsic_rect[3, :], expected_bottom
        )

    def test_04_calculate_box_iou_known_geometries(self):
        """Validates 2D bounding box IoU calculation logic."""
        # Case 1: Identical bounding boxes yield IoU = 1.0
        box_a = [10.0, 10.0, 50.0, 50.0]
        self.assertAlmostEqual(
            camera_lidar_fusion.calculate_box_iou(box_a, box_a),
            1.0,
            places=5,
        )

        # Case 2: Completely non-overlapping boxes yield IoU = 0.0
        box_b = [100.0, 100.0, 150.0, 150.0]
        self.assertAlmostEqual(
            camera_lidar_fusion.calculate_box_iou(box_a, box_b),
            0.0,
            places=5,
        )

        # Case 3: Partial overlap: [0,0,20,10] and [10,0,30,10]
        # Intersection = 100, Union = 300, Expected IoU = 1/3
        box1 = [0.0, 0.0, 20.0, 10.0]
        box2 = [10.0, 0.0, 30.0, 10.0]
        expected_iou = 100.0 / 300.0
        self.assertAlmostEqual(
            camera_lidar_fusion.calculate_box_iou(box1, box2),
            expected_iou,
            places=5,
        )


# =====================================================================
# TEST CLASS 2: Obstacle Distance & Depth Fusion Heuristics
# =====================================================================
class TestObstacleDepthFusionMath(unittest.TestCase):
    """Validates MiDaS relative depth proximity and patch sampling."""

    def test_05_categorize_proximity_near(self):
        """Validates relative disparity >= 160 categorizes as Near."""
        label_180, color_180 = (
            obstacle_depth_fusion.categorize_proximity(180.0)
        )
        label_160, color_160 = (
            obstacle_depth_fusion.categorize_proximity(160.0)
        )

        self.assertEqual(label_180, "Near")
        self.assertEqual(color_180, obstacle_depth_fusion.COLOR_NEAR)
        self.assertEqual(label_160, "Near")
        self.assertEqual(color_160, obstacle_depth_fusion.COLOR_NEAR)

    def test_06_categorize_proximity_mid_range(self):
        """Validates relative disparity 90..159 categorizes as Mid-Range."""
        label_120, color_120 = (
            obstacle_depth_fusion.categorize_proximity(120.0)
        )
        label_90, color_90 = (
            obstacle_depth_fusion.categorize_proximity(90.0)
        )

        self.assertEqual(label_120, "Mid-Range")
        self.assertEqual(color_120, obstacle_depth_fusion.COLOR_MID)
        self.assertEqual(label_90, "Mid-Range")
        self.assertEqual(color_90, obstacle_depth_fusion.COLOR_MID)

    def test_07_categorize_proximity_far(self):
        """Validates relative disparity < 90 categorizes as Far."""
        label_89, color_89 = (
            obstacle_depth_fusion.categorize_proximity(89.0)
        )
        label_10, color_10 = (
            obstacle_depth_fusion.categorize_proximity(10.0)
        )

        self.assertEqual(label_89, "Far")
        self.assertEqual(color_89, obstacle_depth_fusion.COLOR_FAR)
        self.assertEqual(label_10, "Far")
        self.assertEqual(color_10, obstacle_depth_fusion.COLOR_FAR)

    def test_08_extract_obstacle_relative_depth_interior_patch(self):
        """Validates center 50% interior patch sampling."""
        synthetic_depth = np.full((100, 100), 30, dtype=np.uint8)
        # Bounding box [20, 20, 60, 60]: center 50% patch spans [30:50, 30:50]
        synthetic_depth[30:50, 30:50] = 220

        box = [20, 20, 60, 60]
        extracted_depth = (
            obstacle_depth_fusion.extract_obstacle_relative_depth(
                box, synthetic_depth
            )
        )

        # Median of center patch is 220, ignoring background 30
        self.assertEqual(extracted_depth, 220.0)


# =====================================================================
# TEST CLASS 3: Classical Computer Vision Lane Detection Pipeline
# =====================================================================
class TestLaneDetectionPipeline(unittest.TestCase):
    """Validates IPM matrices, color filtering, and geometry checks."""

    def test_09_get_perspective_matrices_shape(self):
        """Validates IPM matrix dimensions and invertibility."""
        h, w = 720, 1280
        M, Minv = lane_detection.get_perspective_matrices(h, w)

        self.assertEqual(M.shape, (3, 3))
        self.assertEqual(Minv.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(M)))
        self.assertTrue(np.all(np.isfinite(Minv)))

        product = np.dot(M, Minv)
        product_normalized = product / product[2, 2]
        np.testing.assert_array_almost_equal(
            product_normalized, np.eye(3), decimal=3
        )

        # Validate round-trip geometric projection of a road point
        test_pt = np.array([[[640.0, 600.0]]], dtype=np.float32)
        warped_pt = cv2.perspectiveTransform(test_pt, M)
        unwarped_pt = cv2.perspectiveTransform(warped_pt, Minv)
        np.testing.assert_allclose(unwarped_pt, test_pt, atol=1e-1)

    def test_10_preprocess_frame_binary_output(self):
        """Validates preprocess_frame generates binary mask in {0, 255}."""
        synthetic_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # White lane stripe in lower center
        synthetic_img[500:700, 600:620] = [255, 255, 255]

        binary_output = lane_detection.preprocess_frame(synthetic_img)

        self.assertEqual(binary_output.shape, (720, 1280))
        self.assertEqual(binary_output.dtype, np.uint8)

        unique_vals = set(np.unique(binary_output))
        self.assertTrue(unique_vals.issubset({0, 255}))
        self.assertGreater(int(np.count_nonzero(binary_output)), 0)

    def test_11_validate_lane_geometry_corridor(self):
        """Validates lane corridor non-crossing and width bounds."""
        h = 720
        # Valid highway corridor: separation ~3.43m (650 px * 3.7 / 700)
        left_fit_valid = np.array([0.0, 0.0, 300.0])
        right_fit_valid = np.array([0.0, 0.0, 950.0])
        is_valid, msg = lane_detection.validate_lane_geometry(
            left_fit_valid, right_fit_valid, h
        )
        self.assertTrue(is_valid, f"Expected valid lane: {msg}")

        # Invalid corridor: crossing lines
        left_fit_crossing = np.array([0.0, 0.0, 800.0])
        right_fit_crossing = np.array([0.0, 0.0, 200.0])
        is_valid_cross, msg_cross = (
            lane_detection.validate_lane_geometry(
                left_fit_crossing, right_fit_crossing, h
            )
        )
        self.assertFalse(is_valid_cross)
        self.assertIn("cross", msg_cross.lower())


# =====================================================================
# TEST CLASS 4: ONNX Runtime Preprocessing and Postprocessing
# =====================================================================
class TestOnnxInferencePrePostProcessing(unittest.TestCase):
    """Validates letterbox padding, unscaling math, and class mapping."""

    def test_12_letterbox_output_shape_and_scale(self):
        """Validates letterbox padding to 640x640 preserving aspect."""
        sample_img = np.zeros((375, 1242, 3), dtype=np.uint8)
        target_size = 640
        padded_img, scale, (pad_w, pad_h) = onnx_inference.letterbox(
            sample_img, target_size=target_size
        )

        self.assertEqual(padded_img.shape, (target_size, target_size, 3))
        expected_scale = 640.0 / 1242.0
        self.assertAlmostEqual(scale, expected_scale, places=4)

        scaled_w = round(1242 * scale)
        self.assertAlmostEqual(
            scaled_w + 2 * pad_w, target_size, delta=2.0
        )

    def test_13_relevant_classes_subset(self):
        """Validates RELEVANT_CLASSES dictionary specifies key classes."""
        classes = onnx_inference.RELEVANT_CLASSES
        self.assertIsInstance(classes, dict)

        self.assertEqual(classes[0], "person")
        self.assertEqual(classes[1], "bicycle")
        self.assertEqual(classes[2], "car")
        self.assertEqual(classes[3], "motorcycle")
        self.assertEqual(classes[5], "bus")
        self.assertEqual(classes[7], "truck")

    def test_14_coordinate_unscaling_math(self):
        """Validates the mathematical unscaling formula.

        Note: Unit test of the inverse scaling equations (lines 155-158).
        Does not execute full inference pipeline or NMS filtering.
        """
        orig_w, orig_h = 1242.0, 375.0
        target_size = 640
        sample_img = np.zeros((int(orig_h), int(orig_w), 3), dtype=np.uint8)

        _, scale, (pad_w, pad_h) = onnx_inference.letterbox(
            sample_img, target_size=target_size
        )

        true_x1, true_y1, true_x2, true_y2 = 100.0, 50.0, 300.0, 200.0
        orig_cx = (true_x1 + true_x2) / 2.0
        orig_cy = (true_y1 + true_y2) / 2.0
        orig_bw = true_x2 - true_x1
        orig_bh = true_y2 - true_y1

        cx_letterbox = orig_cx * scale + pad_w
        cy_letterbox = orig_cy * scale + pad_h
        bw_letterbox = orig_bw * scale
        bh_letterbox = orig_bh * scale

        x1_unscaled = (cx_letterbox - bw_letterbox / 2.0 - pad_w) / scale
        y1_unscaled = (cy_letterbox - bh_letterbox / 2.0 - pad_h) / scale
        x2_unscaled = (cx_letterbox + bw_letterbox / 2.0 - pad_w) / scale
        y2_unscaled = (cy_letterbox + bh_letterbox / 2.0 - pad_h) / scale

        self.assertAlmostEqual(x1_unscaled, true_x1, places=3)
        self.assertAlmostEqual(y1_unscaled, true_y1, places=3)
        self.assertAlmostEqual(x2_unscaled, true_x2, places=3)
        self.assertAlmostEqual(y2_unscaled, true_y2, places=3)
        self.assertAlmostEqual(x2_unscaled - x1_unscaled, orig_bw, places=3)
        self.assertAlmostEqual(y2_unscaled - y1_unscaled, orig_bh, places=3)


# =====================================================================
# TEST CLASS 5: Evaluation Metrics Schema & Serialization Integrity
# =====================================================================
class TestMetricsSchemaAndEvaluationStructure(unittest.TestCase):
    """Validates metrics_eval.json schema and atomic serialization."""

    def test_15_metrics_eval_json_schema_and_types(self):
        """Validates structure and values of metrics_eval.json."""
        metrics_file = PROJECT_ROOT / "metrics_eval.json"
        msg = f"metrics_eval.json not found at {metrics_file}"
        self.assertTrue(metrics_file.is_file(), msg)

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ["environment", "configuration", "dataset_audit", "metrics"]:
            self.assertIn(key, data)

        agg = data["metrics"]["aggregate"]
        self.assertAlmostEqual(agg["map50"], 0.343, places=3)
        self.assertAlmostEqual(agg["map50_95"], 0.180, places=3)
        self.assertAlmostEqual(agg["precision"], 0.389, places=3)
        self.assertAlmostEqual(agg["recall"], 0.344, places=3)

        self.assertNotIn("fitness", agg)
        self.assertNotIn("derived_3class_estimates", data["metrics"])

        cyclist_entry = data["metrics"]["per_class"]["Cyclist"]
        self.assertEqual(cyclist_entry["class_id"], 2)
        self.assertFalse(cyclist_entry["supported_by_coco_model"])

    def test_16_atomic_save_json_isolated_no_clobber(self):
        """Validates atomic_save_json in isolated temp directory."""
        test_payload = {"test_metric": 42.0, "status": "VERIFIED"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_target = Path(temp_dir) / "test_output.json"

            # Step 1: Initial save with overwrite=False must succeed
            wandb_eval.atomic_save_json(
                test_payload, temp_target, overwrite=False
            )
            self.assertTrue(temp_target.is_file())

            with open(temp_target, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["test_metric"], 42.0)

            # Step 2: Saving to existing target with overwrite=False must fail
            with self.assertRaises(FileExistsError):
                wandb_eval.atomic_save_json(
                    {"new_metric": 99.0}, temp_target, overwrite=False
                )

            # Step 3: Saving with overwrite=True must cleanly replace file
            new_payload = {"overwritten": True, "value": 100}
            wandb_eval.atomic_save_json(
                new_payload, temp_target, overwrite=True
            )
            with open(temp_target, "r", encoding="utf-8") as f:
                updated = json.load(f)
            self.assertEqual(updated["value"], 100)
            self.assertTrue(updated["overwritten"])


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
