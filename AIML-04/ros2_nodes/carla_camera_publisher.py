#!/usr/bin/env python3
"""
CarlaCameraPublisherNode
ROS 2 Humble Node for streaming CARLA-compatible camera frames and calibration metadata.

Topics Published:
  - /carla/ego_vehicle/camera/rgb/image_raw   (sensor_msgs/Image, bgr8)
  - /carla/ego_vehicle/camera/rgb/camera_info (sensor_msgs/CameraInfo)

Data Source:
  - Offline KITTI dataset replay harness (image_2/*.png and calib/*.txt).
  - Explicitly identified as offline replay, not live CARLA simulation output.
"""

import os
import sys
import math
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class CarlaCameraPublisherNode(Node):
    def __init__(self):
        super().__init__('carla_camera_publisher')

        # Configurable parameters
        self.declare_parameter(
            'kitti_image_dir',
            os.path.expanduser('~/my intern/Autonomous-Driving-Perception-Stack/data/kitti/training/image_2')
        )
        self.declare_parameter(
            'kitti_calib_dir',
            os.path.expanduser('~/my intern/Autonomous-Driving-Perception-Stack/data/kitti/training/calib')
        )
        self.declare_parameter(
            'fallback_image',
            os.path.expanduser('~/my intern/Autonomous-Driving-Perception-Stack/sample_road.jpg')
        )
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('frame_id', 'ego_vehicle/camera/rgb/front')

        # Parameter validation
        raw_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        if not math.isfinite(raw_rate) or raw_rate <= 0.0:
            self.get_logger().error(f'Invalid publish_rate_hz ({raw_rate}). Clamping to default 10.0 Hz.')
            rate_hz = 10.0
        elif raw_rate > 60.0:
            self.get_logger().warn(f'Publish rate ({raw_rate} Hz) exceeds safe 60.0 Hz ceiling. Clamping to 60.0 Hz.')
            rate_hz = 60.0
        else:
            rate_hz = raw_rate

        self.kitti_img_dir = Path(self.get_parameter('kitti_image_dir').get_parameter_value().string_value)
        self.kitti_calib_dir = Path(self.get_parameter('kitti_calib_dir').get_parameter_value().string_value)
        self.fallback_path = Path(self.get_parameter('fallback_image').get_parameter_value().string_value)
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        self.image_pub = self.create_publisher(Image, '/carla/ego_vehicle/camera/rgb/image_raw', sensor_qos)
        self.info_pub = self.create_publisher(CameraInfo, '/carla/ego_vehicle/camera/rgb/camera_info', sensor_qos)

        self.bridge = CvBridge()
        self.image_paths: List[Path] = []
        self.current_idx = 0
        self.consecutive_read_failures = 0

        # Scan for offline KITTI frames
        if self.kitti_img_dir.is_dir():
            self.image_paths = sorted(self.kitti_img_dir.glob('*.png'))

        if self.image_paths:
            self.get_logger().info(
                f'[OFFLINE-KITTI-HARNESS] Found {len(self.image_paths)} images in {self.kitti_img_dir}. '
                f'(NOTICE: Offline test replay; not live CARLA simulation output).'
            )
        else:
            self.get_logger().warn(
                f'[OFFLINE-REPLAY] No KITTI images found in {self.kitti_img_dir}. Using fallback: {self.fallback_path}'
            )

        self.timer = self.create_timer(1.0 / rate_hz, self.publish_frame)
        self.get_logger().info(f'CarlaCameraPublisherNode initialized at {rate_hz:.1f} Hz.')

    def parse_kitti_calib(self, calib_file: Path) -> Optional[Tuple[List[float], List[float], List[float]]]:
        """
        Parses KITTI calibration text file for rectified color camera 2.
        Validates all values are finite and focal lengths are plausible.
        Returns: (P2 [12], K [9], R [9]) or None if invalid.
        """
        if not calib_file.is_file():
            return None

        p2 = None
        try:
            with open(calib_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0] == 'P2:':
                        p2 = [float(x) for x in parts[1:13]]
                        break

            if p2 is not None and len(p2) == 12:
                if not all(math.isfinite(x) for x in p2):
                    self.get_logger().warn(f'Non-finite values detected in calib file {calib_file}.')
                    return None

                fx = p2[0]
                fy = p2[5]
                cx = p2[2]
                cy = p2[6]

                if not (100.0 <= fx <= 10000.0 and 100.0 <= fy <= 10000.0 and cx > 0.0 and cy > 0.0):
                    self.get_logger().warn(f'Plausibility check failed for camera intrinsics in {calib_file}.')
                    return None

                k = [
                    p2[0], p2[1], p2[2],
                    p2[4], p2[5], p2[6],
                    p2[8], p2[9], p2[10]
                ]
                r_identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                return p2, k, r_identity

        except Exception as e:
            self.get_logger().warn(f'Failed to parse calib file {calib_file}: {e}')
        return None

    def publish_frame(self):
        # 1. Load frame with controlled failure handling
        if self.image_paths:
            img_path = self.image_paths[self.current_idx]
            stem = img_path.stem
            frame = cv2.imread(str(img_path))
            calib_file = self.kitti_calib_dir / f'{stem}.txt'
            self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        else:
            frame = cv2.imread(str(self.fallback_path))
            calib_file = None

        if frame is None:
            self.consecutive_read_failures += 1
            if self.consecutive_read_failures <= 5 or self.consecutive_read_failures % 50 == 0:
                self.get_logger().error(f'Failed to read frame (failure count: {self.consecutive_read_failures}).')
            return

        self.consecutive_read_failures = 0
        h, w = frame.shape[:2]
        now = self.get_clock().now().to_msg()

        # 2. Build sensor_msgs/Image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.frame_id

        # 3. Build sensor_msgs/CameraInfo
        info_msg = CameraInfo()
        info_msg.header.stamp = now
        info_msg.header.frame_id = self.frame_id
        info_msg.height = h
        info_msg.width = w

        calib_data = self.parse_kitti_calib(calib_file) if calib_file else None
        if calib_data is not None:
            p2, k, r = calib_data
            info_msg.distortion_model = 'plumb_bob'
            info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Rectified image
            info_msg.k = k
            info_msg.p = p2
            info_msg.r = r
        else:
            # Fallback per REP-104: empty distortion_model, zero K/P, valid 3x3 identity R
            info_msg.distortion_model = ''
            info_msg.d = []
            info_msg.k = [0.0] * 9
            info_msg.p = [0.0] * 12
            info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            self.get_logger().warn(
                'No valid calibration found. Publishing UNCALIBRATED CameraInfo (distortion_model="", identity R).',
                throttle_duration_sec=10.0
            )

        self.image_pub.publish(img_msg)
        self.info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaCameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
