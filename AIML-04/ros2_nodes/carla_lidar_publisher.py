#!/usr/bin/env python3
"""
CarlaLidarPublisherNode
ROS 2 Humble Node for streaming CARLA-compatible 3D LiDAR point clouds.

Topic Published:
  - /carla/ego_vehicle/lidar/point_cloud (sensor_msgs/PointCloud2)

Data Source:
  - Controlled synthetic 3D point cloud generation.
  - Strict throughput ceiling: points_per_sec <= 200,000, bytes_per_sec <= 3.2 MB/s.
  - Explicitly identified as synthetic test data, not live CARLA simulator output.
"""

import sys
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

MAX_POINTS_PER_SEC = 200_000
MAX_BYTES_PER_SEC = MAX_POINTS_PER_SEC * 16  # 3.2 MB/s


class CarlaLidarPublisherNode(Node):
    def __init__(self):
        super().__init__('carla_lidar_publisher')

        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('frame_id', 'ego_vehicle/lidar')
        self.declare_parameter('num_points', 6000)
        self.declare_parameter('random_seed', 42)

        # Validate publish rate
        raw_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        if not math.isfinite(raw_rate) or raw_rate <= 0.0:
            self.get_logger().error(f'Invalid publish_rate_hz ({raw_rate}). Clamping to default 10.0 Hz.')
            rate_hz = 10.0
        elif raw_rate > 60.0:
            self.get_logger().warn(f'Publish rate ({raw_rate} Hz) exceeds 60.0 Hz ceiling. Clamping to 60.0 Hz.')
            rate_hz = 60.0
        else:
            rate_hz = raw_rate

        # Validate point count
        raw_points = self.get_parameter('num_points').get_parameter_value().integer_value
        if raw_points < 100:
            self.get_logger().warn(f'Point count ({raw_points}) below 100. Clamping to 100.')
            num_points = 100
        elif raw_points > 50000:
            self.get_logger().warn(f'Point count ({raw_points}) exceeds safe 50,000 ceiling. Clamping to 50,000.')
            num_points = 50000
        else:
            num_points = raw_points

        # Enforce combined throughput ceiling
        points_per_sec = rate_hz * num_points
        if points_per_sec > MAX_POINTS_PER_SEC:
            num_points = int(MAX_POINTS_PER_SEC / rate_hz)
            self.get_logger().warn(
                f'Combined throughput ({points_per_sec:.0f} pts/s) exceeds {MAX_POINTS_PER_SEC} ceiling. '
                f'Adjusted point count to {num_points} pts/scan.'
            )

        seed = self.get_parameter('random_seed').get_parameter_value().integer_value
        self.rng = np.random.default_rng(seed if seed >= 0 else None)

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.num_points = num_points

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        self.lidar_pub = self.create_publisher(
            PointCloud2,
            '/carla/ego_vehicle/lidar/point_cloud',
            sensor_qos
        )

        # Standard 4-field FLOAT32 layout (stride = 16 bytes per point)
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.timer = self.create_timer(1.0 / rate_hz, self.publish_scan)
        self.get_logger().info(
            f'CarlaLidarPublisherNode initialized at {rate_hz:.1f} Hz ({self.num_points} pts/scan, '
            f'~{self.num_points * 16 / 1024:.1f} KB/scan, throughput: {rate_hz * self.num_points * 16 / 1024:.1f} KB/s). '
            f'(NOTICE: Emitting synthetic LiDAR data for pipeline testing; not live CARLA simulation output).'
        )

    def generate_synthetic_scan(self) -> np.ndarray:
        """
        Generates synthetic 3D point cloud:
          - Road surface: z ~ -1.6m, x in [2, 45]m, y in [-8, 8]m
          - Obstacles above road: x in [8, 25]m, y in [-3, 3]m, z in [-1.2, 0.4]m (above -1.3m clearance)
          - Roadside bounds: |y| in [5, 12]m
        """
        n_road = self.num_points // 2
        n_obs = self.num_points // 4
        n_side = self.num_points - (n_road + n_obs)

        # 1. Ground plane (road surface at z approx -1.6m)
        x_road = self.rng.uniform(2.0, 45.0, n_road).astype(np.float32)
        y_road = self.rng.uniform(-8.0, 8.0, n_road).astype(np.float32)
        z_road = self.rng.normal(-1.6, 0.03, n_road).astype(np.float32)
        i_road = self.rng.uniform(0.1, 0.25, n_road).astype(np.float32)

        # 2. Obstacles ahead in driving corridor (elevated above road: z >= -1.2m)
        x_obs = self.rng.uniform(8.0, 25.0, n_obs).astype(np.float32)
        y_obs = self.rng.uniform(-3.0, 3.0, n_obs).astype(np.float32)
        z_obs = self.rng.uniform(-1.2, 0.4, n_obs).astype(np.float32)
        i_obs = self.rng.uniform(0.6, 0.9, n_obs).astype(np.float32)

        # 3. Roadside infrastructure
        x_side = self.rng.uniform(4.0, 45.0, n_side).astype(np.float32)
        side_sign = self.rng.choice([-1.0, 1.0], size=n_side).astype(np.float32)
        y_side = side_sign * self.rng.uniform(5.0, 12.0, n_side).astype(np.float32)
        z_side = self.rng.uniform(-1.6, 2.5, n_side).astype(np.float32)
        i_side = self.rng.uniform(0.3, 0.6, n_side).astype(np.float32)

        x = np.concatenate([x_road, x_obs, x_side])
        y = np.concatenate([y_road, y_obs, y_side])
        z = np.concatenate([z_road, z_obs, z_side])
        intensity = np.concatenate([i_road, i_obs, i_side])

        return np.column_stack([x, y, z, intensity]).astype(np.float32)

    def publish_scan(self):
        points = self.generate_synthetic_scan()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        cloud_msg = pc2.create_cloud(header, self.fields, points)
        self.lidar_pub.publish(cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaLidarPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
