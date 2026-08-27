#!/usr/bin/env python3

import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


class TFPoseRecorder(Node):
    def __init__(
        self,
        source_frame: str = 'base_link',
        target_frame: str = 'right_fr3_hand_tcp',
        topic_name: str = '/right_fr3_hand_tcp_pose',
        rate_hz: float = 30.0,
    ) -> None:
        super().__init__('tf_pose_recorder')

        self._source_frame = source_frame
        self._target_frame = target_frame
        self._publisher = self.create_publisher(PoseStamped, topic_name, 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)
        self._lookup_timeout = Duration(seconds=0.1)
        self._warned_once = False

        period = 1.0 / rate_hz
        self._timer = self.create_timer(period, self._publish_pose)

        self.get_logger().info(
            f'Publishing {self._source_frame} -> {self._target_frame} pose to '
            f'{topic_name} at {rate_hz:.1f} Hz'
        )

    def _publish_pose(self) -> None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._source_frame,
                self._target_frame,
                rclpy.time.Time(),
                timeout=self._lookup_timeout,
            )
        except TransformException as exc:
            if not self._warned_once:
                self.get_logger().warn(
                    f'Waiting for transform {self._source_frame} -> {self._target_frame}: {exc}'
                )
                self._warned_once = True
            return

        self._warned_once = False

        msg = PoseStamped()
        msg.header.stamp = transform.header.stamp
        msg.header.frame_id = self._source_frame
        msg.pose.position.x = transform.transform.translation.x
        msg.pose.position.y = transform.transform.translation.y
        msg.pose.position.z = transform.transform.translation.z
        msg.pose.orientation = transform.transform.rotation

        self._publisher.publish(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Publish a TF transform as geometry_msgs/PoseStamped for ros2 bag recording.'
    )
    parser.add_argument('--source-frame', default='base_link', help='Parent frame. Default: base_link')
    parser.add_argument('--target-frame', default='right_fr3_hand_tcp', help='Child frame. Default: right_fr3_hand_tcp')
    parser.add_argument('--topic', default='/right_fr3_hand_tcp_pose', help='Output PoseStamped topic')
    parser.add_argument('--rate', type=float, default=30.0, help='Publish rate in Hz')
    return parser.parse_args()


def main(args=None):
    del args
    cli_args = parse_args()

    if cli_args.rate <= 0.0:
        raise SystemExit('--rate must be > 0')

    rclpy.init()
    node = TFPoseRecorder(
        source_frame=cli_args.source_frame,
        target_frame=cli_args.target_frame,
        topic_name=cli_args.topic,
        rate_hz=cli_args.rate,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
