from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def _parse_target(text: str) -> tuple[float, float, float]:
    values = text.replace(",", " ").split()
    if len(values) != 3:
        raise ValueError("enter exactly three values: x y z")
    return tuple(float(value) for value in values)


def main(args: Optional[list[str]] = None):
    rclpy.init(args=args)
    node = Node("reach_target_cli")
    node.declare_parameter("target_pose_topic", "/reach_target_pose")
    node.declare_parameter("base_frame", "base")
    publisher = node.create_publisher(
        PoseStamped, node.get_parameter("target_pose_topic").value, 10
    )
    frame = str(node.get_parameter("base_frame").value)

    print("Dual FR3 Reach target input")
    print("Training range in base frame: x=0.40..0.60, y=-0.10..0.10, z=0.10..0.35")
    print("Enter 'x y z' (meters), or q to quit.")
    try:
        while rclpy.ok():
            text = input("reach target> ").strip()
            if text.lower() in {"q", "quit", "exit"}:
                break
            try:
                x, y, z = _parse_target(text)
            except ValueError as error:
                print(f"Invalid input: {error}")
                continue
            message = PoseStamped()
            message.header.stamp = node.get_clock().now().to_msg()
            message.header.frame_id = frame
            message.pose.position.x = x
            message.pose.position.y = y
            message.pose.position.z = z
            message.pose.orientation.w = 1.0
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.1)
            print(f"Published target [{x:.3f}, {y:.3f}, {z:.3f}] in '{frame}'")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
