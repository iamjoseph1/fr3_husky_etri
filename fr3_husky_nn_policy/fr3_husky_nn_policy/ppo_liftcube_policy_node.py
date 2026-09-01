from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from fr3_husky_msgs.action import RunPolicyControl
from fr3_husky_msgs.msg import PolicyJointCommand
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from .numpy_actor import NumpyMLPActor
from .observation import (
    DEFAULT_RIGHT_JOINT_POSITION,
    RIGHT_JOINT_NAMES,
    build_liftcube_observation,
)


LOWER_LIMITS = np.asarray(
    [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
    dtype=np.float64,
)
UPPER_LIMITS = np.asarray(
    [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
    dtype=np.float64,
)


class PPOLiftCubePolicyNode(Node):
    """Deploy the dual_fr3_lift_v3 right-arm actor at 20 Hz."""

    def __init__(self):
        super().__init__("ppo_liftcube_policy_node")

        share = Path(get_package_share_directory("fr3_husky_nn_policy"))
        default_model = str(share / "models" / "dual_fr3_lift_v3_actor.npz")

        self.declare_parameter("model_path", default_model)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("object_pose_topic", "/object_pose")
        self.declare_parameter("target_pose_topic", "")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("robot_name", "right")
        self.declare_parameter("command_topic", "/policy_joint_command")
        self.declare_parameter("control_action", "/fr3_policy_control")
        self.declare_parameter("policy_rate_hz", 20.0)
        self.declare_parameter("joint_state_timeout_s", 0.15)
        self.declare_parameter("object_pose_timeout_s", 0.20)
        self.declare_parameter("command_timeout_s", 0.15)
        self.declare_parameter("max_duration_s", 8.0)
        self.declare_parameter("max_policy_target_delta_rad", 0.15)
        self.declare_parameter("max_actuator_step_rad", 0.001)
        self.declare_parameter("joint_velocity_scale", 0.10)
        self.declare_parameter("joint_acceleration_scale", 0.20)
        self.declare_parameter("joint_limit_margin_rad", 0.02)
        self.declare_parameter("ready_tolerance_rad", 0.20)
        self.declare_parameter("control_gripper", True)
        self.declare_parameter("gripper_hysteresis", 0.10)
        self.declare_parameter("shadow_mode", True)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("target_position", [0.45, -0.30, 0.50])
        self.declare_parameter("object_start_min", [0.30, -0.45, 0.32])
        self.declare_parameter("object_start_max", [0.60, -0.05, 0.48])
        self.declare_parameter("object_workspace_min", [0.20, -0.60, 0.25])
        self.declare_parameter("object_workspace_max", [0.70, 0.10, 0.80])
        self.declare_parameter("target_min", [0.35, -0.45, 0.45])
        self.declare_parameter("target_max", [0.55, -0.15, 0.60])

        self.base_frame = self.get_parameter("base_frame").value
        self.robot_name = self.get_parameter("robot_name").value
        if self.robot_name != "right":
            raise ValueError("dual_fr3_lift_v3 was trained for robot_name='right'")

        self.actor = NumpyMLPActor(self.get_parameter("model_path").value)
        if self.actor.input_dim != 36 or self.actor.output_dim != 8:
            raise ValueError(
                f"dual_fr3_lift_v3 requires a 36->8 actor, got "
                f"{self.actor.input_dim}->{self.actor.output_dim}"
            )

        self.target_position = self._vector_parameter("target_position")
        self.object_start_min = self._vector_parameter("object_start_min")
        self.object_start_max = self._vector_parameter("object_start_max")
        self.object_workspace_min = self._vector_parameter("object_workspace_min")
        self.object_workspace_max = self._vector_parameter("object_workspace_max")
        self.target_min = self._vector_parameter("target_min")
        self.target_max = self._vector_parameter("target_max")

        self.joint_position = np.full(9, np.nan, dtype=np.float64)
        self.joint_velocity = np.full(9, np.nan, dtype=np.float64)
        self.joint_update_ns = np.zeros(9, dtype=np.int64)
        self.object_position = np.full(3, np.nan, dtype=np.float64)
        self.object_update_ns = 0
        self.previous_action = np.zeros(8, dtype=np.float32)
        self.sequence = 0
        self.gripper_closed = False
        self.goal_handle = None
        self.goal_active = False
        self.goal_pending = False
        self.auto_start_requested = False
        self.last_status_log_ns = 0

        joint_topic = self.get_parameter("joint_states_topic").value
        object_topic = self.get_parameter("object_pose_topic").value
        target_topic = self.get_parameter("target_pose_topic").value
        command_topic = self.get_parameter("command_topic").value
        action_name = self.get_parameter("control_action").value

        self.create_subscription(JointState, joint_topic, self._joint_state_callback, 10)
        self.create_subscription(PoseStamped, object_topic, self._object_pose_callback, 10)
        if target_topic:
            self.create_subscription(PoseStamped, target_topic, self._target_pose_callback, 10)

        command_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.command_publisher = self.create_publisher(
            PolicyJointCommand, command_topic, command_qos
        )
        self.action_client = ActionClient(self, RunPolicyControl, action_name)
        self.create_service(Trigger, "~/start_policy", self._start_service)
        self.create_service(Trigger, "~/stop_policy", self._stop_service)

        rate = float(self.get_parameter("policy_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("policy_rate_hz must be positive")
        self.timer = self.create_timer(1.0 / rate, self._policy_step)

        mode = "SHADOW" if self.get_parameter("shadow_mode").value else "COMMAND"
        self.get_logger().info(
            f"Loaded {self.actor.model_path} ({self.actor.input_dim}->{self.actor.output_dim}, "
            f"source {self.actor.source_sha256[:12]}...), mode={mode}, rate={rate:.1f} Hz"
        )

    def _vector_parameter(self, name: str) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain three finite values")
        return value

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _joint_state_callback(self, msg: JointState):
        now_ns = self._now_ns()
        position_count = len(msg.position)
        velocity_count = len(msg.velocity)
        name_to_msg_index = {name: index for index, name in enumerate(msg.name)}

        for output_index, joint_name in enumerate(RIGHT_JOINT_NAMES):
            msg_index = name_to_msg_index.get(joint_name)
            if msg_index is None or msg_index >= position_count:
                continue

            new_position = float(msg.position[msg_index])
            if not np.isfinite(new_position):
                continue

            previous_position = self.joint_position[output_index]
            previous_time_ns = self.joint_update_ns[output_index]
            self.joint_position[output_index] = new_position

            if msg_index < velocity_count and np.isfinite(msg.velocity[msg_index]):
                self.joint_velocity[output_index] = float(msg.velocity[msg_index])
            elif np.isfinite(previous_position) and previous_time_ns > 0:
                dt = (now_ns - previous_time_ns) * 1.0e-9
                if dt > 1.0e-4:
                    self.joint_velocity[output_index] = (
                        new_position - previous_position
                    ) / dt
            self.joint_update_ns[output_index] = now_ns

    def _object_pose_callback(self, msg: PoseStamped):
        if msg.header.frame_id != self.base_frame:
            self._log_status(
                f"Ignoring /object_pose frame '{msg.header.frame_id}', expected '{self.base_frame}'"
            )
            return
        position = np.asarray(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        if np.all(np.isfinite(position)):
            self.object_position = position
            self.object_update_ns = self._now_ns()

    def _target_pose_callback(self, msg: PoseStamped):
        if msg.header.frame_id != self.base_frame:
            self._log_status(
                f"Ignoring target frame '{msg.header.frame_id}', expected '{self.base_frame}'"
            )
            return
        target = np.asarray(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        if np.all(np.isfinite(target)):
            self.target_position = target

    def _log_status(self, message: str):
        now_ns = self._now_ns()
        if now_ns - self.last_status_log_ns > 1_000_000_000:
            self.get_logger().warn(message)
            self.last_status_log_ns = now_ns

    def _inputs_ready(self, require_start_region: bool) -> tuple[bool, str]:
        now_ns = self._now_ns()
        joint_timeout_ns = int(
            float(self.get_parameter("joint_state_timeout_s").value) * 1.0e9
        )
        object_timeout_ns = int(
            float(self.get_parameter("object_pose_timeout_s").value) * 1.0e9
        )

        if not np.all(np.isfinite(self.joint_position)):
            return False, "waiting for all 9 right-arm/gripper joint positions"
        if not np.all(np.isfinite(self.joint_velocity)):
            return False, "waiting for all 9 right-arm/gripper joint velocities"
        if np.any(now_ns - self.joint_update_ns > joint_timeout_ns):
            return False, "joint state is stale"
        if not np.all(np.isfinite(self.object_position)) or self.object_update_ns == 0:
            return False, "waiting for /object_pose"
        if now_ns - self.object_update_ns > object_timeout_ns:
            return False, "object pose is stale"
        if not np.all(
            (self.object_position >= self.object_workspace_min)
            & (self.object_position <= self.object_workspace_max)
        ):
            return False, f"object pose outside configured workspace: {self.object_position}"
        if require_start_region and not np.all(
            (self.object_position >= self.object_start_min)
            & (self.object_position <= self.object_start_max)
        ):
            return False, f"object pose outside policy start region: {self.object_position}"
        if not np.all(
            (self.target_position >= self.target_min)
            & (self.target_position <= self.target_max)
        ):
            return False, f"target position outside training range: {self.target_position}"

        ready_error = np.max(
            np.abs(
                self.joint_position[:7]
                - DEFAULT_RIGHT_JOINT_POSITION[:7].astype(np.float64)
            )
        )
        ready_tolerance = float(self.get_parameter("ready_tolerance_rad").value)
        if require_start_region and ready_error > ready_tolerance:
            return False, (
                f"right arm is not at the training ready pose "
                f"(max error {ready_error:.3f} rad)"
            )
        return True, "ready"

    def _start_service(self, _request, response):
        success, message = self._request_start()
        response.success = success
        response.message = message
        return response

    def _request_start(self) -> tuple[bool, str]:
        if self.get_parameter("shadow_mode").value:
            return False, "shadow_mode is true; set it false before starting control"
        if self.goal_active or self.goal_pending:
            return False, "policy control is already active or pending"

        ready, message = self._inputs_ready(require_start_region=True)
        if not ready:
            return False, message
        if not self.action_client.server_is_ready():
            return False, "fr3_policy_control action server is not ready"

        self.previous_action.fill(0.0)
        self.sequence = 0
        self.gripper_closed = False

        goal = RunPolicyControl.Goal()
        goal.robot_name = self.robot_name
        goal.command_timeout_s = float(
            self.get_parameter("command_timeout_s").value
        )
        goal.max_duration_s = float(self.get_parameter("max_duration_s").value)
        goal.max_policy_target_delta_rad = float(
            self.get_parameter("max_policy_target_delta_rad").value
        )
        goal.max_actuator_step_rad = float(
            self.get_parameter("max_actuator_step_rad").value
        )
        goal.joint_velocity_scale = float(
            self.get_parameter("joint_velocity_scale").value
        )
        goal.joint_acceleration_scale = float(
            self.get_parameter("joint_acceleration_scale").value
        )
        goal.control_gripper = bool(self.get_parameter("control_gripper").value)

        self.goal_pending = True
        future = self.action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)
        return True, "policy control goal requested"

    def _goal_response_callback(self, future):
        self.goal_pending = False
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("fr3_policy_control goal was rejected")
            return
        self.goal_handle = goal_handle
        self.goal_active = True
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)
        self.get_logger().info("fr3_policy_control goal accepted")

    def _feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        if feedback.max_joint_error > 0.12:
            self._log_status(
                f"Large policy tracking error: {feedback.max_joint_error:.3f} rad"
            )

    def _result_callback(self, future):
        self.goal_active = False
        self.goal_handle = None
        wrapped = future.result()
        if wrapped is None:
            self.get_logger().error("fr3_policy_control returned no result")
            return
        result = wrapped.result
        self.get_logger().info(
            f"policy control finished: success={result.success}, "
            f"message='{result.message}', last_sequence={result.last_sequence}"
        )

    def _stop_service(self, _request, response):
        if self.goal_handle is None:
            response.success = False
            response.message = "no active policy goal"
            return response
        self.goal_handle.cancel_goal_async()
        response.success = True
        response.message = "policy cancel requested"
        return response

    def _cancel_for_fault(self, message: str):
        self.get_logger().error(message)
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.goal_active = False

    def _debounced_gripper_action(self, raw_action: float) -> float:
        threshold = float(self.get_parameter("gripper_hysteresis").value)
        if raw_action < -threshold:
            self.gripper_closed = True
        elif raw_action > threshold:
            self.gripper_closed = False
        return -1.0 if self.gripper_closed else 1.0

    def _policy_step(self):
        ready, message = self._inputs_ready(require_start_region=False)
        if not ready:
            if self.goal_active:
                self._cancel_for_fault(message)
            else:
                self._log_status(message)
            return

        if (
            self.get_parameter("auto_start").value
            and not self.get_parameter("shadow_mode").value
            and not self.auto_start_requested
            and not self.goal_active
            and not self.goal_pending
        ):
            start_ready, _ = self._inputs_ready(require_start_region=True)
            if start_ready and self.action_client.server_is_ready():
                self.auto_start_requested = True
                success, start_message = self._request_start()
                if not success:
                    self._log_status(start_message)

        try:
            observation = build_liftcube_observation(
                self.joint_position,
                self.joint_velocity,
                self.object_position,
                self.target_position,
                self.previous_action,
            )
            action = self.actor(observation)
        except (ValueError, FloatingPointError) as error:
            if self.goal_active:
                self._cancel_for_fault(f"Policy inference failed: {error}")
            else:
                self._log_status(f"Policy inference failed: {error}")
            return

        delta = 0.1 * action[:7].astype(np.float64)
        max_policy_delta = float(
            self.get_parameter("max_policy_target_delta_rad").value
        )
        if np.max(np.abs(delta)) > max_policy_delta:
            if self.goal_active:
                self._cancel_for_fault(
                    f"Policy output exceeds max target step: {np.max(np.abs(delta)):.3f} rad"
                )
            return

        target = self.joint_position[:7] + delta
        margin = float(self.get_parameter("joint_limit_margin_rad").value)
        if not np.all(
            (target >= LOWER_LIMITS + margin) & (target <= UPPER_LIMITS - margin)
        ):
            if self.goal_active:
                self._cancel_for_fault(f"Policy target violates joint limits: {target}")
            return

        self.previous_action = action.copy()
        if self.get_parameter("shadow_mode").value or not self.goal_active:
            return

        self.sequence += 1
        command = PolicyJointCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self.base_frame
        command.sequence = self.sequence
        command.robot_name = self.robot_name
        command.target_positions = target.tolist()
        command.gripper_action = self._debounced_gripper_action(float(action[7]))
        self.command_publisher.publish(command)


def main(args: Optional[list[str]] = None):
    rclpy.init(args=args)
    node = PPOLiftCubePolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.goal_handle is not None:
            node.goal_handle.cancel_goal_async()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
