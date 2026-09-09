from __future__ import annotations

import time
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
from rclpy.time import Time as RosTime
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .numpy_actor import NumpyMLPActor
from .observation import (
    DEFAULT_DUAL_ARM_JOINT_POSITION,
    DUAL_ARM_JOINT_NAMES,
    build_reach_observation,
)
from .reach_logger import ReachRunLogger, rotate_vector


LOWER_LIMITS = np.tile(
    np.asarray([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]),
    2,
)
UPPER_LIMITS = np.tile(
    np.asarray([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]),
    2,
)


class PPOReachPolicyNode(Node):
    """Deploy the 58->14 dual_fr3_reach actor at 20 Hz."""

    def __init__(self):
        super().__init__("ppo_reach_policy_node")

        share = Path(get_package_share_directory("fr3_husky_nn_policy"))
        default_model = str(share / "models" / "dual_fr3_reach_actor.npz")

        self.declare_parameter("model_path", default_model)
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("target_pose_topic", "/reach_target_pose")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("command_topic", "/policy_joint_command")
        self.declare_parameter("control_action", "/fr3_policy_control")
        self.declare_parameter("policy_rate_hz", 20.0)
        self.declare_parameter("joint_state_timeout_s", 0.15)
        self.declare_parameter("command_timeout_s", 0.15)
        self.declare_parameter("max_duration_s", 0.0)
        self.declare_parameter("action_clip", 1.0)
        self.declare_parameter("max_policy_target_delta_rad", 0.11)
        self.declare_parameter("max_actuator_step_rad", 0.001)
        self.declare_parameter("joint_velocity_scale", 0.10)
        self.declare_parameter("joint_acceleration_scale", 0.20)
        # Kept in the action goal for compatibility. The v2 controller updates
        # the relative target and effort every 1 kHz controller cycle.
        self.declare_parameter("relative_target_refresh_hz", 1000.0)
        self.declare_parameter("joint_limit_margin_rad", 0.02)
        self.declare_parameter("ready_tolerance_rad", 0.20)
        self.declare_parameter("shadow_mode", True)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("target_position", [0.50, 0.0, 0.20])
        self.declare_parameter("target_min", [0.40, -0.10, 0.10])
        self.declare_parameter("target_max", [0.60, 0.10, 0.35])
        self.declare_parameter("log_root", "")
        self.declare_parameter("log_task_name", "dual_fr3_reach")
        self.declare_parameter("left_eef_frame", "left_fr3_link7")
        self.declare_parameter("right_eef_frame", "right_fr3_link7")
        self.declare_parameter("eef_offset_xyz", [0.0, 0.0, 0.132])
        # Kept for log metadata compatibility. TF lookup(base, link) already
        # returns robot-base-local coordinates, so no world/root offset is
        # applied to the measured EEF position.
        self.declare_parameter("robot_root_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("reach_offset_y", 0.20)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.actor = NumpyMLPActor(self.get_parameter("model_path").value)
        if self.actor.input_dim != 58 or self.actor.output_dim != 14:
            raise ValueError(
                f"dual_fr3_reach requires a 58->14 actor, got "
                f"{self.actor.input_dim}->{self.actor.output_dim}"
            )
        if self.actor.output_activation != "tanh":
            raise ValueError(
                "dual_fr3_reach Sim2Real requires a tanh-bounded actor; "
                f"model declares output_activation={self.actor.output_activation!r}"
            )

        self.target_position = self._vector_parameter("target_position")
        self.target_min = self._vector_parameter("target_min")
        self.target_max = self._vector_parameter("target_max")
        self.left_eef_frame = str(self.get_parameter("left_eef_frame").value)
        self.right_eef_frame = str(self.get_parameter("right_eef_frame").value)
        self.eef_offset_xyz = self._vector_parameter("eef_offset_xyz")
        self.robot_root_offset_xyz = self._vector_parameter("robot_root_offset_xyz")
        self.reach_offset_y = float(self.get_parameter("reach_offset_y").value)
        if not np.isfinite(self.reach_offset_y) or self.reach_offset_y < 0.0:
            raise ValueError("reach_offset_y must be finite and non-negative")
        self.joint_position = np.full(14, np.nan, dtype=np.float64)
        self.joint_velocity = np.full(14, np.nan, dtype=np.float64)
        self.joint_update_ns = np.zeros(14, dtype=np.int64)
        self.previous_action = np.zeros(14, dtype=np.float32)
        self.sequence = 0
        self.goal_handle = None
        self.goal_active = False
        self.goal_pending = False
        self.auto_start_requested = False
        self.last_status_log_ns = 0
        self.run_logger: Optional[ReachRunLogger] = None
        self.logging_start_monotonic_ns = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.create_subscription(
            JointState,
            self.get_parameter("joint_states_topic").value,
            self._joint_state_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("target_pose_topic").value,
            self._target_pose_callback,
            10,
        )
        command_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.command_publisher = self.create_publisher(
            PolicyJointCommand,
            self.get_parameter("command_topic").value,
            command_qos,
        )
        self.action_client = ActionClient(
            self, RunPolicyControl, self.get_parameter("control_action").value
        )
        self.create_service(Trigger, "~/start_policy", self._start_service)
        self.create_service(Trigger, "~/stop_policy", self._stop_service)

        rate = float(self.get_parameter("policy_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("policy_rate_hz must be positive")
        self.timer = self.create_timer(1.0 / rate, self._policy_step)

        mode = "SHADOW" if self.get_parameter("shadow_mode").value else "COMMAND"
        self.get_logger().info(
            f"Loaded {self.actor.model_path} ({self.actor.input_dim}->{self.actor.output_dim}, "
            f"output={self.actor.output_activation}, source {self.actor.source_sha256[:12]}...), "
            f"mode={mode}, rate={rate:.1f} Hz"
        )
        self.get_logger().info(
            f"Reach target in '{self.base_frame}': {self.target_position.tolist()}"
        )

    def _vector_parameter(self, name: str) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain three finite values")
        return value

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _log_status(self, message: str):
        now_ns = self._now_ns()
        if now_ns - self.last_status_log_ns > 1_000_000_000:
            self.get_logger().warn(message)
            self.last_status_log_ns = now_ns

    def _joint_state_callback(self, msg: JointState):
        now_ns = self._now_ns()
        name_to_msg_index = {name: index for index, name in enumerate(msg.name)}
        for output_index, joint_name in enumerate(DUAL_ARM_JOINT_NAMES):
            msg_index = name_to_msg_index.get(joint_name)
            if msg_index is None or msg_index >= len(msg.position):
                continue
            new_position = float(msg.position[msg_index])
            if not np.isfinite(new_position):
                continue

            previous_position = self.joint_position[output_index]
            previous_time_ns = self.joint_update_ns[output_index]
            self.joint_position[output_index] = new_position
            if msg_index < len(msg.velocity) and np.isfinite(msg.velocity[msg_index]):
                self.joint_velocity[output_index] = float(msg.velocity[msg_index])
            elif np.isfinite(previous_position) and previous_time_ns > 0:
                dt = (now_ns - previous_time_ns) * 1.0e-9
                if dt > 1.0e-4:
                    self.joint_velocity[output_index] = (
                        new_position - previous_position
                    ) / dt
            self.joint_update_ns[output_index] = now_ns

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
        if not np.all(np.isfinite(target)):
            self._log_status("Ignoring non-finite reach target")
            return
        if not np.all((target >= self.target_min) & (target <= self.target_max)):
            self._log_status(
                f"Ignoring target outside training range: {target.tolist()}"
            )
            return
        self.target_position = target
        self.get_logger().info(f"New reach target: {target.tolist()}")

    def _inputs_ready(self, require_ready_pose: bool) -> tuple[bool, str]:
        if not np.all(np.isfinite(self.joint_position)):
            return False, "waiting for all 14 arm joint positions"
        if not np.all(np.isfinite(self.joint_velocity)):
            return False, "waiting for all 14 arm joint velocities"

        timeout_ns = int(
            float(self.get_parameter("joint_state_timeout_s").value) * 1.0e9
        )
        if np.any(self._now_ns() - self.joint_update_ns > timeout_ns):
            return False, "joint state is stale"
        if not np.all(
            (self.target_position >= self.target_min)
            & (self.target_position <= self.target_max)
        ):
            return False, f"target outside training range: {self.target_position}"

        ready_error = np.max(
            np.abs(
                self.joint_position
                - DEFAULT_DUAL_ARM_JOINT_POSITION.astype(np.float64)
            )
        )
        tolerance = float(self.get_parameter("ready_tolerance_rad").value)
        if require_ready_pose and ready_error > tolerance:
            return False, (
                f"arms are not at the training ready pose "
                f"(max error {ready_error:.3f} rad)"
            )
        return True, "ready"

    def _start_service(self, _request, response):
        response.success, response.message = self._request_start()
        return response

    def _request_start(self) -> tuple[bool, str]:
        if self.get_parameter("shadow_mode").value:
            return False, "shadow_mode is true; set it false before starting control"
        if self.goal_active or self.goal_pending:
            return False, "policy control is already active or pending"
        ready, message = self._inputs_ready(require_ready_pose=True)
        if not ready:
            return False, message
        if not self.action_client.server_is_ready():
            return False, "fr3_policy_control action server is not ready"

        self.previous_action.fill(0.0)
        self.sequence = 0
        goal = RunPolicyControl.Goal()
        goal.robot_name = "dual"
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
        goal.isaac_relative_control = True
        goal.relative_target_refresh_hz = float(
            self.get_parameter("relative_target_refresh_hz").value
        )
        goal.control_gripper = False

        self.goal_pending = True
        future = self.action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)
        self._start_reach_logging()
        return True, "dual-arm policy control goal requested"

    def _start_reach_logging(self):
        if self.run_logger is not None:
            return

        configured_root = str(self.get_parameter("log_root").value).strip()
        log_root = (
            Path(configured_root).expanduser()
            if configured_root
            else Path(__file__).resolve().parents[1] / "log"
        )
        try:
            self.run_logger = ReachRunLogger(
                log_root=log_root,
                task_name=str(self.get_parameter("log_task_name").value),
                base_frame=self.base_frame,
                left_eef_frame=self.left_eef_frame,
                right_eef_frame=self.right_eef_frame,
                eef_offset_xyz=self.eef_offset_xyz,
                robot_root_offset_xyz=self.robot_root_offset_xyz,
                reach_offset_y=self.reach_offset_y,
                sample_rate_hz=float(self.get_parameter("policy_rate_hz").value),
            )
            self.logging_start_monotonic_ns = time.monotonic_ns()
            self.get_logger().info(
                f"Recording dual-arm EEF trajectory to {self.run_logger.run_dir}"
            )
            self._record_eef_sample()
        except (OSError, ValueError) as error:
            self.run_logger = None
            self.get_logger().error(f"Failed to start EEF trajectory logging: {error}")

    def _record_eef_sample(self):
        if self.run_logger is None:
            return
        try:
            left_position = self._eef_position_in_base(self.left_eef_frame)
            right_position = self._eef_position_in_base(self.right_eef_frame)
            elapsed_s = (
                time.monotonic_ns() - self.logging_start_monotonic_ns
            ) * 1.0e-9
            self.run_logger.append(
                elapsed_s=elapsed_s,
                ros_time_ns=self._now_ns(),
                target_center=self.target_position,
                left_eef=left_position,
                right_eef=right_position,
            )
        except TransformException as error:
            self._log_status(
                f"Waiting for EEF transforms in '{self.base_frame}': {error}"
            )
        except (OSError, ValueError) as error:
            self._log_status(f"Skipping invalid EEF log sample: {error}")

    def _eef_position_in_base(self, source_frame: str) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(
            self.base_frame, source_frame, RosTime()
        ).transform
        translation = np.asarray(
            [transform.translation.x, transform.translation.y, transform.translation.z],
            dtype=np.float64,
        )
        quaternion_xyzw = np.asarray(
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ],
            dtype=np.float64,
        )
        # lookup_transform(base, link) is already expressed in `base_frame`.
        # Subtracting Isaac's world-space spawn height here would shift only
        # the diagnostic trajectory by 0.405 m and make it disagree with the
        # base-frame target used by the policy.
        return translation + rotate_vector(quaternion_xyzw, self.eef_offset_xyz)

    def finalize_reach_log(self):
        if self.run_logger is None:
            return
        try:
            run_dir, plot_paths = self.run_logger.finalize()
            self.get_logger().info(
                f"Saved {self.run_logger.sample_count} EEF samples and "
                f"{len(plot_paths)} plots to {run_dir}"
            )
        except Exception as error:  # Keep shutdown progressing if plotting fails.
            self.get_logger().error(f"Failed to finalize EEF trajectory plots: {error}")

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
        self.get_logger().info("dual-arm policy control goal accepted")

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

    def _policy_step(self):
        ready, message = self._inputs_ready(require_ready_pose=False)
        if not ready:
            if self.goal_active:
                self._cancel_for_fault(message)
            else:
                self._log_status(message)
            return

        self._record_eef_sample()

        if (
            self.get_parameter("auto_start").value
            and not self.get_parameter("shadow_mode").value
            and not self.auto_start_requested
            and not self.goal_active
            and not self.goal_pending
        ):
            start_ready, _ = self._inputs_ready(require_ready_pose=True)
            if start_ready and self.action_client.server_is_ready():
                self.auto_start_requested = True
                success, start_message = self._request_start()
                if not success:
                    self._log_status(start_message)

        try:
            observation = build_reach_observation(
                self.joint_position,
                self.joint_velocity,
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

        action_clip = float(self.get_parameter("action_clip").value)
        if not np.isfinite(action_clip) or action_clip <= 0.0:
            self._cancel_for_fault("action_clip must be finite and positive")
            return
        clipped_action = np.clip(action, -action_clip, action_clip)
        delta = 0.1 * clipped_action.astype(np.float64)
        max_policy_delta = float(
            self.get_parameter("max_policy_target_delta_rad").value
        )
        if np.max(np.abs(delta)) > max_policy_delta:
            if self.goal_active:
                self._cancel_for_fault(
                    f"Policy output exceeds max target delta: "
                    f"{np.max(np.abs(delta)):.3f} rad"
                )
            return

        target = self.joint_position + delta

        # Keep the learned policy goal/action semantics, but project the
        # real-robot reference into a small safe band at the joint limits.
        # The controller applies the same projection at every 1 kHz update,
        # so this does not terminate the policy when a joint reaches a limit.
        margin = float(self.get_parameter("joint_limit_margin_rad").value)
        safe_lower = LOWER_LIMITS + margin
        safe_upper = UPPER_LIMITS - margin
        if (not np.isfinite(margin)) or np.any(safe_lower >= safe_upper):
            self._cancel_for_fault("joint_limit_margin_rad leaves no safe range")
            return
        target = np.clip(target, safe_lower, safe_upper)
        delta = target - self.joint_position

        if self.get_parameter("shadow_mode").value:
            # Shadow inference deliberately advances the previous-action term
            # without publishing commands.
            self.previous_action = clipped_action.copy()
            return
        if not self.goal_active:
            # In command mode, previous_action means the last action actually
            # applied in Isaac Lab. Keep it at the reset value while waiting
            # for an explicit start_policy request and action-goal acceptance.
            return

        self.previous_action = clipped_action.copy()

        elapsed_s = (time.monotonic_ns() - self.logging_start_monotonic_ns) * 1.0e-9
        if self.run_logger is not None:
            self.run_logger.append_policy_trace(
                elapsed_s, self._now_ns(), clipped_action,
                self.joint_position, self.joint_velocity, target)

        self.sequence += 1
        command = PolicyJointCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self.base_frame
        command.sequence = self.sequence
        command.robot_name = "dual"
        # Hold this processed offset for one 20 Hz policy step. The controller
        # adds it to the latest measured position at each 1 kHz PD update,
        # matching Isaac Lab's RelativeJointPositionAction.apply_actions().
        command.target_positions = delta.tolist()
        command.relative_position_offsets = True
        command.gripper_action = 0.0
        self.command_publisher.publish(command)


def main(args: Optional[list[str]] = None):
    rclpy.init(args=args)
    node = PPOReachPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.goal_handle is not None:
            node.goal_handle.cancel_goal_async()
        node.finalize_reach_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
