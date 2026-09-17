from __future__ import annotations

"""FR3 dual-arm consistency probe.

Drives the *same* controller path that reach v2 uses
(``RunPolicyControl`` with ``isaac_relative_control=True``), but replaces the
neural-network action with a deterministic per-joint excitation profile
(step / ramp / pulse).  It logs the high-rate joint response so a MuJoCo /
Isaac Lab rollout with the identical profile can be overlaid for sim-to-real
system identification.

Two run modes:
    * single joint  (``sweep_mode:=false``): excite ``target_joint`` once.
    * automatic sweep (``sweep_mode:=true``): iterate over ``sweep_joints``
      sequentially -- settle, excite, coast, then advance to the next joint.
      Only one joint moves at any instant (single-joint isolation), and each
      joint gets its own CSV, mirroring the Hyundai ``real_sim_sync_experi``
      per-joint files.

For repeatable hardware identification, ``return_to_start_between_trials``
captures the 14-joint start configuration (or uses explicit
``home_joint_positions``), then keeps one ``RunPolicyControl`` goal active for
the entire sweep.  Every trial runs
``q0 hold -> excite -> q0 return -> measured position/velocity settle`` before
the next joint starts.  This deliberately avoids handing the arm from
PolicyControl to the joint-trajectory controller between trials.

Key physics (see ``policy_control_action_server.cpp``):
    Every 1 kHz controller cycle recomputes ``q_target = q_measured + offset``
    so a held relative offset has the PD error term:

        tau = Kp * offset - Kd * qdot

    In legacy mode this is a torque-level excitation: a constant held offset
    behaves like ``Kp * offset - Kd * qdot``.  With
    ``return_to_start_between_trials:=true``, the probe instead refreshes
    ``offset = q_reference - q_measured`` at ``command_rate_hz``.  This creates
    an absolute q0-relative step/ramp while retaining the exact same
    PolicyControl and effort-control path.

The exported torque reconstruction is the raw PD law ``Kp*offset - Kd*qdot``
(pre magnitude-clamp / rate-limit).  The measured ``effort`` column from
``/joint_states`` is the hardware ground truth.
"""

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from fr3_husky_msgs.action import RunPolicyControl
from fr3_husky_msgs.msg import PolicyJointCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from .observation import DUAL_ARM_JOINT_NAMES

# Isaac reach v2 actuator (mirrors policy_control_action_server.hpp).
ISAAC_STIFFNESS = np.array([80.0] * 14, dtype=np.float64)
ISAAC_DAMPING = np.array([4.0] * 14, dtype=np.float64)

# These must stay aligned with PolicyControl's safe band.  Checking the full
# requested offset here is intentional: the controller projects a
# measured-relative target into this band, but an excitation that is already
# clipped is not a useful excitation input (and may keep pushing into a stop).
LOWER_LIMITS = np.tile(
    np.asarray([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]),
    2,
)
UPPER_LIMITS = np.tile(
    np.asarray([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]),
    2,
)
JOINT_LIMIT_MARGIN_RAD = 0.02

PROFILE_TYPES = ("step", "ramp", "pulse", "hold_zero")


def _is_wrist(joint_index: int) -> bool:
    """FR3 joints 5-7 (0-based within-arm index 4-6) have a 12 Nm effort limit."""
    return (joint_index % 7) >= 4


class FR3ConsistencyProbeNode(Node):
    """Deterministic joint excitation over the reach v2 effort controller."""

    def __init__(
        self,
        *,
        node_name: str = "fr3_consistency_probe_node",
        dry_run_default: bool = False,
        run_label: str = "consistency probe",
    ):
        """Create the common safe measured-relative excitation harness.

        ``FR3SysidNode`` deliberately reuses this lifecycle rather than
        reimplementing action ownership, q0 capture, limit checks, and CSV
        cleanup.  The two keyword arguments keep the public consistency-probe
        behaviour unchanged while allowing that node to use its own ROS name
        and fail-safe (dry-run) default.
        """
        super().__init__(node_name)
        self._run_label = run_label

        # --- interfaces (match ppo_reach_policy_node defaults) ---
        # The dual-arm broadcaster contains exactly the 14 controlled joints.
        # Keep this correct for `ros2 run` as well as the dedicated launch file.
        self.declare_parameter("joint_states_topic", "/dual_fr3/joint_states")
        self.declare_parameter("command_topic", "/policy_joint_command")
        self.declare_parameter("control_action", "/fr3_policy_control")

        # --- RunPolicyControl goal fields ---
        self.declare_parameter("command_timeout_s", 0.15)
        self.declare_parameter("max_duration_s", 0.0)
        self.declare_parameter("max_policy_target_delta_rad", 0.30)
        self.declare_parameter("max_actuator_step_rad", 0.001)
        self.declare_parameter("joint_velocity_scale", 0.10)
        self.declare_parameter("joint_acceleration_scale", 0.20)
        self.declare_parameter("relative_target_refresh_hz", 1000.0)

        # --- excitation profile ---
        self.declare_parameter("profile_type", "pulse")    # step|ramp|pulse|hold_zero
        self.declare_parameter("amplitude_rad", 0.05)      # main-joint offset amplitude
        self.declare_parameter("amplitude_wrist_rad", 0.03)  # joints 5-7 (12 Nm limit)
        self.declare_parameter("t_start_s", 1.0)           # settle before each excitation
        self.declare_parameter("ramp_time_s", 1.0)         # ramp duration (profile=ramp)
        self.declare_parameter("pulse_width_s", 0.5)       # pulse duration (profile=pulse)
        self.declare_parameter("hold_time_s", 2.0)         # step hold after ramp/step
        self.declare_parameter("command_rate_hz", 100.0)   # profile publish rate

        # --- run mode: single joint vs automatic sweep ---
        self.declare_parameter("target_joint", 5)          # single-joint mode target
        self.declare_parameter("sweep_mode", False)
        self.declare_parameter("sweep_joints", list(range(14)))  # indices to sweep

        # --- repeatable multi-trial hardware experiment ---
        # This stays opt-in so the original direct simulation / smoke-test
        # mode still works.  When enabled, q0 return is performed via the
        # already-active PolicyControl action; MoveToJoint is for manually
        # reaching the initial collision-free q0 before start_probe only.
        self.declare_parameter("return_to_start_between_trials", False)
        self.declare_parameter("return_to_start_after_final_trial", True)
        self.declare_parameter("home_joint_positions", [])
        self.declare_parameter("home_position_tolerance_rad", 0.01)
        self.declare_parameter("home_velocity_tolerance_rad_s", 0.02)
        self.declare_parameter("initial_settle_s", 0.75)
        self.declare_parameter("home_settle_s", 0.75)
        self.declare_parameter("home_timeout_s", 45.0)

        # --- logging / safety ---
        self.declare_parameter("log_root", "")
        self.declare_parameter("joint_state_timeout_s", 0.2)
        self.declare_parameter(
            "dry_run", dry_run_default
        )  # compute+log only, no goal/commands
        # Prevent an accidental large live excitation.  This guard does not
        # apply to dry-run so profiles can be previewed before being authorised
        # for the robot.
        self.declare_parameter("max_live_amplitude_rad", 0.05)
        self.declare_parameter("auto_start", False)

        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.max_duration_s = float(self.get_parameter("max_duration_s").value)
        self.max_delta = float(self.get_parameter("max_policy_target_delta_rad").value)
        self.profile_type = str(self.get_parameter("profile_type").value)
        self.amplitude = float(self.get_parameter("amplitude_rad").value)
        self.amplitude_wrist = float(self.get_parameter("amplitude_wrist_rad").value)
        self.t_start = float(self.get_parameter("t_start_s").value)
        ramp_time_s = float(self.get_parameter("ramp_time_s").value)
        # A zero-duration ramp is treated as an instantaneous step, but a
        # negative duration must never be silently converted into one.
        self.ramp_time = max(1.0e-3, ramp_time_s)
        self.pulse_width = float(self.get_parameter("pulse_width_s").value)
        self.hold_time = float(self.get_parameter("hold_time_s").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.return_to_start = bool(
            self.get_parameter("return_to_start_between_trials").value
        )
        self.return_after_final = bool(
            self.get_parameter("return_to_start_after_final_trial").value
        )
        self.max_live_amplitude = float(
            self.get_parameter("max_live_amplitude_rad").value
        )

        if self.profile_type not in PROFILE_TYPES:
            raise ValueError(f"profile_type must be one of {PROFILE_TYPES}")
        for name, amp in (("amplitude_rad", self.amplitude),
                          ("amplitude_wrist_rad", self.amplitude_wrist)):
            if not np.isfinite(amp) or abs(amp) > self.max_delta:
                raise ValueError(
                    f"|{name}| must be finite and <= max_policy_target_delta_rad "
                    f"({self.max_delta}); got {amp}"
                )
        for name, value in (
            ("t_start_s", self.t_start),
            ("ramp_time_s", ramp_time_s),
            ("pulse_width_s", self.pulse_width),
            ("hold_time_s", self.hold_time),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative; got {value}")
        if not np.isfinite(self.max_live_amplitude) or self.max_live_amplitude <= 0.0:
            raise ValueError("max_live_amplitude_rad must be finite and positive")
        self.home_position_tolerance = float(
            self.get_parameter("home_position_tolerance_rad").value
        )
        self.home_velocity_tolerance = float(
            self.get_parameter("home_velocity_tolerance_rad_s").value
        )
        self.initial_settle_s = float(self.get_parameter("initial_settle_s").value)
        self.home_settle_s = float(self.get_parameter("home_settle_s").value)
        self.home_timeout_s = float(self.get_parameter("home_timeout_s").value)
        for name, value in (
            ("home_position_tolerance_rad", self.home_position_tolerance),
            ("home_velocity_tolerance_rad_s", self.home_velocity_tolerance),
            ("initial_settle_s", self.initial_settle_s),
            ("home_settle_s", self.home_settle_s),
            ("home_timeout_s", self.home_timeout_s),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative; got {value}")
        if self.home_position_tolerance <= 0.0 or self.home_velocity_tolerance <= 0.0:
            raise ValueError("home position and velocity tolerances must be positive")
        if self.home_timeout_s <= 0.0:
            raise ValueError("home_timeout_s must be positive")
        if self.return_to_start and not self.return_after_final:
            self.get_logger().warn(
                "return_to_start_after_final:=false is ignored in q0-reference mode; "
                "the final trial also returns to and settles at q0"
            )
        configured_home = list(self.get_parameter("home_joint_positions").value)
        if configured_home and len(configured_home) != 14:
            raise ValueError("home_joint_positions must be empty or contain exactly 14 values")
        self.configured_home = (
            np.asarray(configured_home, dtype=np.float64) if configured_home else None
        )
        if self.configured_home is not None and not np.all(np.isfinite(self.configured_home)):
            raise ValueError("home_joint_positions must be finite")
        # Build the list of joints to excite (single-joint == sweep of length 1).
        if bool(self.get_parameter("sweep_mode").value):
            self.sweep_list = [int(j) for j in self.get_parameter("sweep_joints").value]
        else:
            self.sweep_list = [int(self.get_parameter("target_joint").value)]
        if not self.sweep_list or any(j < 0 or j >= 14 for j in self.sweep_list):
            raise ValueError("sweep joints must be non-empty and each in [0, 13]")
        if not self.dry_run and any(
            abs(self._amplitude_for(j)) > self.max_live_amplitude
            for j in self.sweep_list
        ):
            raise ValueError(
                "live amplitude exceeds max_live_amplitude_rad; use dry_run to "
                "preview it or explicitly raise the live safety limit"
            )

        self.joint_position = np.full(14, np.nan, dtype=np.float64)
        self.joint_velocity = np.full(14, np.nan, dtype=np.float64)
        self.joint_effort = np.full(14, np.nan, dtype=np.float64)
        self.joint_update_ns = np.zeros(14, dtype=np.int64)

        self.sequence = 0
        self.goal_handle = None
        self.goal_active = False
        self.goal_pending = False
        self.cancel_requested = False
        self.finished = False
        self.auto_start_requested = False

        # In q0-reference mode a single PolicyControl goal owns every trial.
        # There is intentionally no inter-trial MoveToJoint/JTC handoff.
        self.initial_settle_pending = False
        self.initial_settle_request_ns = 0
        self.initial_stable_since_ns = 0
        self.return_stable_since_ns = 0
        self.return_started_ns = 0
        self.home_position = np.full(14, np.nan, dtype=np.float64)

        # Sweep state machine.
        self.sweep_idx = -1              # position in sweep_list; -1 == not started
        self.active_joint = -1           # joint currently being excited
        self.joint_start_ns = 0          # start time of the current joint's profile

        self._csv_file = None
        self._csv_writer = None
        self._sample_count = 0
        self._last_offset = np.zeros(14, dtype=np.float64)
        self._last_reference = np.full(14, np.nan, dtype=np.float64)

        self.create_subscription(
            JointState,
            self.get_parameter("joint_states_topic").value,
            self._joint_state_callback,
            50,
        )
        command_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.command_publisher = self.create_publisher(
            PolicyJointCommand, self.get_parameter("command_topic").value, command_qos
        )
        self.action_client = ActionClient(
            self, RunPolicyControl, self.get_parameter("control_action").value
        )
        self.create_service(Trigger, "~/start_probe", self._start_service)
        self.create_service(Trigger, "~/stop_probe", self._stop_service)

        rate = float(self.get_parameter("command_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("command_rate_hz must be positive")
        self.command_period_s = 1.0 / rate
        self.timer = self.create_timer(self.command_period_s, self._probe_step)

        joint_names = [DUAL_ARM_JOINT_NAMES[j] for j in self.sweep_list]
        self.get_logger().info(
            f"{self._run_label}: profile={self.profile_type}, rate={rate:.1f} Hz, "
            f"dry_run={self.dry_run}, "
            f"{'SWEEP' if len(self.sweep_list) > 1 else 'SINGLE'} over {joint_names}; "
            f"return_to_start={self.return_to_start}"
        )

    # --------------------------------------------------------------- helpers
    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _amplitude_for(self, joint_index: int) -> float:
        return self.amplitude_wrist if _is_wrist(joint_index) else self.amplitude

    # ------------------------------------------------------------------ state
    def _joint_state_callback(self, msg: JointState):
        now_ns = self._now_ns()
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        for out_idx, joint_name in enumerate(DUAL_ARM_JOINT_NAMES):
            i = name_to_index.get(joint_name)
            if i is None or i >= len(msg.position):
                continue
            pos = float(msg.position[i])
            if not np.isfinite(pos):
                continue
            previous_position = self.joint_position[out_idx]
            previous_time_ns = self.joint_update_ns[out_idx]
            self.joint_position[out_idx] = pos
            if i < len(msg.velocity) and np.isfinite(msg.velocity[i]):
                self.joint_velocity[out_idx] = float(msg.velocity[i])
            # The dual broadcaster normally supplies qdot.  Estimate it when
            # a hardware bridge omits velocity, matching the reach node, so a
            # valid 14-joint state stream is not rejected solely for that.
            elif np.isfinite(previous_position) and previous_time_ns > 0:
                dt = (now_ns - previous_time_ns) * 1.0e-9
                if dt > 1.0e-4:
                    self.joint_velocity[out_idx] = (pos - previous_position) / dt
            if i < len(msg.effort) and np.isfinite(msg.effort[i]):
                self.joint_effort[out_idx] = float(msg.effort[i])
            self.joint_update_ns[out_idx] = now_ns
        # High-rate logging is driven by the incoming joint state.
        if self.goal_active and self.active_joint >= 0 and self._csv_writer is not None:
            self._log_row()

    def _inputs_ready(self) -> tuple[bool, str]:
        if not np.all(np.isfinite(self.joint_position)):
            return False, "waiting for all 14 arm joint positions"
        if not np.all(np.isfinite(self.joint_velocity)):
            return False, "waiting for all 14 arm joint velocities"
        timeout_ns = int(float(self.get_parameter("joint_state_timeout_s").value) * 1e9)
        if np.any(self._now_ns() - self.joint_update_ns > timeout_ns):
            return False, "joint state is stale"
        return True, "ready"

    def _capture_home_position(self) -> tuple[bool, str]:
        """Latch the one q0 used by every trial in this run."""
        home = (
            self.configured_home.copy()
            if self.configured_home is not None
            else self.joint_position.copy()
        )
        if not np.all(np.isfinite(home)):
            return False, "cannot capture a finite 14-joint home configuration"
        lower = LOWER_LIMITS + JOINT_LIMIT_MARGIN_RAD
        upper = UPPER_LIMITS - JOINT_LIMIT_MARGIN_RAD
        invalid = np.flatnonzero((home < lower) | (home > upper))
        if invalid.size:
            names = [DUAL_ARM_JOINT_NAMES[int(i)] for i in invalid]
            return False, f"home configuration is outside safe joint bands: {names}"
        if self.configured_home is not None:
            mismatch = np.abs(self.joint_position - home)
            if np.any(mismatch > self.home_position_tolerance):
                worst = int(np.argmax(mismatch))
                return False, (
                    "configured home_joint_positions does not match the stationary "
                    "robot; move to q0 before start_probe "
                    f"(worst {DUAL_ARM_JOINT_NAMES[worst]} error="
                    f"{mismatch[worst]:.4f} rad)"
                )
        self.home_position = home
        source = "home_joint_positions" if self.configured_home is not None else "live q0"
        self.get_logger().info(
            f"captured repeatable {self._run_label} home from {source}: "
            + np.array2string(home, precision=4, separator=", ")
        )
        return True, "home captured"

    def _home_busy(self) -> bool:
        return self.initial_settle_pending

    # ---------------------------------------------------------------- profile
    def _offset_vector(self, elapsed_s: float) -> tuple[np.ndarray, str]:
        """q0-relative excitation and phase label for the active joint."""
        offset = np.zeros(14, dtype=np.float64)
        j = self.active_joint
        if j < 0:
            return offset, "idle"
        a = self._amplitude_for(j)
        t = elapsed_s - self.t_start
        if t < 0.0:
            return offset, "q0_prehold" if self.return_to_start else "settle"
        if self.profile_type == "hold_zero":
            return offset, "q0_hold" if self.return_to_start else "hold_zero"
        if self.profile_type == "step":
            if t <= self.hold_time:
                offset[j] = a
                return offset, "step_up"
            return offset, "q0_stabilize" if self.return_to_start else "return"
        if self.profile_type == "ramp":
            if t <= self.ramp_time:
                offset[j] = a * (t / self.ramp_time)
                return offset, "ramp_up"
            if t <= self.ramp_time + self.hold_time:
                offset[j] = a
                return offset, "hold"
            if self.return_to_start and t <= 2.0 * self.ramp_time + self.hold_time:
                offset[j] = a * (
                    1.0 - (t - self.ramp_time - self.hold_time) / self.ramp_time
                )
                return offset, "ramp_back"
            return offset, "q0_stabilize" if self.return_to_start else "return"
        if self.profile_type == "pulse":
            if t <= self.pulse_width:
                offset[j] = a
                return offset, "pulse_up"
            return offset, "q0_stabilize" if self.return_to_start else "return"
        return offset, "unknown"

    def _reference_vector(self, elapsed_s: float) -> tuple[np.ndarray, str]:
        """Absolute q reference in q0 mode, or legacy relative offsets."""
        excitation, phase = self._offset_vector(elapsed_s)
        if not self.return_to_start:
            return excitation, phase
        # q0 mode deliberately holds every arm joint at its latched q0 while
        # only the active joint receives the excitation displacement.
        return self.home_position + excitation, phase

    def _profile_end_s(self) -> float:
        """First time at which the commanded reference is back at q0."""
        if self.profile_type == "step":
            return self.t_start + self.hold_time
        if self.profile_type == "ramp":
            return self.t_start + self.hold_time + (
                2.0 * self.ramp_time if self.return_to_start else self.ramp_time
            )
        if self.profile_type == "pulse":
            return self.t_start + self.pulse_width
        if self.profile_type == "hold_zero":
            return self.t_start + self.hold_time
        return self.t_start

    def _profile_done(self, elapsed_s: float) -> bool:
        profile_end_s = self._profile_end_s()
        if elapsed_s < profile_end_s:
            return False

        # Preserve the original fast, measured-relative smoke-test mode when
        # q0 return is disabled.  It has a short zero-offset coast, not an
        # absolute return command.
        if not self.return_to_start:
            if self.profile_type == "pulse":
                return elapsed_s > profile_end_s + 1.5
            if self.profile_type in ("step", "ramp"):
                return elapsed_s > profile_end_s + 1.0
            return True

        # q0 mode keeps the same PolicyControl goal and keeps sending q0 as
        # the reference until measured position *and* velocity are stable.
        now_ns = self._now_ns()
        if self.return_started_ns == 0:
            self.return_started_ns = now_ns
            self.return_stable_since_ns = 0
            self.get_logger().info(
                "profile return started; holding q0 through PolicyControl "
                "until measured position/velocity settle"
            )
        q_error = np.abs(self.joint_position - self.home_position)
        qd_abs = np.abs(self.joint_velocity)
        at_home = bool(
            np.all(q_error <= self.home_position_tolerance)
            and np.all(qd_abs <= self.home_velocity_tolerance)
        )
        if at_home:
            if self.return_stable_since_ns == 0:
                self.return_stable_since_ns = now_ns
                return False
            if (now_ns - self.return_stable_since_ns) * 1.0e-9 >= self.home_settle_s:
                return True
        else:
            self.return_stable_since_ns = 0

        if (now_ns - self.return_started_ns) * 1.0e-9 > self.home_timeout_s:
            self._fail_home_return(
                f"timed out after {self.home_timeout_s:.1f} s waiting for "
                "PolicyControl q0 return"
            )
        return False

    def _joint_is_safe_to_excite(self, joint_index: int) -> tuple[bool, str]:
        """Require the complete signed offset to remain inside the safe band."""
        q0_or_current = (
            self.home_position[joint_index]
            if self.return_to_start and np.isfinite(self.home_position[joint_index])
            else self.joint_position[joint_index]
        )
        target = q0_or_current + self._amplitude_for(joint_index)
        lower = LOWER_LIMITS[joint_index] + JOINT_LIMIT_MARGIN_RAD
        upper = UPPER_LIMITS[joint_index] - JOINT_LIMIT_MARGIN_RAD
        if lower <= target <= upper:
            return True, "safe"
        return False, (
            f"joint {joint_index} ({DUAL_ARM_JOINT_NAMES[joint_index]}) cannot apply "
            f"the requested {self._amplitude_for(joint_index):+.4f} rad offset "
            f"without entering its [{lower:.4f}, {upper:.4f}] rad safe band "
            f"(q0={q0_or_current:.4f})"
        )

    def _sweep_is_safe_to_start(self) -> tuple[bool, str]:
        if self.dry_run:
            return True, "dry_run"
        for joint_index in self.sweep_list:
            safe, message = self._joint_is_safe_to_excite(joint_index)
            if not safe:
                return False, message
        return True, "safe"

    def _warn_if_effort_is_missing(self):
        missing = [
            DUAL_ARM_JOINT_NAMES[j]
            for j in self.sweep_list
            if not np.isfinite(self.joint_effort[j])
        ]
        if missing:
            self.get_logger().warn(
                "joint_states has no finite effort for "
                f"{missing}; tau_meas_* will be NaN. The sweep still runs, but "
                "this is kinematic response logging rather than torque-validated SysID."
            )

    # ------------------------------------------------------------------- loop
    def _probe_step(self):
        ready, message = self._inputs_ready()
        if not ready:
            if self.goal_active or self._home_busy():
                self._cancel("input fault: " + message)
            return

        if self.initial_settle_pending:
            self._monitor_initial_settle()
            return

        if (
            self.get_parameter("auto_start").value
            and not self.auto_start_requested
            and not self.goal_active
            and not self.goal_pending
            and not self.finished
        ):
            self.auto_start_requested = True
            ok, msg = self._request_start()
            if not ok:
                self.get_logger().warn(f"auto_start failed: {msg}")

        if not self.goal_active or self.sweep_idx < 0:
            return

        elapsed_s = (self._now_ns() - self.joint_start_ns) * 1e-9
        reference, _phase = self._reference_vector(elapsed_s)
        if self.return_to_start:
            # PolicyControl only accepts measured-relative commands.  Refresh
            # this conversion at the profile rate so q_target tracks the
            # absolute q0-relative reference without any JTC handoff.
            offset = reference - self.joint_position
            if not np.all(np.isfinite(offset)) or np.any(np.abs(offset) > self.max_delta):
                self._fail_home_return(
                    "q0 reference differs from measured position by more than "
                    f"max_policy_target_delta_rad ({self.max_delta:.3f} rad)"
                )
                return
        else:
            offset = reference
        self._last_offset = offset
        self._last_reference = reference

        if not self.dry_run:
            self.sequence += 1
            command = PolicyJointCommand()
            command.header.stamp = self.get_clock().now().to_msg()
            command.header.frame_id = "base"
            command.sequence = self.sequence
            command.robot_name = "dual"
            command.target_positions = offset.tolist()
            command.relative_position_offsets = True
            command.gripper_action = 0.0
            self.command_publisher.publish(command)

        if self._profile_done(elapsed_s):
            self._finish_trial(self.sweep_idx + 1)

    def _finish_trial(self, next_idx: int):
        """End a trial only after its commanded q0 return has settled."""
        self._close_csv()
        self.active_joint = -1
        self._last_offset.fill(0.0)
        self._last_reference.fill(np.nan)
        self.return_stable_since_ns = 0
        self.return_started_ns = 0

        # q0-reference mode keeps the same PolicyControl goal for all trials.
        # No MoveToJoint action is sent here or between joints.
        if self.return_to_start:
            if next_idx >= len(self.sweep_list):
                self.get_logger().info("sweep complete; q0 is stable")
                self._cancel("sweep complete")
                self.finished = True
            else:
                self._advance_to_joint(next_idx)
            return

        # Legacy mode deliberately retains its original zero-offset coast.
        if next_idx >= len(self.sweep_list):
            self.get_logger().info("sweep complete; stopping")
            self._cancel("sweep complete")
            self.finished = True
        else:
            self._advance_to_joint(next_idx)

    def _monitor_initial_settle(self):
        """Capture q0 only after the operator's initial move has stopped."""
        now_ns = self._now_ns()
        if (now_ns - self.initial_settle_request_ns) * 1.0e-9 > self.home_timeout_s:
            self._fail_home_return(
                f"timed out after {self.home_timeout_s:.1f} s waiting for initial q0 stop"
            )
            return
        if not np.all(np.abs(self.joint_velocity) <= self.home_velocity_tolerance):
            self.initial_stable_since_ns = 0
            return
        if self.initial_stable_since_ns == 0:
            self.initial_stable_since_ns = now_ns
            return
        if (now_ns - self.initial_stable_since_ns) * 1.0e-9 < self.initial_settle_s:
            return

        self.initial_settle_pending = False
        ok, message = self._capture_home_position()
        if not ok:
            self._fail_home_return(message)
            return
        safe, message = self._sweep_is_safe_to_start()
        if not safe:
            self._fail_home_return(message)
            return
        self._warn_if_effort_is_missing()
        self.get_logger().info(
            f"initial q0 was stationary for {self.initial_settle_s:.2f} s; starting trial 1"
        )
        if self.dry_run:
            self.goal_active = True
            self._advance_to_joint(0)
            return
        ok, message = self._request_profile_goal(0)
        if not ok:
            self._fail_home_return(message)

    def _fail_home_return(self, message: str):
        self.get_logger().error("q0 return failed: " + message)
        self.initial_settle_pending = False
        self.return_stable_since_ns = 0
        self.return_started_ns = 0
        self.finished = True
        self._cancel("q0 return failure")

    def _advance_to_joint(self, next_idx: int):
        """Close the current joint's CSV and start the next one (or finish)."""
        self._close_csv()
        if next_idx >= len(self.sweep_list):
            self.get_logger().info("sweep complete; stopping")
            self._cancel("sweep complete")
            self.finished = True
            return
        self.sweep_idx = next_idx
        self.active_joint = self.sweep_list[next_idx]
        safe, message = self._joint_is_safe_to_excite(self.active_joint)
        if not self.dry_run and not safe:
            self.get_logger().error(message)
            self.finished = True
            self._cancel("joint entered unsafe excitation range")
            return
        self.joint_start_ns = self._now_ns()
        self._open_csv(self.active_joint)
        self.get_logger().info(
            f"[{next_idx + 1}/{len(self.sweep_list)}] exciting joint "
            f"{self.active_joint} ({DUAL_ARM_JOINT_NAMES[self.active_joint]}), "
            f"amplitude={self._amplitude_for(self.active_joint):.4f} rad"
        )

    # ----------------------------------------------------------------- action
    def _start_service(self, _request, response):
        response.success, response.message = self._request_start()
        return response

    def _request_start(self) -> tuple[bool, str]:
        if self.goal_active or self.goal_pending or self._home_busy():
            return False, "probe already active or pending"
        ready, message = self._inputs_ready()
        if not ready:
            return False, message

        self.cancel_requested = False
        self.finished = False
        self.sweep_idx = -1
        self.active_joint = -1
        self._last_offset.fill(0.0)

        if self.return_to_start:
            # Do not capture a moving q0.  The timer observes qdot at the
            # requested tolerance for initial_settle_s, then latches q0 and
            # starts the first PolicyControl profile.  The operator may use
            # MoveToJoint before this point, but the probe itself never needs
            # MoveIt or starts a trajectory action between trials.
            self.initial_settle_pending = True
            self.initial_settle_request_ns = self._now_ns()
            self.initial_stable_since_ns = 0
            return True, "waiting for the initial q0 to become stationary"

        safe, message = self._sweep_is_safe_to_start()
        if not safe:
            return False, message
        self._warn_if_effort_is_missing()

        if self.dry_run:
            # No controller goal: run the profile clock locally for logging.
            self.goal_active = True
            self._advance_to_joint(0)
            return True, "dry_run: logging profile without commanding the robot"

        self.sequence = 0
        return self._request_profile_goal(0)

    def _request_profile_goal(self, start_idx: int) -> tuple[bool, str]:
        """Acquire PolicyControl for exactly one trial (or legacy sweep segment)."""
        if self.goal_active or self.goal_pending:
            return False, "PolicyControl goal already active or pending"

        if not self.action_client.server_is_ready():
            return False, "fr3_policy_control action server is not ready"

        goal = RunPolicyControl.Goal()
        goal.robot_name = "dual"
        goal.command_timeout_s = self.command_timeout_s
        goal.max_duration_s = self.max_duration_s
        goal.max_policy_target_delta_rad = self.max_delta
        goal.max_actuator_step_rad = float(self.get_parameter("max_actuator_step_rad").value)
        goal.joint_velocity_scale = float(self.get_parameter("joint_velocity_scale").value)
        goal.joint_acceleration_scale = float(self.get_parameter("joint_acceleration_scale").value)
        goal.isaac_relative_control = True
        goal.relative_target_refresh_hz = float(
            self.get_parameter("relative_target_refresh_hz").value
        )
        goal.control_gripper = False

        self.profile_start_idx = start_idx
        self.goal_pending = True
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)
        return True, f"{self._run_label} goal requested for trial {start_idx + 1}"

    def _goal_response_callback(self, future):
        self.goal_pending = False
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("fr3_policy_control goal was rejected")
            return
        self.goal_handle = goal_handle
        if self.cancel_requested:
            # A stop request may arrive while send_goal_async is in flight.
            # Do not start a sweep after that request; cancel as soon as ROS
            # gives us the accepted handle.
            goal_handle.cancel_goal_async()
            goal_handle.get_result_async().add_done_callback(self._result_callback)
            self.get_logger().info("accepted probe goal canceled before sweep start")
            return
        self.goal_active = True
        goal_handle.get_result_async().add_done_callback(self._result_callback)
        self.get_logger().info(f"{self._run_label} goal accepted; excitation started")
        self._advance_to_joint(self.profile_start_idx)

    def _result_callback(self, future):
        self.goal_active = False
        self.goal_handle = None
        wrapped = future.result()
        if wrapped is not None:
            r = wrapped.result
            self.get_logger().info(
                f"probe controller finished: success={r.success}, message='{r.message}'"
            )
        self._close_csv()

    def _stop_service(self, _request, response):
        if not self.goal_active and not self.goal_pending and not self._home_busy():
            response.success = False
            response.message = "no active, pending, or home-return operation"
            return response
        self._cancel("stop requested")
        response.success = True
        response.message = "probe stop requested"
        return response

    def _cancel(self, message: str):
        self.get_logger().info(f"probe cancel: {message}")
        self.cancel_requested = True
        self.goal_active = False
        self.active_joint = -1
        self._last_offset.fill(0.0)
        self._last_reference.fill(np.nan)
        self.initial_settle_pending = False
        self.return_stable_since_ns = 0
        self.return_started_ns = 0
        self._close_csv()
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    # -------------------------------------------------------------------- csv
    def _open_csv(self, joint_index: int):
        if self._csv_writer is not None:
            return
        configured_root = str(self.get_parameter("log_root").value).strip()
        root = (
            Path(configured_root).expanduser()
            if configured_root
            else Path(__file__).resolve().parents[1] / "log" / "consistency"
        )
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%y%m%d_%H%M%S")
        jname = DUAL_ARM_JOINT_NAMES[joint_index]
        amp = self._amplitude_for(joint_index)
        path = root / f"{jname}_{self.profile_type}_amp{amp:.3f}_{stamp}.csv"
        self._csv_file = open(path, "w", newline="")
        header = (
            [
                "t_sec",
                "phase",
                "active_joint",
                "q_ref_active_rad",
                "q_ref_delta_from_q0_rad",
                "cmd_offset_rad",
                "tau_cmd_recon_nm",
            ]
            + [f"q_{n}" for n in DUAL_ARM_JOINT_NAMES]
            + [f"qd_{n}" for n in DUAL_ARM_JOINT_NAMES]
            + [f"tau_meas_{n}" for n in DUAL_ARM_JOINT_NAMES]
            + [f"q0_{n}" for n in DUAL_ARM_JOINT_NAMES]
        )
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(header)
        self._sample_count = 0
        self.get_logger().info(f"logging consistency-probe response to {path}")

    def _log_row(self):
        elapsed_s = (self._now_ns() - self.joint_start_ns) * 1e-9
        reference, phase = self._reference_vector(elapsed_s)
        j = self.active_joint
        if self.dry_run and self.return_to_start:
            cmd_offset = float(reference[j] - self.joint_position[j])
        elif self.dry_run:
            cmd_offset = float(reference[j])
        else:
            cmd_offset = float(self._last_offset[j])
        if self.return_to_start:
            q_ref_active = float(reference[j])
            q_ref_delta = float(reference[j] - self.home_position[j])
        else:
            q_ref_active = float(self.joint_position[j] + reference[j])
            q_ref_delta = float(reference[j])
        qd_j = self.joint_velocity[j]
        tau_recon = ISAAC_STIFFNESS[j] * cmd_offset - ISAAC_DAMPING[j] * (
            qd_j if np.isfinite(qd_j) else 0.0
        )
        row = (
            [
                f"{elapsed_s:.6f}",
                phase,
                j,
                f"{q_ref_active:.6f}",
                f"{q_ref_delta:.6f}",
                f"{cmd_offset:.6f}",
                f"{tau_recon:.6f}",
            ]
            + [f"{v:.6f}" for v in self.joint_position]
            + [f"{v:.6f}" for v in self.joint_velocity]
            + [f"{v:.6f}" for v in self.joint_effort]
            + [f"{v:.6f}" for v in self.home_position]
        )
        self._csv_writer.writerow(row)
        self._sample_count += 1

    def _close_csv(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self.get_logger().info(f"closed CSV ({self._sample_count} rows)")
        self._csv_file = None
        self._csv_writer = None


def main(args: Optional[list[str]] = None):
    rclpy.init(args=args)
    node = FR3ConsistencyProbeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.goal_handle is not None:
            node.goal_handle.cancel_goal_async()
        node._close_csv()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
