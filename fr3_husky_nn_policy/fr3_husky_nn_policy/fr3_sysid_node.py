"""Safe rich-excitation data collection for torque-validated FR3 SysID.

This is intentionally separate from ``fr3_consistency_probe_node``.  The
consistency probe checks that two closed-loop position responses are wired
consistently; this node collects the ``(q, qdot, tau_meas)`` data needed to fit
and validate a *plant* model.  It keeps the same measured-relative PD/action
path because closed-loop tracking is the safe way to create motion on hardware;
the offline fit uses the measured torque and motion rather than the tracking
error.

Each selected joint receives two complementary excitations while every other
joint stays at one latched, stationary q0:

* a bidirectional multi-speed sweep with actual constant-velocity intervals,
  for Coulomb and viscous friction;
* Hann-windowed sines, for acceleration/reflected-inertia excitation.

The base consistency-probe class owns the safety-critical lifecycle: 14-joint
state freshness, q0 capture, joint-limit margins, one active PolicyControl
goal, q0 return/settling between trials, cancellation, and high-rate logging.
Only the profile and richer log schema live here.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy

from .fr3_consistency_probe_node import (
    ISAAC_DAMPING,
    ISAAC_STIFFNESS,
    JOINT_LIMIT_MARGIN_RAD,
    LOWER_LIMITS,
    UPPER_LIMITS,
    FR3ConsistencyProbeNode,
)
from .fr3_sysid_excitation import Trajectory, build_trajectory
from .observation import DUAL_ARM_JOINT_NAMES


def _finite_positive_list(value: object, name: str) -> list[float]:
    """Read a ROS double-array parameter with a useful configuration error."""
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty list of finite numbers")
    if np.any(values <= 0.0):
        raise ValueError(f"{name} values must all be positive")
    return [float(item) for item in values]


class FR3SysidNode(FR3ConsistencyProbeNode):
    """Collect one q0-referenced rich-excitation CSV per selected arm joint."""

    def __init__(self):
        # A physical excitation must be opt-in.  The separate launch file also
        # exposes dry_run:=true by default, but this protects ``ros2 run``.
        super().__init__(
            node_name="fr3_sysid_node",
            dry_run_default=True,
            run_label="SysID data collection",
        )

        # The two identification stages.  Both default to enabled because the
        # experiment was designed to identify friction *and* acceleration
        # effects, not merely compare closed-loop position traces.
        self.declare_parameter("do_friction_sweep", True)
        self.declare_parameter("friction_amplitude_rad", 0.15)
        self.declare_parameter("friction_speeds_rad_s", [0.05, 0.10, 0.15])
        self.declare_parameter("friction_blend_s", 0.30)
        self.declare_parameter("friction_dwell_s", 0.40)
        self.declare_parameter("do_inertia_sine", True)
        self.declare_parameter("inertia_amplitude_rad", 0.05)
        self.declare_parameter("inertia_frequencies_hz", [0.5, 1.0])
        self.declare_parameter("inertia_cycles", 3)
        self.declare_parameter("inertia_dwell_s", 0.40)
        self.declare_parameter("q0_prehold_s", 1.0)
        # Joints 5--7 have a lower effort limit.  Their profile is scaled
        # without changing the speed samples, so their data remain useful but
        # carry a lower position/acceleration excursion by default.
        self.declare_parameter("wrist_excitation_scale", 0.50)

        # Independent live guards.  They protect the *reference* trajectory;
        # joint-limit checks below protect q0 +/- full excursion for each joint.
        self.declare_parameter("max_live_excursion_rad", 0.20)
        self.declare_parameter("max_live_speed_rad_s", 0.40)
        self.declare_parameter("max_live_accel_rad_s2", 5.0)

        self.do_friction = bool(self.get_parameter("do_friction_sweep").value)
        self.do_inertia = bool(self.get_parameter("do_inertia_sine").value)
        if not self.do_friction and not self.do_inertia:
            raise ValueError("enable at least one of do_friction_sweep or do_inertia_sine")
        # SysID samples must share q0.  Measured-relative smoke-test operation
        # would make the stored reference ambiguous and is deliberately barred.
        if not self.return_to_start:
            raise ValueError(
                "fr3_sysid_node requires return_to_start_between_trials:=true"
            )

        self.friction_amplitude = self._positive_scalar("friction_amplitude_rad")
        self.friction_speeds = _finite_positive_list(
            self.get_parameter("friction_speeds_rad_s").value,
            "friction_speeds_rad_s",
        )
        self.friction_blend_s = self._positive_scalar("friction_blend_s")
        self.friction_dwell_s = self._nonnegative_scalar("friction_dwell_s")
        self.inertia_amplitude = self._positive_scalar("inertia_amplitude_rad")
        self.inertia_frequencies = _finite_positive_list(
            self.get_parameter("inertia_frequencies_hz").value,
            "inertia_frequencies_hz",
        )
        self.inertia_cycles = int(self.get_parameter("inertia_cycles").value)
        if self.inertia_cycles <= 0:
            raise ValueError("inertia_cycles must be a positive integer")
        self.inertia_dwell_s = self._nonnegative_scalar("inertia_dwell_s")
        self.q0_prehold_s = self._nonnegative_scalar("q0_prehold_s")
        self.wrist_scale = self._positive_scalar("wrist_excitation_scale")
        if self.wrist_scale > 1.0:
            raise ValueError("wrist_excitation_scale must be in (0, 1]")

        self.max_live_excursion = self._positive_scalar("max_live_excursion_rad")
        self.max_live_speed = self._positive_scalar("max_live_speed_rad_s")
        self.max_live_accel = self._positive_scalar("max_live_accel_rad_s2")

        # Cache ordinary and wrist trajectories.  ``build_trajectory`` rejects
        # a sweep with no constant-speed interval (amplitude <= speed*blend),
        # before any live command can be sent.
        scales = {1.0, self.wrist_scale}
        self._trajectories = {scale: self._build_trajectory(scale) for scale in scales}
        self._active_trajectory: Optional[Trajectory] = None
        self._validate_live_trajectory()

        self.get_logger().info(
            "SysID excitation ready: "
            f"friction={self.do_friction}, inertia={self.do_inertia}, "
            f"dry_run={self.dry_run}; normal-joint duration="
            f"{self._trajectories[1.0].duration:.1f} s"
        )

    # ---------------------------------------------------------------- config
    def _positive_scalar(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def _nonnegative_scalar(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def _build_trajectory(self, scale: float) -> Trajectory:
        return build_trajectory(
            settle_s=self.q0_prehold_s,
            friction_amplitude=self.friction_amplitude * scale,
            friction_speeds=self.friction_speeds,
            inertia_amplitude=self.inertia_amplitude * scale,
            inertia_freqs=self.inertia_frequencies,
            inertia_cycles=self.inertia_cycles,
            dwell_s=max(self.friction_dwell_s, self.inertia_dwell_s),
            friction_blend_s=self.friction_blend_s,
            friction_dwell_s=self.friction_dwell_s,
            inertia_dwell_s=self.inertia_dwell_s,
            do_friction=self.do_friction,
            do_inertia=self.do_inertia,
        )

    def _scale_for_joint(self, joint_index: int) -> float:
        return self.wrist_scale if (joint_index % 7) >= 4 else 1.0

    def _trajectory_for_joint(self, joint_index: int) -> Trajectory:
        return self._trajectories[self._scale_for_joint(joint_index)]

    def _excursion_for_joint(self, joint_index: int) -> float:
        scale = self._scale_for_joint(joint_index)
        components = []
        if self.do_friction:
            components.append(self.friction_amplitude * scale)
        if self.do_inertia:
            components.append(self.inertia_amplitude * scale)
        return max(components)

    def _validate_live_trajectory(self):
        for scale, trajectory in self._trajectories.items():
            # Position bounds are analytic: the components never overlap, and
            # each is bounded by the passed amplitude.  The remaining guards
            # use analytic upper bounds rather than a sampled preview, so a
            # high-frequency parameter change cannot slip between samples.
            excursion = max(
                self.friction_amplitude * scale if self.do_friction else 0.0,
                self.inertia_amplitude * scale if self.do_inertia else 0.0,
            )
            if not self.dry_run and excursion > self.max_live_excursion:
                raise ValueError(
                    "SysID live excursion exceeds max_live_excursion_rad "
                    f"({excursion:.3f} > {self.max_live_excursion:.3f})"
                )
            speed = trajectory.speed_upper_bound()
            accel = trajectory.accel_upper_bound()
            if not self.dry_run and speed > self.max_live_speed:
                raise ValueError(
                    "SysID reference speed exceeds max_live_speed_rad_s "
                    f"({speed:.3f} > {self.max_live_speed:.3f})"
                )
            if not self.dry_run and accel > self.max_live_accel:
                raise ValueError(
                    "SysID reference acceleration exceeds max_live_accel_rad_s2 "
                    f"({accel:.3f} > {self.max_live_accel:.3f})"
                )

    # -------------------------------------------------------------- lifecycle
    def _warn_if_effort_is_missing(self):
        missing = [
            DUAL_ARM_JOINT_NAMES[j]
            for j in self.sweep_list
            if not np.isfinite(self.joint_effort[j])
        ]
        if missing:
            self.get_logger().warn(
                "joint_states has no finite effort for "
                f"{missing}; tau_meas_* will be NaN and this run cannot be used "
                "for torque-validated SysID."
            )

    def _joint_is_safe_to_excite(self, joint_index: int) -> tuple[bool, str]:
        """Require the complete +/- q0-relative trajectory inside the safe band."""
        q0 = self.home_position[joint_index]
        if not np.isfinite(q0):
            return False, "no finite q0 has been captured"
        excursion = self._excursion_for_joint(joint_index)
        lower = LOWER_LIMITS[joint_index] + JOINT_LIMIT_MARGIN_RAD
        upper = UPPER_LIMITS[joint_index] - JOINT_LIMIT_MARGIN_RAD
        if lower <= q0 - excursion and q0 + excursion <= upper:
            return True, "safe"
        return False, (
            f"joint {joint_index} ({DUAL_ARM_JOINT_NAMES[joint_index]}) cannot run "
            f"the full +/-{excursion:.4f} rad SysID trajectory inside its "
            f"[{lower:.4f}, {upper:.4f}] rad safe band (q0={q0:.4f})"
        )

    def _offset_vector(self, elapsed_s: float) -> tuple[np.ndarray, str]:
        """Return q0-relative SysID position reference for the active joint."""
        offset = np.zeros(14, dtype=np.float64)
        if self.active_joint < 0 or self._active_trajectory is None:
            return offset, "idle"
        delta, _vel, _acc, phase, _done = self._active_trajectory.eval(elapsed_s)
        offset[self.active_joint] = delta
        return offset, phase

    def _profile_done(self, elapsed_s: float) -> bool:
        """After the trajectory, retain q0 until all 14 joints settle."""
        if self._active_trajectory is None or elapsed_s < self._active_trajectory.duration:
            return False
        now_ns = self._now_ns()
        if self.return_started_ns == 0:
            self.return_started_ns = now_ns
            self.return_stable_since_ns = 0
            self.get_logger().info("SysID trajectory ended; holding q0 until it settles")
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
                f"timed out after {self.home_timeout_s:.1f} s waiting for SysID q0 return"
            )
        return False

    def _advance_to_joint(self, next_idx: int):
        """Start the cached rich trajectory for the next selected joint."""
        self._close_csv()
        if next_idx >= len(self.sweep_list):
            self.get_logger().info("SysID sweep complete; stopping at stable q0")
            self._cancel("SysID sweep complete")
            self.finished = True
            return
        self.sweep_idx = next_idx
        self.active_joint = self.sweep_list[next_idx]
        self._active_trajectory = self._trajectory_for_joint(self.active_joint)
        safe, message = self._joint_is_safe_to_excite(self.active_joint)
        if not self.dry_run and not safe:
            self.get_logger().error(message)
            self.finished = True
            self._cancel("SysID joint entered unsafe excitation range")
            return
        self.joint_start_ns = self._now_ns()
        self._open_csv(self.active_joint)
        self.get_logger().info(
            f"[{next_idx + 1}/{len(self.sweep_list)}] SysID excitation "
            f"{DUAL_ARM_JOINT_NAMES[self.active_joint]}: "
            f"duration={self._active_trajectory.duration:.1f} s, "
            f"excursion=+/-{self._excursion_for_joint(self.active_joint):.3f} rad"
        )

    def _goal_response_callback(self, future):
        self.goal_pending = False
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("fr3_policy_control rejected the SysID goal")
            return
        self.goal_handle = goal_handle
        if self.cancel_requested:
            goal_handle.cancel_goal_async()
            goal_handle.get_result_async().add_done_callback(self._result_callback)
            self.get_logger().info("accepted SysID goal canceled before excitation start")
            return
        self.goal_active = True
        goal_handle.get_result_async().add_done_callback(self._result_callback)
        self.get_logger().info("SysID PolicyControl goal accepted; excitation started")
        self._advance_to_joint(self.profile_start_idx)

    # ---------------------------------------------------------------- logging
    def _open_csv(self, joint_index: int):
        if self._csv_writer is not None:
            return
        configured_root = str(self.get_parameter("log_root").value).strip()
        root = (
            Path(configured_root).expanduser()
            if configured_root
            else Path(__file__).resolve().parents[1] / "log" / "sysid"
        )
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%y%m%d_%H%M%S")
        joint_name = DUAL_ARM_JOINT_NAMES[joint_index]
        path = root / f"{joint_name}_sysid_rich_{stamp}.csv"
        self._csv_file = open(path, "w", newline="")
        header = (
            [
                "t_sec",
                "phase",
                "active_joint",
                "q_ref_active_rad",
                "q_ref_delta_from_q0_rad",
                "qd_ref_active_rad_s",
                "qdd_ref_active_rad_s2",
                "cmd_offset_rad",
                "tau_cmd_recon_nm",
            ]
            + [f"q_{name}" for name in DUAL_ARM_JOINT_NAMES]
            + [f"qd_{name}" for name in DUAL_ARM_JOINT_NAMES]
            + [f"tau_meas_{name}" for name in DUAL_ARM_JOINT_NAMES]
            + [f"q0_{name}" for name in DUAL_ARM_JOINT_NAMES]
        )
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(header)
        self._sample_count = 0
        self.get_logger().info(f"logging SysID data to {path}")

    def _log_row(self):
        if self._active_trajectory is None or self.active_joint < 0:
            return
        elapsed_s = (self._now_ns() - self.joint_start_ns) * 1.0e-9
        delta, ref_vel, ref_acc, phase, _done = self._active_trajectory.eval(elapsed_s)
        joint_index = self.active_joint
        q_ref = self.home_position[joint_index] + delta
        if self.dry_run:
            cmd_offset = float(q_ref - self.joint_position[joint_index])
        else:
            cmd_offset = float(self._last_offset[joint_index])
        measured_velocity = self.joint_velocity[joint_index]
        tau_recon = ISAAC_STIFFNESS[joint_index] * cmd_offset - ISAAC_DAMPING[
            joint_index
        ] * (measured_velocity if np.isfinite(measured_velocity) else 0.0)
        row = (
            [
                f"{elapsed_s:.6f}",
                phase,
                joint_index,
                f"{q_ref:.6f}",
                f"{delta:.6f}",
                f"{ref_vel:.6f}",
                f"{ref_acc:.6f}",
                f"{cmd_offset:.6f}",
                f"{tau_recon:.6f}",
            ]
            + [f"{value:.6f}" for value in self.joint_position]
            + [f"{value:.6f}" for value in self.joint_velocity]
            + [f"{value:.6f}" for value in self.joint_effort]
            + [f"{value:.6f}" for value in self.home_position]
        )
        self._csv_writer.writerow(row)
        self._sample_count += 1


def main(args: Optional[list[str]] = None):
    rclpy.init(args=args)
    node = FR3SysidNode()
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
