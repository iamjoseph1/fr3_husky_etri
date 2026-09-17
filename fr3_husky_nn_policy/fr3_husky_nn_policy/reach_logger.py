from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def rotate_vector(quaternion_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by an XYZW quaternion."""
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(quaternion_xyzw)
    if (
        quaternion_xyzw.shape != (4,)
        or vector.shape != (3,)
        or not np.all(np.isfinite(quaternion_xyzw))
        or not np.all(np.isfinite(vector))
        or norm <= 1.0e-12
    ):
        raise ValueError("invalid EEF transform quaternion or offset")
    quaternion_xyzw = quaternion_xyzw / norm
    q_xyz = quaternion_xyzw[:3]
    twice_cross = 2.0 * np.cross(q_xyz, vector)
    return vector + quaternion_xyzw[3] * twice_cross + np.cross(q_xyz, twice_cross)


class ReachRunLogger:
    """Stream a dual-arm reach run to CSV and render its terminal plots."""

    FIELDNAMES = (
        "elapsed_s",
        "ros_time_ns",
        "target_center_x",
        "target_center_y",
        "target_center_z",
        "left_target_x",
        "left_target_y",
        "left_target_z",
        "right_target_x",
        "right_target_y",
        "right_target_z",
        "left_eef_x",
        "left_eef_y",
        "left_eef_z",
        "right_eef_x",
        "right_eef_y",
        "right_eef_z",
    )
    POLICY_FIELDNAMES = (
        "elapsed_s", "ros_time_ns",
        *(f"action_{i}" for i in range(14)),
        *(f"joint_position_{i}" for i in range(14)),
        *(f"joint_velocity_{i}" for i in range(14)),
        *(f"q_target_{i}" for i in range(14)),
    )

    def __init__(
        self,
        log_root: Path,
        task_name: str,
        base_frame: str,
        left_eef_frame: str,
        right_eef_frame: str,
        eef_offset_xyz: np.ndarray,
        robot_root_offset_xyz: np.ndarray,
        reach_offset_y: float,
        sample_rate_hz: float,
        run_context: Mapping[str, Any] | None = None,
    ):
        started_at = datetime.now().astimezone()
        safe_task_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name).strip("._")
        if not safe_task_name:
            safe_task_name = "dual_fr3_reach"

        timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f_%Z")
        task_dir = Path(log_root).expanduser() / safe_task_name
        run_dir = task_dir / timestamp
        suffix = 1
        while run_dir.exists():
            run_dir = task_dir / f"{timestamp}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True)

        self.run_dir = run_dir
        self.csv_path = run_dir / "eef_trajectory.csv"
        self.policy_trace_path = run_dir / "policy_trace.csv"
        self.started_at = started_at
        self.base_frame = base_frame
        self.left_eef_frame = left_eef_frame
        self.right_eef_frame = right_eef_frame
        self.eef_offset_xyz = np.asarray(eef_offset_xyz, dtype=np.float64).copy()
        self.robot_root_offset_xyz = np.asarray(robot_root_offset_xyz, dtype=np.float64).copy()
        self.eef_coordinate_frame = "robot_root_local"
        self.reach_offset_y = float(reach_offset_y)
        self.sample_rate_hz = float(sample_rate_hz)
        self.run_context = dict(run_context or {})
        self.sample_count = 0
        self._finalized = False
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._csv_file.flush()
        self._policy_file = self.policy_trace_path.open("w", newline="", encoding="utf-8")
        self._policy_writer = csv.DictWriter(self._policy_file, fieldnames=self.POLICY_FIELDNAMES)
        self._policy_writer.writeheader()
        self._policy_file.flush()

    def append_policy_trace(self, elapsed_s, ros_time_ns, action, joint_position,
                            joint_velocity, q_target) -> None:
        if self._finalized:
            return
        arrays = [np.asarray(value, dtype=np.float64) for value in
                  (action, joint_position, joint_velocity, q_target)]
        if any(value.shape != (14,) or not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("policy trace values must be finite 14-vectors")
        row = {"elapsed_s": f"{float(elapsed_s):.9f}", "ros_time_ns": str(int(ros_time_ns))}
        for prefix, value in zip(("action", "joint_position", "joint_velocity", "q_target"), arrays):
            row.update({f"{prefix}_{i}": f"{x:.9f}" for i, x in enumerate(value)})
        self._policy_writer.writerow(row)
        self._policy_file.flush()

    def append(
        self,
        elapsed_s: float,
        ros_time_ns: int,
        target_center: np.ndarray,
        left_eef: np.ndarray,
        right_eef: np.ndarray,
    ) -> None:
        if self._finalized:
            return

        target_center = self._position(target_center, "target_center")
        left_eef = self._position(left_eef, "left_eef")
        right_eef = self._position(right_eef, "right_eef")
        left_target = target_center + np.asarray(
            [0.0, self.reach_offset_y, 0.0], dtype=np.float64
        )
        right_target = target_center - np.asarray(
            [0.0, self.reach_offset_y, 0.0], dtype=np.float64
        )

        values = np.concatenate(
            (target_center, left_target, right_target, left_eef, right_eef)
        )
        row = {
            "elapsed_s": f"{float(elapsed_s):.9f}",
            "ros_time_ns": str(int(ros_time_ns)),
        }
        row.update(
            {
                name: f"{value:.9f}"
                for name, value in zip(self.FIELDNAMES[2:], values)
            }
        )
        self._writer.writerow(row)
        self.sample_count += 1
        if self.sample_count % max(1, round(self.sample_rate_hz)) == 0:
            self._csv_file.flush()

    def finalize(self) -> tuple[Path, tuple[Path, ...]]:
        if self._finalized:
            return self.run_dir, self._plot_paths()

        self._finalized = True
        self._csv_file.flush()
        self._csv_file.close()
        self._policy_file.close()
        finished_at = datetime.now().astimezone()
        metadata = {
            "task_name": self.run_dir.parent.name,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "sample_count": self.sample_count,
            "nominal_sample_rate_hz": self.sample_rate_hz,
            "base_frame": self.base_frame,
            "left_eef_source_frame": self.left_eef_frame,
            "right_eef_source_frame": self.right_eef_frame,
            "eef_local_offset_xyz_m": self.eef_offset_xyz.tolist(),
            "robot_root_offset_xyz_m": self.robot_root_offset_xyz.tolist(),
            "per_arm_target_offset_y_m": self.reach_offset_y,
            "csv": self.csv_path.name,
            "policy_trace_csv": self.policy_trace_path.name,
            "run_context": self.run_context,
        }
        (self.run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self._render_plots()
        return self.run_dir, self._plot_paths()

    @staticmethod
    def _position(value: np.ndarray, name: str) -> np.ndarray:
        position = np.asarray(value, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{name} must contain three finite values")
        return position

    def _plot_paths(self) -> tuple[Path, ...]:
        return (
            self.run_dir / "eef_x_vs_target.png",
            self.run_dir / "eef_y_vs_target.png",
            self.run_dir / "eef_z_vs_target.png",
            self.run_dir / "eef_trajectory_3d.png",
        )

    def _load_csv(self) -> np.ndarray:
        if self.sample_count == 0:
            return np.empty(0, dtype=[(name, np.float64) for name in self.FIELDNAMES])
        return np.atleast_1d(
            np.genfromtxt(self.csv_path, delimiter=",", names=True, dtype=np.float64)
        )

    def _render_plots(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        # Ubuntu's system mpl_toolkits can be imported before a user-installed
        # Matplotlib, leaving the 3D toolkit at an incompatible version. Prefer
        # the toolkit installed beside the selected Matplotlib distribution.
        import mpl_toolkits

        matching_toolkits = (
            Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
        )
        if matching_toolkits.is_dir():
            matching_path = str(matching_toolkits)
            mpl_toolkits.__path__ = [
                matching_path,
                *(path for path in mpl_toolkits.__path__ if path != matching_path),
            ]
        from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401

        import matplotlib.pyplot as plt

        data = self._load_csv()
        axis_names = ("x", "y", "z")
        for axis_name, output_path in zip(axis_names, self._plot_paths()[:3]):
            fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
            if data.size:
                time_s = data["elapsed_s"]
                axis.plot(time_s, data[f"left_eef_{axis_name}"], label="Left EEF")
                axis.plot(
                    time_s,
                    data[f"left_target_{axis_name}"],
                    "--",
                    label="Left target",
                )
                axis.plot(time_s, data[f"right_eef_{axis_name}"], label="Right EEF")
                axis.plot(
                    time_s,
                    data[f"right_target_{axis_name}"],
                    "--",
                    label="Right target",
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No valid EEF samples were recorded",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_title(f"EEF {axis_name.upper()} position vs. reach target")
            axis.set_xlabel("Time since start_policy [s]")
            axis.set_ylabel(f"{axis_name.upper()} in {self.eef_coordinate_frame} [m]")
            axis.grid(True, alpha=0.3)
            if data.size:
                axis.legend()
            fig.savefig(output_path, dpi=160)
            plt.close(fig)

        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
        axis = fig.add_subplot(111, projection="3d")
        if data.size:
            self._plot_3d_series(axis, data, "left", "Left")
            self._plot_3d_series(axis, data, "right", "Right")
            all_xyz = np.concatenate(
                tuple(
                    np.column_stack(
                        (
                            data[f"{side}_{kind}_x"],
                            data[f"{side}_{kind}_y"],
                            data[f"{side}_{kind}_z"],
                        )
                    )
                    for side in ("left", "right")
                    for kind in ("eef", "target")
                ),
                axis=0,
            )
            center = 0.5 * (all_xyz.min(axis=0) + all_xyz.max(axis=0))
            radius = max(0.05, 0.5 * float(np.ptp(all_xyz, axis=0).max()))
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.legend()
        else:
            axis.text2D(
                0.5,
                0.5,
                "No valid EEF samples were recorded",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_title(f"Dual-arm EEF trajectories in {self.eef_coordinate_frame}")
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.set_box_aspect((1.0, 1.0, 1.0))
        fig.savefig(self._plot_paths()[3], dpi=160)
        plt.close(fig)

    @staticmethod
    def _plot_3d_series(axis, data: np.ndarray, side: str, label: str) -> None:
        eef_line = axis.plot(
            data[f"{side}_eef_x"],
            data[f"{side}_eef_y"],
            data[f"{side}_eef_z"],
            label=f"{label} EEF",
        )[0]
        axis.plot(
            data[f"{side}_target_x"],
            data[f"{side}_target_y"],
            data[f"{side}_target_z"],
            "--",
            color=eef_line.get_color(),
            label=f"{label} target",
        )
        axis.scatter(
            data[f"{side}_target_x"][-1],
            data[f"{side}_target_y"][-1],
            data[f"{side}_target_z"][-1],
            color=eef_line.get_color(),
            marker="*",
            s=100,
        )
