#!/usr/bin/env python3
# Copyright (c) 2026, DYROS.
# SPDX-License-Identifier: BSD-3-Clause

"""Record a MuJoCo torque response matching the Isaac Reach-v2 experiment.

The Isaac JSON metadata is the experiment specification.  This script loads
the same initial joint positions, sample period, pulse timing, torque values,
directions, torque slew-rate, and safety displacement, then writes a CSV with
the same columns as record_reach_v2_torque_response.py.

This is a direct MuJoCo dynamics test.  It intentionally bypasses ROS 2, the
policy, and all position/torque controllers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_REFERENCE_METADATA = (
    REPOSITORY_ROOT.parent
    / "dual_fr3_lab/logs/torque_step/physx_reach_v2_torque_step.json"
)
DEFAULT_MODEL_XACRO = (
    REPOSITORY_ROOT / "fr3_husky_description/mjcf/dual_fr3.xml.xacro"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "logs/torque_step/mujoco_reach_v2_torque_step.csv"
)
SOURCE_PACKAGE_PATHS = {
    "fr3_husky_description": REPOSITORY_ROOT / "fr3_husky_description",
    "franka_description": REPOSITORY_ROOT.parent / "franka_description",
    "husky_description": REPOSITORY_ROOT.parent / "husky/husky_description",
}

EXPECTED_JOINT_ORDER = [
    *(f"left_fr3_joint{index}" for index in range(1, 8)),
    *(f"right_fr3_joint{index}" for index in range(1, 8)),
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an Isaac Reach-v2 open-loop torque experiment in MuJoCo "
            "and write a schema-compatible CSV."
        )
    )
    parser.add_argument(
        "--reference_metadata",
        "--reference-metadata",
        type=Path,
        default=DEFAULT_REFERENCE_METADATA,
        help=(
            "Isaac experiment JSON. Its initial pose and excitation settings "
            f"are reused. Default: {DEFAULT_REFERENCE_METADATA}"
        ),
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model_xacro",
        "--model-xacro",
        type=Path,
        default=DEFAULT_MODEL_XACRO,
        help=f"Dual-FR3 MJCF xacro. Default: {DEFAULT_MODEL_XACRO}",
    )
    model_group.add_argument(
        "--mjcf",
        type=Path,
        default=None,
        help="Already-expanded MJCF XML. This bypasses xacro processing.",
    )
    parser.add_argument(
        "--xacro_executable",
        "--xacro-executable",
        default="xacro",
        help="xacro executable used with --model_xacro. Default: xacro",
    )
    parser.add_argument(
        "--joint_ids",
        "--joint-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of action-order joint indices [0, 13].",
    )
    parser.add_argument(
        "--torques",
        type=float,
        nargs="+",
        default=None,
        metavar="NM",
        help=(
            "Override reference amplitudes with 1, 7, or 14 positive values. "
            "Seven values are repeated for the right arm."
        ),
    )
    parser.add_argument(
        "--directions",
        choices=("positive", "negative", "both"),
        default=None,
        help="Override the reference direction selection.",
    )
    parser.add_argument(
        "--settle_s",
        "--settle-s",
        type=float,
        default=None,
        help="Override the reference zero-torque settle duration.",
    )
    parser.add_argument(
        "--pulse_s",
        "--pulse-s",
        type=float,
        default=None,
        help="Override the reference torque-pulse duration.",
    )
    parser.add_argument(
        "--recovery_s",
        "--recovery-s",
        type=float,
        default=None,
        help="Override the reference zero-torque recovery duration.",
    )
    parser.add_argument(
        "--torque_rate_limit",
        "--torque-rate-limit",
        type=float,
        default=None,
        help="Override the reference torque slew-rate in Nm/s.",
    )
    parser.add_argument(
        "--max_displacement_rad",
        "--max-displacement-rad",
        type=float,
        default=None,
        help="Override the reference displacement safety stop.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--viz",
        choices=("none", "passive"),
        default="none",
        help="Use none for headless execution or passive for the MuJoCo viewer.",
    )
    return parser


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {result}")
    return result


def _expand_torques(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) == 1:
        result *= 14
    elif len(result) == 7:
        result *= 2
    elif len(result) != 14:
        raise ValueError(f"{name} requires exactly 1, 7, or 14 values")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be finite and positive")
    return result


def _load_experiment(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = args.reference_metadata.expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Isaac reference metadata does not exist: {metadata_path}\n"
            "Pass it explicitly with --reference_metadata."
        )
    metadata = json.loads(metadata_path.read_text())

    joint_order = list(metadata.get("joint_order", []))
    if joint_order != EXPECTED_JOINT_ORDER:
        raise ValueError(
            "Reference joint_order does not match left joint1..7 then right "
            f"joint1..7: {joint_order}"
        )

    initial_q = [float(value) for value in metadata["default_arm_joint_position_rad"]]
    if len(initial_q) != 14 or any(not math.isfinite(value) for value in initial_q):
        raise ValueError("default_arm_joint_position_rad must contain 14 finite values")

    dt = _finite_nonnegative(metadata["physics_dt_s"], "physics_dt_s")
    if dt <= 0.0:
        raise ValueError("physics_dt_s must be positive")

    selected_joint_ids = (
        list(metadata.get("selected_action_indices", range(14)))
        if args.joint_ids is None
        else list(args.joint_ids)
    )
    if (
        not selected_joint_ids
        or len(set(selected_joint_ids)) != len(selected_joint_ids)
        or any(index < 0 or index >= 14 for index in selected_joint_ids)
    ):
        raise ValueError("joint_ids must contain unique values in [0, 13]")

    reference_torques = metadata["torque_amplitude_nm_by_action_index"]
    torque_by_joint = _expand_torques(
        reference_torques if args.torques is None else args.torques,
        "torque amplitudes",
    )
    directions = metadata["directions"] if args.directions is None else args.directions
    if directions not in ("positive", "negative", "both"):
        raise ValueError(f"Unsupported directions value: {directions}")

    def reference_or_override(key: str, override: float | None) -> float:
        return _finite_nonnegative(metadata[key] if override is None else override, key)

    settle_s = reference_or_override("settle_s", args.settle_s)
    pulse_s = reference_or_override("pulse_s", args.pulse_s)
    recovery_s = reference_or_override("recovery_s", args.recovery_s)
    torque_rate_limit = reference_or_override(
        "torque_rate_limit_nm_s", args.torque_rate_limit
    )
    max_displacement = reference_or_override(
        "max_displacement_rad", args.max_displacement_rad
    )
    if pulse_s <= 0.0:
        raise ValueError("pulse_s must be positive")
    if max_displacement <= 0.0:
        raise ValueError("max_displacement_rad must be positive")

    def reference_vector(*keys: str) -> list[float]:
        key = next((candidate for candidate in keys if candidate in metadata), None)
        if key is None:
            raise ValueError(
                f"Reference metadata is missing all supported keys: {', '.join(keys)}"
            )
        values = [float(value) for value in metadata[key]]
        if len(values) != 14 or any(not math.isfinite(value) for value in values):
            raise ValueError(f"{key} must contain 14 finite values")
        return values

    return {
        "reference_path": metadata_path,
        "reference": metadata,
        "joint_order": joint_order,
        "initial_q": initial_q,
        "dt": dt,
        "selected_joint_ids": selected_joint_ids,
        "torque_by_joint": torque_by_joint,
        "directions": directions,
        "settle_s": settle_s,
        "pulse_s": pulse_s,
        "recovery_s": recovery_s,
        "torque_rate_limit": torque_rate_limit,
        "max_displacement": max_displacement,
        "armature": reference_vector("armature_kg_m2_by_joint"),
        "static_friction": reference_vector(
            "static_friction_effort_nm_by_joint",
            "static_friction_by_joint",
        ),
        "dynamic_friction": reference_vector(
            "dynamic_friction_effort_nm_by_joint",
            "dynamic_friction_by_joint",
        ),
        "viscous_friction": reference_vector(
            "viscous_friction_nm_s_per_rad_by_joint",
            "viscous_friction_by_joint",
        ),
        "effort_limit": reference_vector("effort_limit_nm_by_joint")
        if "effort_limit_nm_by_joint" in metadata
        else [87.0] * 4 + [12.0] * 3 + [87.0] * 4 + [12.0] * 3,
    }


def _expand_model_xacro(args: argparse.Namespace) -> tuple[str, str]:
    if args.mjcf is not None:
        model_path = args.mjcf.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"MJCF does not exist: {model_path}")
        return model_path.read_text(), str(model_path)

    xacro_path = args.model_xacro.expanduser().resolve()
    if not xacro_path.is_file():
        raise FileNotFoundError(f"MJCF xacro does not exist: {xacro_path}")
    command = [
        args.xacro_executable,
        str(xacro_path),
        "hand:=false",
        "mobile:=false",
        "with_realsense:=false",
        "with_azure:=false",
        "control_mode:=effort",
    ]
    try:
        # The source workspace does not need to be built merely to expand this
        # model. Supply a temporary ament index for the three source packages
        # used by $(find ...), without touching the user's install space.
        with tempfile.TemporaryDirectory(prefix="reach_v2_xacro_") as temp_dir:
            temp_prefix = Path(temp_dir)
            resource_dir = (
                temp_prefix / "share/ament_index/resource_index/packages"
            )
            resource_dir.mkdir(parents=True)
            for package_name, source_path in SOURCE_PACKAGE_PATHS.items():
                if not (source_path / "package.xml").is_file():
                    raise FileNotFoundError(
                        f"Required source package {package_name!r} not found: {source_path}"
                    )
                (resource_dir / package_name).write_text("")
                (temp_prefix / "share" / package_name).symlink_to(
                    source_path, target_is_directory=True
                )

            command_env = os.environ.copy()
            existing_prefixes = command_env.get("AMENT_PREFIX_PATH", "")
            command_env["AMENT_PREFIX_PATH"] = str(temp_prefix)
            if existing_prefixes:
                command_env["AMENT_PREFIX_PATH"] += os.pathsep + existing_prefixes
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=command_env,
            )
            expanded_xml = completed.stdout
            for package_name, source_path in SOURCE_PACKAGE_PATHS.items():
                expanded_xml = expanded_xml.replace(
                    str(temp_prefix / "share" / package_name),
                    str(source_path),
                )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Could not run {args.xacro_executable!r}. Source the ROS workspace "
            "or pass an expanded model with --mjcf."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "xacro expansion failed. Source the ROS 2 and ETRI workspaces first.\n"
            f"command: {' '.join(command)}\n{error.stderr.strip()}"
        ) from error
    return expanded_xml, f"{xacro_path} (expanded at runtime)"


def _phase(sample: int, settle_steps: int, pulse_steps: int) -> str:
    if sample < settle_steps:
        return "settle"
    if sample < settle_steps + pulse_steps:
        return "pulse"
    return "recovery"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mujoco_ids(mujoco: Any, model: Any, joint_order: Sequence[str]) -> dict[str, list[int]]:
    joint_ids: list[int] = []
    qpos_ids: list[int] = []
    dof_ids: list[int] = []
    actuator_ids: list[int] = []
    for name in joint_order:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo model is missing joint {name!r}")
        if actuator_id < 0:
            raise RuntimeError(f"MuJoCo model is missing effort actuator {name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"Expected hinge joint for {name!r}")
        joint_ids.append(int(joint_id))
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(int(actuator_id))
    return {
        "joint_ids": joint_ids,
        "qpos_ids": qpos_ids,
        "dof_ids": dof_ids,
        "actuator_ids": actuator_ids,
    }


def _configure_model(
    mujoco: Any,
    model: Any,
    experiment: dict[str, Any],
    ids: dict[str, list[int]],
) -> None:
    model.opt.timestep = experiment["dt"]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # Isaac keeps world gravity at -9.81 m/s^2 but disables gravity on every
    # robot rigid body. With no other dynamic objects, zero model gravity is
    # the equivalent MuJoCo experiment.
    model.opt.gravity[:] = 0.0

    # Copy the plant parameters measured and validated by the Isaac script.
    # MuJoCo has one load-independent dry-friction bound (frictionloss), while
    # current PhysX exposes separate static and dynamic friction efforts.  The
    # Reach v2 setup deliberately uses equal values, which permits this mapping.
    dof_ids = ids["dof_ids"]
    static_friction = np.asarray(experiment["static_friction"], dtype=np.float64)
    dynamic_friction = np.asarray(experiment["dynamic_friction"], dtype=np.float64)
    if not np.allclose(static_friction, dynamic_friction, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "MuJoCo frictionloss cannot independently represent different PhysX "
            "static/dynamic joint-friction efforts. The reference values must match: "
            f"static={static_friction.tolist()}, dynamic={dynamic_friction.tolist()}"
        )

    model.dof_armature[dof_ids] = experiment["armature"]
    model.dof_damping[dof_ids] = experiment["viscous_friction"]
    model.dof_frictionloss[dof_ids] = static_friction

    # Match the Reach v2 87/12 Nm limits at both the motor control and joint
    # actuator-force layers. Test pulses should normally remain well below them.
    effort_limit = np.asarray(experiment["effort_limit"], dtype=np.float64)
    actuator_ids = ids["actuator_ids"]
    joint_ids = ids["joint_ids"]
    model.actuator_ctrlrange[actuator_ids, 0] = -effort_limit
    model.actuator_ctrlrange[actuator_ids, 1] = effort_limit
    model.actuator_ctrllimited[actuator_ids] = True
    model.jnt_actfrcrange[joint_ids, 0] = -effort_limit
    model.jnt_actfrcrange[joint_ids, 1] = effort_limit
    model.jnt_actfrclimited[joint_ids] = True

    # Isaac ran without a ground plane. Disable all contact generation so the
    # visual floor and fixed box bases in dual_fr3.xml.xacro cannot contaminate
    # the free-space arm response.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)


def _validate_model_matches_reference(
    mujoco: Any,
    model: Any,
    experiment: dict[str, Any],
    ids: dict[str, list[int]],
) -> dict[str, list[float]]:
    """Validate the effective MuJoCo plant and direct-effort transmission."""

    dof_ids = ids["dof_ids"]
    joint_ids = ids["joint_ids"]
    actuator_ids = ids["actuator_ids"]
    expected_by_name = {
        "armature": np.asarray(experiment["armature"], dtype=np.float64),
        "dry_friction": np.asarray(experiment["static_friction"], dtype=np.float64),
        "viscous_friction": np.asarray(
            experiment["viscous_friction"], dtype=np.float64
        ),
    }
    actual_by_name = {
        "armature": np.asarray(model.dof_armature[dof_ids]),
        "dry_friction": np.asarray(model.dof_frictionloss[dof_ids]),
        "viscous_friction": np.asarray(model.dof_damping[dof_ids]),
    }
    for name, expected in expected_by_name.items():
        actual = actual_by_name[name]
        if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(
                f"MuJoCo {name} does not match the Isaac reference: "
                f"expected={expected.tolist()}, actual={actual.tolist()}"
            )

    transmission_joint_ids = np.asarray(model.actuator_trnid[actuator_ids, 0])
    if not np.array_equal(transmission_joint_ids, np.asarray(joint_ids)):
        raise RuntimeError(
            "MuJoCo effort actuators are not connected one-to-one to the expected joints"
        )
    gears = np.asarray(model.actuator_gear[actuator_ids])
    if not (
        np.allclose(gears[:, 0], 1.0, rtol=0.0, atol=1.0e-12)
        and np.allclose(gears[:, 1:], 0.0, rtol=0.0, atol=1.0e-12)
    ):
        raise RuntimeError(
            "MuJoCo effort actuators must use unit joint transmission gear; "
            f"actual={gears.tolist()}"
        )
    gains = np.asarray(model.actuator_gainprm[actuator_ids])
    if not (
        np.all(
            model.actuator_gaintype[actuator_ids]
            == mujoco.mjtGain.mjGAIN_FIXED
        )
        and np.allclose(gains[:, 0], 1.0, rtol=0.0, atol=1.0e-12)
        and np.allclose(gains[:, 1:], 0.0, rtol=0.0, atol=1.0e-12)
        and np.all(
            model.actuator_biastype[actuator_ids]
            == mujoco.mjtBias.mjBIAS_NONE
        )
    ):
        raise RuntimeError("MuJoCo actuators are not unit-gain, zero-bias motors")

    effort_limit = np.asarray(experiment["effort_limit"], dtype=np.float64)
    expected_range = np.column_stack((-effort_limit, effort_limit))
    if not np.allclose(
        model.actuator_ctrlrange[actuator_ids], expected_range, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("MuJoCo actuator ctrlrange does not match the Isaac effort limits")
    if not np.allclose(
        model.jnt_actfrcrange[joint_ids], expected_range, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError(
            "MuJoCo joint actuator-force range does not match the Isaac effort limits"
        )

    return {name: values.tolist() for name, values in actual_by_name.items()}


def _run(
    mujoco: Any,
    model: Any,
    data: Any,
    experiment: dict[str, Any],
    ids: dict[str, list[int]],
    output_path: Path,
    viewer: Any | None,
) -> tuple[int, int, bool]:
    joint_order = experiment["joint_order"]
    dt = experiment["dt"]
    settle_steps = round(experiment["settle_s"] / dt)
    pulse_steps = round(experiment["pulse_s"] / dt)
    recovery_steps = round(experiment["recovery_s"] / dt)
    total_steps = settle_steps + pulse_steps + recovery_steps
    if pulse_steps < 1 or total_steps < 1:
        raise ValueError("Experiment durations are shorter than one physics step")

    direction_signs = {
        "positive": (1.0,),
        "negative": (-1.0,),
        "both": (1.0, -1.0),
    }[experiment["directions"]]

    scalar_columns = [
        "trial",
        "sample",
        "time_s",
        "phase",
        "excited_action_index",
        "excited_joint_name",
        "direction",
        "pulse_amplitude_nm",
        "max_displacement_rad",
        "safety_stop",
    ]
    vector_columns: list[str] = []
    for prefix, unit in (
        ("q", "rad"),
        ("qd", "rad_s"),
        ("qdd_fd", "rad_s2"),
        ("tau_requested", "nm"),
        ("tau_commanded", "nm"),
        ("tau_applied", "nm"),
    ):
        vector_columns.extend(f"{prefix}_{unit}__{name}" for name in joint_order)

    qpos_ids = ids["qpos_ids"]
    dof_ids = ids["dof_ids"]
    actuator_ids = ids["actuator_ids"]
    initial_q = np.asarray(experiment["initial_q"], dtype=np.float64)

    trial_count = 0
    safety_stop_count = 0
    interrupted = False
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(scalar_columns + vector_columns)

        for action_index in experiment["selected_joint_ids"]:
            for direction in direction_signs:
                if viewer is not None and not viewer.is_running():
                    interrupted = True
                    break

                mujoco.mj_resetData(model, data)
                data.qpos[qpos_ids] = initial_q
                data.qvel[dof_ids] = 0.0
                data.ctrl[:] = 0.0
                mujoco.mj_forward(model, data)

                reset_q = np.array(data.qpos[qpos_ids], copy=True)
                previous_qd = np.array(data.qvel[dof_ids], copy=True)
                previous_command = np.zeros(14, dtype=np.float64)
                pulse_amplitude = (
                    direction * experiment["torque_by_joint"][action_index]
                )
                trial_safety_stop = False

                print(
                    f"[trial {trial_count:02d}] {joint_order[action_index]} "
                    f"torque={pulse_amplitude:+.3f} Nm"
                )

                for sample in range(total_steps):
                    phase = _phase(sample, settle_steps, pulse_steps)
                    requested = np.zeros(14, dtype=np.float64)
                    if phase == "pulse":
                        requested[action_index] = pulse_amplitude

                    rate_limit = experiment["torque_rate_limit"]
                    if rate_limit > 0.0:
                        max_step = rate_limit * dt
                        commanded = previous_command + np.clip(
                            requested - previous_command, -max_step, max_step
                        )
                    else:
                        commanded = requested
                    previous_command = np.array(commanded, copy=True)

                    data.ctrl[actuator_ids] = commanded
                    mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()

                    q = np.array(data.qpos[qpos_ids], copy=True)
                    qd = np.array(data.qvel[dof_ids], copy=True)
                    qdd_fd = (qd - previous_qd) / dt
                    previous_qd = qd
                    applied = np.array(data.qfrc_actuator[dof_ids], copy=True)
                    max_displacement = float(np.max(np.abs(q - reset_q)))
                    finite = bool(
                        np.isfinite(q).all()
                        and np.isfinite(qd).all()
                        and np.isfinite(applied).all()
                    )
                    trial_safety_stop = (
                        not finite
                        or max_displacement > experiment["max_displacement"]
                    )

                    writer.writerow(
                        [
                            trial_count,
                            sample,
                            (sample + 1) * dt,
                            phase,
                            action_index,
                            joint_order[action_index],
                            "positive" if direction > 0.0 else "negative",
                            pulse_amplitude,
                            max_displacement,
                            int(trial_safety_stop),
                            *q.tolist(),
                            *qd.tolist(),
                            *qdd_fd.tolist(),
                            *requested.tolist(),
                            *commanded.tolist(),
                            *applied.tolist(),
                        ]
                    )

                    if trial_safety_stop:
                        safety_stop_count += 1
                        print(
                            "  [safety stop] "
                            f"max displacement={max_displacement:.4f} rad, "
                            f"finite={finite}"
                        )
                        break
                    if viewer is not None and not viewer.is_running():
                        interrupted = True
                        break

                data.ctrl[actuator_ids] = 0.0
                trial_count += 1
                if interrupted:
                    break
            if interrupted:
                break

    return trial_count, safety_stop_count, interrupted


def main() -> None:
    args = _parser().parse_args()
    experiment = _load_experiment(args)

    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError(
            "The MuJoCo Python package is required. Install MuJoCo 3.x Python "
            "bindings in this environment (for example: python3 -m pip install mujoco)."
        ) from error

    xml, model_source = _expand_model_xacro(args)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ids = _mujoco_ids(mujoco, model, experiment["joint_order"])
    _configure_model(mujoco, model, experiment, ids)
    effective_plant = _validate_model_matches_reference(
        mujoco, model, experiment, ids
    )

    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"--output must end in .csv: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")

    print(f"[reference] {experiment['reference_path']}")
    print(f"[model]     {model_source}")
    print(
        "[condition] "
        f"dt={experiment['dt']:.6f} s, "
        f"initial_q={experiment['initial_q']}, "
        f"rate_limit={experiment['torque_rate_limit']:.3f} Nm/s"
    )
    print(f"[condition] armature={effective_plant['armature']}")
    print(f"[condition] dry friction={effective_plant['dry_friction']}")
    print(f"[condition] viscous friction={effective_plant['viscous_friction']}")
    print("[condition] measurement Kp/Kd=0/0; unit-gain direct-effort motors verified")

    if args.viz == "passive":
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as passive_viewer:
            results = _run(
                mujoco, model, data, experiment, ids, output_path, passive_viewer
            )
    else:
        results = _run(mujoco, model, data, experiment, ids, output_path, None)

    trial_count, safety_stop_count, interrupted = results
    reference = experiment["reference"]
    metadata = {
        "schema_version": 2,
        "simulator": "MuJoCo",
        "mujoco_version": mujoco.__version__,
        "purpose": "open-loop joint torque response for PhysX-MuJoCo comparison",
        "model": model_source,
        "reference_metadata": str(experiment["reference_path"]),
        "reference_metadata_sha256": _sha256(experiment["reference_path"]),
        "physics_dt_s": experiment["dt"],
        "physics_rate_hz": 1.0 / experiment["dt"],
        "source_world_gravity_m_s2": reference.get("world_gravity_m_s2"),
        "source_robot_disable_gravity": reference.get("robot_disable_gravity"),
        "mujoco_gravity_m_s2": model.opt.gravity.tolist(),
        "integrator": "implicitfast",
        "measurement_controller": "direct MuJoCo motor ctrl; no policy or PD controller",
        "measurement_kp": 0.0,
        "measurement_kd": 0.0,
        "effort_transmission": "unit-gain, zero-bias joint motor",
        "contacts_disabled": True,
        "torque_rate_limit_nm_s": experiment["torque_rate_limit"],
        "joint_order": experiment["joint_order"],
        "mujoco_joint_ids": ids["joint_ids"],
        "mujoco_qpos_ids": ids["qpos_ids"],
        "mujoco_dof_ids": ids["dof_ids"],
        "mujoco_actuator_ids": ids["actuator_ids"],
        "selected_action_indices": experiment["selected_joint_ids"],
        "torque_amplitude_nm_by_action_index": experiment["torque_by_joint"],
        "directions": experiment["directions"],
        "settle_s": round(experiment["settle_s"] / experiment["dt"])
        * experiment["dt"],
        "pulse_s": round(experiment["pulse_s"] / experiment["dt"])
        * experiment["dt"],
        "recovery_s": round(experiment["recovery_s"] / experiment["dt"])
        * experiment["dt"],
        "max_displacement_rad": experiment["max_displacement"],
        "default_arm_joint_position_rad": experiment["initial_q"],
        "effort_limit_nm_by_joint": experiment["effort_limit"],
        "armature_kg_m2_by_joint": effective_plant["armature"],
        "friction_mapping": (
            "Isaac static_friction == dynamic_friction -> MuJoCo frictionloss"
        ),
        "static_friction_effort_nm_by_joint": experiment["static_friction"],
        "dynamic_friction_effort_nm_by_joint": experiment["dynamic_friction"],
        "mujoco_frictionloss_nm_by_joint": effective_plant["dry_friction"],
        "viscous_friction_nm_s_per_rad_by_joint": effective_plant[
            "viscous_friction"
        ],
        "trials_completed": trial_count,
        "safety_stops": safety_stop_count,
        "interrupted": interrupted,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[done] CSV:      {output_path}")
    print(f"[done] metadata: {metadata_path}")
    print(f"[done] trials={trial_count}, safety_stops={safety_stop_count}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
