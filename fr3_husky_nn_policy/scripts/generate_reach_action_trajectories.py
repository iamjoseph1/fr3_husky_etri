#!/usr/bin/env python3
"""Build nominal and 100 Hz-matched-noise Reach action trajectories.

The nominal action rows are copied from the 1 kHz/matched MuJoCo policy trace.
The noise scale is estimated per joint from the steady-state difference between
the aligned 100 Hz/matched and 1 kHz/matched traces.  Per-joint mean differences
are removed so the generated perturbation is zero-mean and does not import the
100 Hz policy's bias into the nominal command.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
ABLATION_ROOT = PACKAGE_ROOT / "ablation_mujoco" / "ablation_mujoco"
DEFAULT_NOMINAL_RUN = ABLATION_ROOT / (
    "20260911_125736_563151_KST(1000hz,matched dynamics)"
)
DEFAULT_NOISE_REFERENCE_RUN = ABLATION_ROOT / (
    "20260911_125556_978004_KST(100hz,matched dynamics)"
)
DEFAULT_UNMATCHED_NOISE_REFERENCE_RUN = ABLATION_ROOT / (
    "20260911_125923_617059_KST(100hz,unmatched dynamics)"
)
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "trajectories"

sys.path.insert(0, str(PACKAGE_ROOT))
from fr3_husky_nn_policy.reach_trajectory import (  # noqa: E402
    ACTION_COLUMNS,
    NOISE_COLUMNS,
    TARGET_COLUMNS,
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nominal-run", type=Path, default=DEFAULT_NOMINAL_RUN,
        help="1 kHz/matched run containing policy_trace.csv and eef_trajectory.csv",
    )
    parser.add_argument(
        "--noise-reference-run", type=Path, default=DEFAULT_NOISE_REFERENCE_RUN,
        help="100 Hz/matched run used to estimate per-joint action perturbation scale",
    )
    parser.add_argument(
        "--unmatched-noise-reference-run",
        type=Path,
        default=DEFAULT_UNMATCHED_NOISE_REFERENCE_RUN,
        help="100 Hz/unmatched run used for the unmatched perturbation profile",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the estimated per-joint standard deviations",
    )
    parser.add_argument(
        "--steady-after-s",
        type=float,
        default=2.0,
        help="Only samples this long after each target change estimate noise scale",
    )
    return parser


def _read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def _actions(data: dict[str, np.ndarray], path: Path) -> np.ndarray:
    missing = [name for name in ACTION_COLUMNS if name not in data]
    if missing:
        raise ValueError(f"{path} is missing action columns: {', '.join(missing)}")
    values = np.column_stack([data[name] for name in ACTION_COLUMNS])
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0 + 1.0e-6):
        raise ValueError(f"{path} contains invalid raw actions")
    return values


def _target_at_policy_times(
    policy_time: np.ndarray, eef: dict[str, np.ndarray], path: Path
) -> np.ndarray:
    missing = [name for name in ("elapsed_s", *TARGET_COLUMNS) if name not in eef]
    if missing:
        raise ValueError(f"{path} is missing target columns: {', '.join(missing)}")
    eef_time = eef["elapsed_s"]
    if np.any(np.diff(eef_time) <= 0.0):
        raise ValueError(f"{path} elapsed_s must be strictly increasing")
    # The logger records EEF immediately before each policy trace row. Use the
    # most recent target rather than interpolating across a discrete goal change.
    indices = np.searchsorted(eef_time, policy_time, side="right") - 1
    indices = np.clip(indices, 0, len(eef_time) - 1)
    return np.column_stack([eef[name][indices] for name in TARGET_COLUMNS])


def _ages_since_target_change(time_s: np.ndarray, targets: np.ndarray) -> np.ndarray:
    changed = np.r_[True, np.any(np.diff(targets, axis=0) != 0.0, axis=1)]
    start_indices = np.maximum.accumulate(np.where(changed, np.arange(len(time_s)), 0))
    return time_s - time_s[start_indices]


def _estimate_noise_std(
    nominal_time: np.ndarray,
    nominal_actions: np.ndarray,
    nominal_targets: np.ndarray,
    reference_time: np.ndarray,
    reference_actions: np.ndarray,
    steady_after_s: float,
) -> tuple[np.ndarray, int]:
    nominal_relative_time = nominal_time - nominal_time[0]
    reference_relative_time = reference_time - reference_time[0]
    reference_aligned = np.column_stack(
        [
            np.interp(nominal_relative_time, reference_relative_time, reference_actions[:, i])
            for i in range(14)
        ]
    )
    ages = _ages_since_target_change(nominal_time, nominal_targets)
    steady = ages >= steady_after_s
    # In the source logger the action row can precede the EEF row carrying a
    # newly switched target by one 20 Hz sample. Exclude the sample immediately
    # before every logged target transition from the noise estimate.
    transition_next = np.r_[
        np.any(np.diff(nominal_targets, axis=0) != 0.0, axis=1), False
    ]
    steady &= ~transition_next
    if np.count_nonzero(steady) < 2:
        raise ValueError("Not enough steady-state samples to estimate noise")
    difference = reference_aligned[steady] - nominal_actions[steady]
    # Remove the per-joint policy bias. Only the residual scale is used as noise.
    std = np.std(difference - np.mean(difference, axis=0), axis=0, ddof=1)
    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError(f"Estimated invalid noise standard deviations: {std.tolist()}")
    return std, int(np.count_nonzero(steady))


def _write_trajectory(
    path: Path,
    time_s: np.ndarray,
    targets: np.ndarray,
    actions: np.ndarray,
    noise: np.ndarray,
) -> None:
    fieldnames = (
        "step", "source_elapsed_s", *TARGET_COLUMNS, *ACTION_COLUMNS, *NOISE_COLUMNS
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(len(time_s)):
            row = {
                "step": str(step),
                "source_elapsed_s": f"{time_s[step]:.9f}",
            }
            row.update(
                {name: f"{value:.9f}" for name, value in zip(TARGET_COLUMNS, targets[step])}
            )
            row.update(
                {name: f"{value:.9f}" for name, value in zip(ACTION_COLUMNS, actions[step])}
            )
            row.update(
                {name: f"{value:.9f}" for name, value in zip(NOISE_COLUMNS, noise[step])}
            )
            writer.writerow(row)


def main() -> None:
    args = _parser().parse_args()
    if not np.isfinite(args.noise_scale) or args.noise_scale < 0.0:
        raise ValueError("--noise-scale must be finite and non-negative")
    if not np.isfinite(args.steady_after_s) or args.steady_after_s < 0.0:
        raise ValueError("--steady-after-s must be finite and non-negative")

    nominal_run = args.nominal_run.expanduser().resolve()
    reference_run = args.noise_reference_run.expanduser().resolve()
    unmatched_reference_run = args.unmatched_noise_reference_run.expanduser().resolve()
    nominal_trace_path = nominal_run / "policy_trace.csv"
    nominal_eef_path = nominal_run / "eef_trajectory.csv"
    reference_trace_path = reference_run / "policy_trace.csv"
    unmatched_reference_trace_path = unmatched_reference_run / "policy_trace.csv"
    for path in (
        nominal_trace_path,
        nominal_eef_path,
        reference_trace_path,
        unmatched_reference_trace_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    nominal_trace = _read_numeric_csv(nominal_trace_path)
    nominal_eef = _read_numeric_csv(nominal_eef_path)
    reference_trace = _read_numeric_csv(reference_trace_path)
    nominal_time = nominal_trace["elapsed_s"]
    nominal_actions = _actions(nominal_trace, nominal_trace_path)
    nominal_targets = _target_at_policy_times(nominal_time, nominal_eef, nominal_eef_path)
    reference_time = reference_trace["elapsed_s"]
    reference_actions = _actions(reference_trace, reference_trace_path)
    unmatched_reference_trace = _read_numeric_csv(unmatched_reference_trace_path)
    unmatched_reference_time = unmatched_reference_trace["elapsed_s"]
    unmatched_reference_actions = _actions(
        unmatched_reference_trace, unmatched_reference_trace_path
    )

    noise_std, steady_sample_count = _estimate_noise_std(
        nominal_time,
        nominal_actions,
        nominal_targets,
        reference_time,
        reference_actions,
        args.steady_after_s,
    )
    unmatched_noise_std, unmatched_steady_sample_count = _estimate_noise_std(
        nominal_time,
        nominal_actions,
        nominal_targets,
        unmatched_reference_time,
        unmatched_reference_actions,
        args.steady_after_s,
    )
    requested_noise_std = args.noise_scale * noise_std
    unmatched_requested_noise_std = args.noise_scale * unmatched_noise_std
    rng = np.random.default_rng(args.seed)
    standardized_noise = rng.standard_normal(nominal_actions.shape)
    requested_noise = standardized_noise * requested_noise_std
    unmatched_requested_noise = standardized_noise * unmatched_requested_noise_std
    noisy_actions = np.clip(nominal_actions + requested_noise, -1.0, 1.0)
    unmatched_noisy_actions = np.clip(
        nominal_actions + unmatched_requested_noise, -1.0, 1.0
    )
    applied_noise = noisy_actions - nominal_actions
    unmatched_applied_noise = unmatched_noisy_actions - nominal_actions

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    nominal_output = output_dir / "reach_1khz_matched_nominal.csv"
    noisy_output = output_dir / "reach_1khz_matched_nominal_100hz_matched_noise.csv"
    unmatched_noisy_output = (
        output_dir / "reach_1khz_matched_nominal_100hz_unmatched_noise.csv"
    )
    metadata_output = output_dir / "reach_trajectory_metadata.json"

    zero_noise = np.zeros_like(nominal_actions)
    _write_trajectory(
        nominal_output, nominal_time, nominal_targets, nominal_actions, zero_noise
    )
    _write_trajectory(
        noisy_output, nominal_time, nominal_targets, noisy_actions, applied_noise
    )
    _write_trajectory(
        unmatched_noisy_output,
        nominal_time,
        nominal_targets,
        unmatched_noisy_actions,
        unmatched_applied_noise,
    )

    metadata = {
        "schema_version": 1,
        "policy_rate_hz": 20.0,
        "trajectory_rows": int(len(nominal_time)),
        "nominal_duration_s": float(nominal_time[-1] - nominal_time[0]),
        "nominal_source_policy_trace": str(nominal_trace_path),
        "nominal_source_policy_trace_sha256": sha256_file(nominal_trace_path),
        "nominal_source_eef_trajectory": str(nominal_eef_path),
        "nominal_source_eef_trajectory_sha256": sha256_file(nominal_eef_path),
        "noise_reference_policy_trace": str(reference_trace_path),
        "noise_reference_policy_trace_sha256": sha256_file(reference_trace_path),
        "unmatched_noise_reference_policy_trace": str(
            unmatched_reference_trace_path
        ),
        "unmatched_noise_reference_policy_trace_sha256": sha256_file(
            unmatched_reference_trace_path
        ),
        "noise_model": (
            "independent zero-mean Gaussian raw-action noise; per-joint std is "
            "the de-biased steady-state difference between time-aligned 100 Hz/matched "
            "and 1 kHz/matched policy traces"
        ),
        "unmatched_noise_model": (
            "independent zero-mean Gaussian raw-action noise; per-joint std is "
            "the de-biased steady-state difference between time-aligned "
            "100 Hz/unmatched and 1 kHz/matched policy traces"
        ),
        "steady_after_target_change_s": args.steady_after_s,
        "noise_estimation_sample_count": steady_sample_count,
        "unmatched_noise_estimation_sample_count": unmatched_steady_sample_count,
        "noise_seed": args.seed,
        "noise_scale": args.noise_scale,
        "estimated_raw_action_std_by_joint": noise_std.tolist(),
        "requested_raw_action_noise_std_by_joint": requested_noise_std.tolist(),
        "realized_raw_action_noise_std_by_joint": np.std(applied_noise, axis=0).tolist(),
        "unmatched_estimated_raw_action_std_by_joint": unmatched_noise_std.tolist(),
        "unmatched_requested_raw_action_noise_std_by_joint": (
            unmatched_requested_noise_std.tolist()
        ),
        "unmatched_realized_raw_action_noise_std_by_joint": np.std(
            unmatched_applied_noise, axis=0
        ).tolist(),
        "clipped_value_count": int(
            np.count_nonzero(noisy_actions != nominal_actions + requested_noise)
        ),
        "unmatched_clipped_value_count": int(
            np.count_nonzero(
                unmatched_noisy_actions
                != nominal_actions + unmatched_requested_noise
            )
        ),
        "nominal_trajectory": nominal_output.name,
        "nominal_trajectory_sha256": sha256_file(nominal_output),
        "noisy_trajectory": noisy_output.name,
        "noisy_trajectory_sha256": sha256_file(noisy_output),
        "unmatched_noisy_trajectory": unmatched_noisy_output.name,
        "unmatched_noisy_trajectory_sha256": sha256_file(unmatched_noisy_output),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"nominal: {nominal_output}")
    print(f"noisy:   {noisy_output}")
    print(f"unmatched noisy: {unmatched_noisy_output}")
    print(f"metadata:{metadata_output}")
    print(f"raw-action noise RMS: {np.sqrt(np.mean(applied_noise**2)):.6f}")
    print(
        "unmatched raw-action noise RMS: "
        f"{np.sqrt(np.mean(unmatched_applied_noise**2)):.6f}"
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
