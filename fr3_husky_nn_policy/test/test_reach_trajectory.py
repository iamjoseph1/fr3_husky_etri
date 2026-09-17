import csv
import json
from pathlib import Path

import numpy as np
import pytest

from fr3_husky_nn_policy.reach_trajectory import ReachActionTrajectory, sha256_file


PACKAGE_ROOT = Path(__file__).parents[1]
TRAJECTORY_ROOT = PACKAGE_ROOT / "trajectories"
ABLATION_ROOT = PACKAGE_ROOT / "ablation_mujoco" / "ablation_mujoco"
NOMINAL_SOURCE = ABLATION_ROOT / (
    "20260911_125736_563151_KST(1000hz,matched dynamics)"
) / "policy_trace.csv"


def test_reach_action_trajectory_loads_optional_noise(tmp_path):
    path = tmp_path / "trajectory.csv"
    columns = [
        "step", "source_elapsed_s", "target_center_x", "target_center_y",
        "target_center_z", *(f"action_{i}" for i in range(14)),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for step in range(2):
            row = {
                "step": step,
                "source_elapsed_s": 0.05 * step,
                "target_center_x": 0.5,
                "target_center_y": 0.0,
                "target_center_z": 0.2,
            }
            row.update({f"action_{i}": 0.01 * i for i in range(14)})
            writer.writerow(row)

    trajectory = ReachActionTrajectory.load(path)
    assert len(trajectory) == 2
    assert trajectory.actions.shape == (2, 14)
    np.testing.assert_array_equal(trajectory.noise, np.zeros((2, 14)))


def test_reach_action_trajectory_rejects_out_of_range_action(tmp_path):
    path = tmp_path / "bad.csv"
    columns = [
        "step", "source_elapsed_s", "target_center_x", "target_center_y",
        "target_center_z", *(f"action_{i}" for i in range(14)),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        row = {
            "step": 0,
            "source_elapsed_s": 0.0,
            "target_center_x": 0.5,
            "target_center_y": 0.0,
            "target_center_z": 0.2,
        }
        row.update({f"action_{i}": 1.1 if i == 0 else 0.0 for i in range(14)})
        writer.writerow(row)

    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        ReachActionTrajectory.load(path)


def test_reach_action_trajectory_rescales_stored_noise(tmp_path):
    path = tmp_path / "trajectory.csv"
    columns = [
        "step", "source_elapsed_s", "target_center_x", "target_center_y",
        "target_center_z", *(f"action_{i}" for i in range(14)),
        *(f"noise_{i}" for i in range(14)),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        row = {
            "step": 0,
            "source_elapsed_s": 0.0,
            "target_center_x": 0.5,
            "target_center_y": 0.0,
            "target_center_z": 0.2,
        }
        row.update({f"action_{i}": 0.3 for i in range(14)})
        row.update({f"noise_{i}": 0.1 for i in range(14)})
        writer.writerow(row)

    trajectory = ReachActionTrajectory.load(path)
    np.testing.assert_allclose(trajectory.action_at(0, 0.0), 0.2)
    np.testing.assert_allclose(trajectory.action_at(0, 1.0), 0.3)
    np.testing.assert_allclose(trajectory.action_at(0, 2.0), 0.4)


def test_packaged_nominal_actions_exactly_copy_source_trace():
    nominal = ReachActionTrajectory.load(
        TRAJECTORY_ROOT / "reach_1khz_matched_nominal.csv"
    )
    source = np.genfromtxt(NOMINAL_SOURCE, delimiter=",", names=True)
    source_actions = np.column_stack([source[f"action_{i}"] for i in range(14)])

    assert len(nominal) == 321
    np.testing.assert_array_equal(nominal.actions, source_actions.astype(np.float32))
    np.testing.assert_array_equal(nominal.noise, np.zeros_like(nominal.noise))


def test_packaged_noisy_trajectory_and_metadata_are_self_consistent():
    nominal = ReachActionTrajectory.load(
        TRAJECTORY_ROOT / "reach_1khz_matched_nominal.csv"
    )
    noisy = ReachActionTrajectory.load(
        TRAJECTORY_ROOT / "reach_1khz_matched_nominal_100hz_matched_noise.csv"
    )
    metadata_path = TRAJECTORY_ROOT / "reach_trajectory_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    np.testing.assert_allclose(
        noisy.actions, nominal.actions + noisy.noise, rtol=0.0, atol=1.0e-7
    )
    np.testing.assert_array_equal(noisy.target_centers, nominal.target_centers)
    assert metadata["trajectory_rows"] == len(noisy)
    assert metadata["noise_seed"] == 100
    assert metadata["nominal_trajectory_sha256"] == sha256_file(nominal.path)
    assert metadata["noisy_trajectory_sha256"] == sha256_file(noisy.path)

    unmatched = ReachActionTrajectory.load(
        TRAJECTORY_ROOT / "reach_1khz_matched_nominal_100hz_unmatched_noise.csv"
    )
    np.testing.assert_allclose(
        unmatched.actions,
        nominal.actions + unmatched.noise,
        rtol=0.0,
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(unmatched.target_centers, nominal.target_centers)
    assert metadata["unmatched_noisy_trajectory_sha256"] == sha256_file(
        unmatched.path
    )
