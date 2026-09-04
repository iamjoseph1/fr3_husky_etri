import csv
import json

import numpy as np

from fr3_husky_nn_policy.reach_logger import ReachRunLogger, rotate_vector


def test_eef_offset_is_rotated_from_link_frame_to_base():
    half_sqrt_two = np.sqrt(0.5)
    quaternion_xyzw = np.asarray([0.0, 0.0, half_sqrt_two, half_sqrt_two])
    rotated = rotate_vector(quaternion_xyzw, np.asarray([0.132, 0.0, 0.0]))
    np.testing.assert_allclose(rotated, [0.0, 0.132, 0.0], atol=1.0e-12)


def test_reach_run_logger_writes_csv_metadata_and_four_plots(tmp_path):
    logger = ReachRunLogger(
        log_root=tmp_path,
        task_name="dual_fr3_reach",
        base_frame="base",
        left_eef_frame="left_fr3_link7",
        right_eef_frame="right_fr3_link7",
        eef_offset_xyz=np.asarray([0.0, 0.0, 0.132]),
        robot_root_offset_xyz=np.asarray([0.0, 0.0, 0.405]),
        reach_offset_y=0.20,
        sample_rate_hz=20.0,
    )
    logger.append(
        elapsed_s=0.0,
        ros_time_ns=100,
        target_center=np.asarray([0.5, 0.0, 0.2]),
        left_eef=np.asarray([0.4, 0.1, 0.3]),
        right_eef=np.asarray([0.4, -0.1, 0.3]),
    )
    logger.append(
        elapsed_s=0.05,
        ros_time_ns=200,
        target_center=np.asarray([0.5, 0.05, 0.2]),
        left_eef=np.asarray([0.45, 0.2, 0.25]),
        right_eef=np.asarray([0.45, -0.05, 0.25]),
    )

    run_dir, plot_paths = logger.finalize()

    assert len(plot_paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in plot_paths)
    with (run_dir / "eef_trajectory.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert float(rows[0]["left_target_y"]) == 0.20
    assert float(rows[0]["right_target_y"]) == -0.20
    assert float(rows[1]["left_target_y"]) == 0.25
    assert float(rows[1]["right_target_y"]) == -0.15

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["sample_count"] == 2
    assert metadata["eef_local_offset_xyz_m"] == [0.0, 0.0, 0.132]
    assert metadata["per_arm_target_offset_y_m"] == 0.20
