"""Pure helpers for Reach policy/controller action semantics."""

from __future__ import annotations

import numpy as np


def encode_reach_joint_command(
    absolute_target: np.ndarray,
    relative_offset: np.ndarray,
    policy_step_reference: bool,
) -> tuple[np.ndarray, bool]:
    """Select the ROS command representation for one processed Reach action.

    Returns the 14 command values and the ``relative_position_offsets`` flag.
    Policy-step references send the absolute target captured by the policy
    node; physics-step references send the offset that the controller adds to
    every new joint-position measurement.
    """

    absolute_target = np.asarray(absolute_target, dtype=np.float64)
    relative_offset = np.asarray(relative_offset, dtype=np.float64)
    if absolute_target.shape != (14,) or relative_offset.shape != (14,):
        raise ValueError("Reach joint targets and offsets must be 14-vectors")
    if not np.all(np.isfinite(absolute_target)) or not np.all(
        np.isfinite(relative_offset)
    ):
        raise ValueError("Reach joint targets and offsets must be finite")
    if policy_step_reference:
        return absolute_target.copy(), False
    return relative_offset.copy(), True
