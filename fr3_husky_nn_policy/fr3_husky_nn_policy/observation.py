from __future__ import annotations

import numpy as np


RIGHT_JOINT_NAMES = (
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
    "right_fr3_finger_joint1",
    "right_fr3_finger_joint2",
)

DEFAULT_RIGHT_JOINT_POSITION = np.asarray(
    [0.5236, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 0.04, 0.04],
    dtype=np.float32,
)


def build_liftcube_observation(
    joint_position,
    joint_velocity,
    object_position,
    target_position,
    previous_action,
) -> np.ndarray:
    """Build the exact 36-D actor input used by dual_fr3_lift_v3."""

    joint_position = np.asarray(joint_position, dtype=np.float32)
    joint_velocity = np.asarray(joint_velocity, dtype=np.float32)
    object_position = np.asarray(object_position, dtype=np.float32)
    target_position = np.asarray(target_position, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)

    expected = {
        "joint_position": (joint_position, (9,)),
        "joint_velocity": (joint_velocity, (9,)),
        "object_position": (object_position, (3,)),
        "target_position": (target_position, (3,)),
        "previous_action": (previous_action, (8,)),
    }
    for name, (value, shape) in expected.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains NaN or Inf")

    target_quaternion_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    observation = np.concatenate(
        (
            joint_position - DEFAULT_RIGHT_JOINT_POSITION,
            joint_velocity,
            object_position,
            target_position,
            target_quaternion_xyzw,
            previous_action,
        )
    )
    if observation.shape != (36,):
        raise RuntimeError(f"Internal observation layout error: {observation.shape}")
    return observation
