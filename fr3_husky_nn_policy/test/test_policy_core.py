from pathlib import Path

import numpy as np

from fr3_husky_nn_policy.numpy_actor import NumpyMLPActor
from fr3_husky_nn_policy.observation import (
    DEFAULT_DUAL_ARM_JOINT_POSITION,
    DEFAULT_RIGHT_JOINT_POSITION,
    build_liftcube_observation,
    build_reach_observation,
)


def test_numpy_actor_elu(tmp_path):
    model_path = tmp_path / "actor.npz"
    np.savez(
        model_path,
        activation=np.asarray("elu"),
        num_layers=np.asarray(2, dtype=np.int64),
        source_sha256=np.asarray("test"),
        weight_0=np.asarray([[1.0, -1.0], [-2.0, 0.5]], dtype=np.float32),
        bias_0=np.asarray([0.0, 0.25], dtype=np.float32),
        weight_1=np.asarray([[0.5, -0.25]], dtype=np.float32),
        bias_1=np.asarray([0.1], dtype=np.float32),
    )
    actor = NumpyMLPActor(model_path)
    value = np.asarray([0.2, 0.4], dtype=np.float32)
    hidden = np.asarray([np.expm1(-0.2), 0.05], dtype=np.float32)
    expected = np.asarray([0.5 * hidden[0] - 0.25 * hidden[1] + 0.1])
    np.testing.assert_allclose(actor(value), expected, rtol=1.0e-6, atol=1.0e-6)


def test_liftcube_observation_layout():
    joint_position = DEFAULT_RIGHT_JOINT_POSITION + 0.1
    joint_velocity = np.arange(9, dtype=np.float32)
    object_position = np.asarray([0.4, -0.2, 0.4], dtype=np.float32)
    target_position = np.asarray([0.45, -0.3, 0.5], dtype=np.float32)
    previous_action = np.arange(8, dtype=np.float32) * 0.01

    observation = build_liftcube_observation(
        joint_position,
        joint_velocity,
        object_position,
        target_position,
        previous_action,
    )

    assert observation.shape == (36,)
    np.testing.assert_allclose(observation[0:9], 0.1, atol=1.0e-6)
    np.testing.assert_array_equal(observation[9:18], joint_velocity)
    np.testing.assert_array_equal(observation[18:21], object_position)
    np.testing.assert_array_equal(observation[21:24], target_position)
    np.testing.assert_array_equal(
        observation[24:28], np.asarray([0.0, 0.0, 0.0, 1.0])
    )
    np.testing.assert_array_equal(observation[28:36], previous_action)


def test_deployed_model_metadata():
    model_path = (
        Path(__file__).parents[1] / "models" / "dual_fr3_lift_v3_actor.npz"
    )
    actor = NumpyMLPActor(model_path)
    assert actor.input_dim == 36
    assert actor.output_dim == 8
    assert (
        actor.source_sha256
        == "047f9314068ddfe43de22d0e0e5a5cbe675cb55c106f26f36b4725886bd3972e"
    )



def test_reach_observation_layout():
    joint_position = DEFAULT_DUAL_ARM_JOINT_POSITION + 0.1
    joint_velocity = np.arange(14, dtype=np.float32)
    target_position = np.asarray([0.5, 0.05, 0.2], dtype=np.float32)
    previous_action = np.arange(14, dtype=np.float32) * 0.01

    observation = build_reach_observation(
        joint_position,
        joint_velocity,
        target_position,
        previous_action,
    )

    assert observation.shape == (58,)
    np.testing.assert_allclose(observation[0:14], 0.1, atol=1.0e-6)
    np.testing.assert_array_equal(observation[14:28], joint_velocity)
    np.testing.assert_array_equal(observation[28:31], target_position)
    np.testing.assert_array_equal(observation[31:34], target_position)
    np.testing.assert_array_equal(
        observation[34:38], np.asarray([1.0, 0.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(observation[38:52], previous_action)
    np.testing.assert_array_equal(observation[52:58], np.zeros(6))


def test_reach_deployed_model_metadata():
    model_path = (
        Path(__file__).parents[1] / "models" / "dual_fr3_reach_actor.npz"
    )
    actor = NumpyMLPActor(model_path)
    assert actor.input_dim == 58
    assert actor.output_dim == 14
    assert (
        actor.source_sha256
        == "0e57514b34712119b72e78b98eaf515d9b5c64140261d0eb91bfa0df80883b8e"
    )
