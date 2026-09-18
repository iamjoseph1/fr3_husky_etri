from pathlib import Path

import numpy as np
import pytest

from fr3_husky_nn_policy.numpy_actor import NumpyMLPActor
from fr3_husky_nn_policy.observation import (
    DEFAULT_DUAL_ARM_JOINT_POSITION,
    DEFAULT_RIGHT_JOINT_POSITION,
    build_liftcube_observation,
    build_reach_observation,
)
from fr3_husky_nn_policy.reach_goal_sequence import (
    ReachGoalSequencePlayer,
    load_reach_goal_sequence,
)


def test_load_reach_goal_sequence_uses_dual_fr3_lab_format(tmp_path):
    sequence_path = tmp_path / "goals.json"
    sequence_path.write_text(
        '{"goals": [{"label": "one", "position": [0.5, 0, 0.2], '
        '"duration_s": 4.0}]}'
    )

    assert load_reach_goal_sequence(str(sequence_path)) == [
        {"label": "one", "position": [0.5, 0.0, 0.2], "duration_s": 4.0}
    ]


def test_load_reach_goal_sequence_rejects_invalid_duration(tmp_path):
    sequence_path = tmp_path / "goals.json"
    sequence_path.write_text('[{"position": [0.5, 0.0, 0.2], "duration_s": 0}]')

    with pytest.raises(ValueError, match="duration_s"):
        load_reach_goal_sequence(str(sequence_path))


def test_reach_goal_sequence_player_uses_absolute_deadlines():
    player = ReachGoalSequencePlayer(
        [
            {"label": "one", "position": [0.5, 0.0, 0.2], "duration_s": 1.0},
            {"label": "two", "position": [0.4, 0.0, 0.2], "duration_s": 2.0},
        ]
    )

    assert player.start(10_000_000_000)["label"] == "one"
    goal, transitions = player.advance(11_500_000_000)
    assert goal["label"] == "two"
    assert transitions == 1
    # The second deadline stays anchored at 11 s + 2 s, not 11.5 s + 2 s.
    assert player.deadline_ns == 13_000_000_000

    goal, transitions = player.advance(13_000_000_000)
    assert goal is None
    assert transitions == 1
    assert player.completed


def test_reach_goal_sequence_player_loops():
    player = ReachGoalSequencePlayer(
        [
            {"label": "one", "position": [0.5, 0.0, 0.2], "duration_s": 1.0},
            {"label": "two", "position": [0.4, 0.0, 0.2], "duration_s": 1.0},
        ],
        loop=True,
    )

    player.start(0)
    goal, transitions = player.advance(2_000_000_000)
    assert goal["label"] == "one"
    assert transitions == 2
    assert player.deadline_ns == 3_000_000_000


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
    assert actor.format_version == 1
    assert actor.output_activation == "identity"
    value = np.asarray([0.2, 0.4], dtype=np.float32)
    hidden = np.asarray([np.expm1(-0.2), 0.05], dtype=np.float32)
    expected = np.asarray([0.5 * hidden[0] - 0.25 * hidden[1] + 0.1])
    np.testing.assert_allclose(actor(value), expected, rtol=1.0e-6, atol=1.0e-6)


def test_numpy_actor_tanh_output(tmp_path):
    model_path = tmp_path / "tanh_actor.npz"
    np.savez(
        model_path,
        format_version=np.asarray(2, dtype=np.int64),
        activation=np.asarray("elu"),
        output_activation=np.asarray("tanh"),
        num_layers=np.asarray(1, dtype=np.int64),
        source_sha256=np.asarray("test-tanh"),
        weight_0=np.asarray([[2.0, -1.0], [-0.5, 0.25]], dtype=np.float32),
        bias_0=np.asarray([0.1, -0.2], dtype=np.float32),
    )
    actor = NumpyMLPActor(model_path)
    value = np.asarray([0.4, -0.3], dtype=np.float32)
    expected = np.tanh(actor.weights[0] @ value + actor.biases[0])

    assert actor.format_version == 2
    assert actor.output_activation == "tanh"
    np.testing.assert_allclose(actor(value), expected, rtol=1.0e-6, atol=1.0e-6)
    assert np.all(np.abs(actor(value)) < 1.0)


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
        observation[34:38], np.asarray([0.0, 0.0, 0.0, 1.0])
    )
    np.testing.assert_array_equal(observation[38:52], previous_action)
    np.testing.assert_array_equal(observation[52:58], np.zeros(6))


def test_reach_deployed_model_metadata():
    expected_models = {
        "dual_fr3_reach_actor_friction_w_1000hz.npz": (
            "150eac1b3014b8e1f172809cc81c618a90f2c976abd6def7c6cc7d3c5e5188cc"
        ),
        "dual_fr3_reach_actor_friction_w_100hz.npz": (
            "8954c5b977cce74cf854fcf76074ddc2110e8c8e2ff74c27ded68083ddcbbe19"
        ),
        "dual_fr3_reach_actor_no_friction_w_100hz.npz": (
            "e74393b2f4013cf044ba3b4bbb5fe56852565ef0a4930d918eb78de2aa937636"
        ),
    }

    model_dir = Path(__file__).parents[1] / "models"
    for model_name, source_sha256 in expected_models.items():
        actor = NumpyMLPActor(model_dir / model_name)
        assert actor.input_dim == 58
        assert actor.output_dim == 14
        assert actor.format_version == 2
        assert actor.output_activation == "tanh"
        assert actor.source_sha256 == source_sha256
