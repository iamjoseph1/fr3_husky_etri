import numpy as np

from fr3_husky_nn_policy.action_semantics import encode_reach_joint_command


def test_reach_joint_command_encodes_both_reference_modes():
    measured = np.linspace(-0.7, 0.6, 14)
    offset = np.linspace(-0.1, 0.1, 14)
    target = measured + offset

    values, is_relative = encode_reach_joint_command(target, offset, False)
    np.testing.assert_array_equal(values, offset)
    assert is_relative is True

    values, is_relative = encode_reach_joint_command(target, offset, True)
    np.testing.assert_array_equal(values, target)
    assert is_relative is False
