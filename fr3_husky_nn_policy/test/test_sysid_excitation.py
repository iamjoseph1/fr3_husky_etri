import pytest

from fr3_husky_nn_policy.fr3_sysid_excitation import build_trajectory


def _default_trajectory():
    return build_trajectory(
        settle_s=0.5,
        friction_amplitude=0.15,
        friction_speeds=[0.05, 0.10, 0.15],
        inertia_amplitude=0.05,
        inertia_freqs=[0.5, 1.0],
        inertia_cycles=3,
        dwell_s=0.4,
        friction_blend_s=0.30,
    )


def test_rich_sysid_trajectory_returns_to_q0_smoothly():
    trajectory = _default_trajectory()

    assert trajectory.duration > 0.0
    start = trajectory.eval(0.0)
    end = trajectory.eval(trajectory.duration)
    assert start[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    assert end[:3] == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    assert end[4] is False  # an endpoint belongs to the final q0 hold

    # Adjacent segments share position, velocity, and acceleration.  This is
    # important for a physical reference: the friction cruise is not preceded
    # by the discontinuous velocity jump that a plain position ramp creates.
    elapsed = 0.0
    for previous, following in zip(trajectory.segments, trajectory.segments[1:]):
        end_of_previous = previous.eval(previous.dur)
        start_of_following = following.eval(0.0)
        assert end_of_previous == pytest.approx(start_of_following, abs=1.0e-11), (
            previous.phase,
            following.phase,
        )
        elapsed += previous.dur


def test_friction_stage_has_positive_and_negative_constant_speed_samples():
    trajectory = _default_trajectory()
    phases = [segment.phase for segment in trajectory.segments]

    assert any("fric_cruise_pos_out" in phase for phase in phases)
    assert any("fric_cruise_neg_out" in phase for phase in phases)
    assert any("inertia_sine_f0.50" == phase for phase in phases)
    assert any("inertia_sine_f1.00" == phase for phase in phases)
    assert trajectory.peak_displacement() <= 0.150001
    assert trajectory.peak_speed() <= 0.400001
    assert trajectory.peak_accel() <= 5.000001
    assert trajectory.speed_upper_bound() >= trajectory.peak_speed()
    assert trajectory.accel_upper_bound() >= trajectory.peak_accel()


def test_friction_profile_rejects_no_cruise_interval():
    with pytest.raises(ValueError, match="constant-speed interval"):
        build_trajectory(
            settle_s=0.0,
            friction_amplitude=0.02,
            friction_speeds=[0.10],
            inertia_amplitude=0.01,
            inertia_freqs=[0.5],
            inertia_cycles=1,
            dwell_s=0.0,
            friction_blend_s=0.30,
            do_inertia=False,
        )
