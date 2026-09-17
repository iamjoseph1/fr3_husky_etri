"""Pure (ROS-free) rich-excitation trajectory generator for FR3 SysID.

This module has *no* ROS dependency so the excitation math can be unit-checked
in isolation.  It produces a single-joint, q0-relative reference ``delta(t)``
together with its analytic velocity and acceleration.  The ROS node commands a
measured-relative tracking target from ``delta`` and logs the commanded
``(delta, vel, acc)`` alongside the measured ``(q, qd, tau_meas)`` so the
offline inverse-dynamics fit can use the excitation directly.

Two stages, concatenated per joint (see about_sysid.md, roadmap step 2):

* ``friction_sweep`` -- constant-velocity out-and-back segments at several
  speeds and both directions.  During a constant-velocity segment ``q̈ ≈ 0`` so
  ``tau_meas ≈ g(q) + Fc·sign(q̇) + Fv·q̇``; sweeping several speeds in both
  directions isolates Coulomb (``Fc``) and viscous (``Fv``) friction.
* ``inertia_sine`` -- Hann-windowed sinusoids (acceleration-rich) to excite the
  link inertia and reflected rotor inertia (armature).  The window makes each
  segment start and end at zero position *and* zero velocity, so there is no
  command discontinuity.

Every trajectory starts and ends at ``delta = 0`` (i.e. back at q0) and is
bounded in amplitude, speed, and acceleration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Segment:
    kind: str          # "velocity_blend" | "linear" | "hold" | "windowed_sine"
    dur: float         # segment duration (s)
    phase: str         # human-readable phase label
    # linear: delta = start + slope * tau
    start: float = 0.0
    slope: float = 0.0
    # velocity_blend: smoothly change velocity from v_start to v_end.  This
    # is the acceleration/deceleration part surrounding every friction
    # cruise.  Unlike a position ramp followed by a hold, its position,
    # velocity, and acceleration all join the neighbouring segments
    # continuously.
    v_start: float = 0.0
    v_end: float = 0.0
    # hold: delta = value
    value: float = 0.0
    # windowed_sine: delta = amp * w(tau) * sin(2*pi*f*tau)
    amp: float = 0.0
    freq: float = 0.0

    def eval(self, tau: float) -> tuple[float, float, float]:
        """Return (delta, vel, acc) at local time ``tau`` in [0, dur]."""
        if self.kind == "velocity_blend":
            if self.dur <= 0.0:
                return self.start, self.v_end, 0.0
            x = min(1.0, max(0.0, tau / self.dur))
            # A half-cosine velocity blend has zero acceleration at both
            # boundaries.  Its integral is 1/2, which makes the end position
            # exact and avoids a hidden displacement correction at joins.
            h = 0.5 * (1.0 - math.cos(math.pi * x))
            hd = 0.5 * math.pi * math.sin(math.pi * x) / self.dur
            dv = self.v_end - self.v_start
            delta = self.start + self.v_start * tau + dv * self.dur * (
                0.5 * x - math.sin(math.pi * x) / (2.0 * math.pi)
            )
            vel = self.v_start + dv * h
            acc = dv * hd
            return delta, vel, acc
        if self.kind == "linear":
            return self.start + self.slope * tau, self.slope, 0.0
        if self.kind == "hold":
            return self.value, 0.0, 0.0
        if self.kind == "windowed_sine":
            T = self.dur
            two_pi_f = 2.0 * math.pi * self.freq
            two_pi_T = 2.0 * math.pi / T
            # Hann window w and its derivatives.
            w = 0.5 * (1.0 - math.cos(two_pi_T * tau))
            wd = 0.5 * two_pi_T * math.sin(two_pi_T * tau)
            wdd = 0.5 * two_pi_T * two_pi_T * math.cos(two_pi_T * tau)
            s = math.sin(two_pi_f * tau)
            c = math.cos(two_pi_f * tau)
            sd = two_pi_f * c
            sdd = -two_pi_f * two_pi_f * s
            delta = self.amp * w * s
            vel = self.amp * (wd * s + w * sd)
            acc = self.amp * (wdd * s + 2.0 * wd * sd + w * sdd)
            return delta, vel, acc
        raise ValueError(f"unknown segment kind {self.kind!r}")

    def speed_upper_bound(self) -> float:
        """Conservative analytic bound on ``abs(velocity)`` for this segment."""
        if self.kind == "velocity_blend":
            return max(abs(self.v_start), abs(self.v_end))
        if self.kind == "linear":
            return abs(self.slope)
        if self.kind == "hold":
            return 0.0
        if self.kind == "windowed_sine":
            # |w'| <= pi / T and |sin'| <= 2*pi*f.
            return abs(self.amp) * (math.pi / self.dur + 2.0 * math.pi * self.freq)
        raise ValueError(f"unknown segment kind {self.kind!r}")

    def accel_upper_bound(self) -> float:
        """Conservative analytic bound on ``abs(acceleration)``."""
        if self.kind == "velocity_blend":
            return abs(self.v_end - self.v_start) * math.pi / (2.0 * self.dur)
        if self.kind in ("linear", "hold"):
            return 0.0
        if self.kind == "windowed_sine":
            # delta'' = A(w''s + 2w's' + ws''); use |w| <= 1,
            # |w'| <= pi/T, |w''| <= 2*pi^2/T^2.
            wdd = 2.0 * math.pi * math.pi / (self.dur * self.dur)
            cross = 4.0 * math.pi * math.pi * self.freq / self.dur
            sine = 4.0 * math.pi * math.pi * self.freq * self.freq
            return abs(self.amp) * (wdd + cross + sine)
        raise ValueError(f"unknown segment kind {self.kind!r}")


@dataclass
class Trajectory:
    segments: list[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(seg.dur for seg in self.segments)

    def eval(self, t: float) -> tuple[float, float, float, str, bool]:
        """Return (delta, vel, acc, phase, done) at global time ``t`` (s)."""
        if t < 0.0:
            return 0.0, 0.0, 0.0, "pre", False
        acc_time = 0.0
        for seg in self.segments:
            if t <= acc_time + seg.dur:
                delta, vel, acc = seg.eval(t - acc_time)
                return delta, vel, acc, seg.phase, False
            acc_time += seg.dur
        return 0.0, 0.0, 0.0, "done", True

    def peak_speed(self, dt: float = 1.0e-3) -> float:
        return max(abs(self.eval(t)[1]) for t in _timeline(self.duration, dt))

    def peak_accel(self, dt: float = 1.0e-3) -> float:
        return max(abs(self.eval(t)[2]) for t in _timeline(self.duration, dt))

    def peak_displacement(self, dt: float = 1.0e-3) -> float:
        """Numerical peak of the q0-relative position command.

        This is primarily a preview/reporting helper.  The node's live joint
        limit check also uses the configured component amplitudes directly,
        so safety does not depend on this sampling resolution.
        """
        return max(abs(self.eval(t)[0]) for t in _timeline(self.duration, dt))

    def speed_upper_bound(self) -> float:
        return max((seg.speed_upper_bound() for seg in self.segments), default=0.0)

    def accel_upper_bound(self) -> float:
        return max((seg.accel_upper_bound() for seg in self.segments), default=0.0)


def _timeline(total: float, dt: float):
    n = int(total / dt) + 1
    return (i * dt for i in range(n + 1))


def build_friction_sweep(
    amplitude: float,
    speeds: list[float],
    dwell_s: float,
    blend_s: float,
) -> list[Segment]:
    """Constant-velocity out-and-back at each speed, both directions.

    For each speed v: 0 -> +A (at +v), hold, +A -> 0 (at -v), hold,
    0 -> -A (at -v), hold, -A -> 0 (at +v), hold.
    """
    if blend_s <= 0.0:
        raise ValueError("friction blend_s must be positive")

    segs: list[Segment] = []
    a = abs(amplitude)

    def add_leg(start: float, end: float, speed: float, tag: str):
        distance = abs(end - start)
        signed_speed = math.copysign(speed, end - start)
        # Each velocity blend travels |v| * blend_s / 2.  Thus a full leg
        # needs amplitude > |v| * blend_s to leave a non-zero constant-speed
        # interval.  Failing early is important: without that interval the
        # requested data cannot identify viscous/Coulomb friction reliably.
        cruise_distance = distance - speed * blend_s
        if cruise_distance <= 0.0:
            raise ValueError(
                "friction_amplitude_rad must exceed speed * friction_blend_s "
                "for every speed to leave a constant-speed interval "
                f"(got {distance:.4f} <= {speed * blend_s:.4f})"
            )
        blend_distance = 0.5 * signed_speed * blend_s
        cruise_start = start + blend_distance
        cruise_duration = cruise_distance / speed
        segs.extend(
            [
                Segment(
                    "velocity_blend",
                    blend_s,
                    f"fric_accel_{tag}_v{speed:.3f}",
                    start=start,
                    v_start=0.0,
                    v_end=signed_speed,
                ),
                Segment(
                    "linear",
                    cruise_duration,
                    f"fric_cruise_{tag}_v{speed:.3f}",
                    start=cruise_start,
                    slope=signed_speed,
                ),
                Segment(
                    "velocity_blend",
                    blend_s,
                    f"fric_decel_{tag}_v{speed:.3f}",
                    start=end - blend_distance,
                    v_start=signed_speed,
                    v_end=0.0,
                ),
            ]
        )

    for speed in speeds:
        speed = abs(speed)
        if speed <= 0.0 or a <= 0.0:
            continue
        add_leg(0.0, +a, speed, "pos_out")
        segs.append(Segment("hold", dwell_s, f"fric_hold_pos_v{speed:.3f}", value=+a))
        add_leg(+a, 0.0, speed, "neg_from_pos")
        segs.append(Segment("hold", dwell_s, f"fric_hold_zero_v{speed:.3f}", value=0.0))
        add_leg(0.0, -a, speed, "neg_out")
        segs.append(Segment("hold", dwell_s, f"fric_hold_neg_v{speed:.3f}", value=-a))
        add_leg(-a, 0.0, speed, "pos_from_neg")
        segs.append(Segment("hold", dwell_s, f"fric_hold_zero2_v{speed:.3f}", value=0.0))
    return segs


def build_inertia_sine(
    amplitude: float,
    freqs: list[float],
    cycles: int,
    dwell_s: float,
) -> list[Segment]:
    """Hann-windowed sinusoids at each frequency (acceleration-rich)."""
    segs: list[Segment] = []
    a = abs(amplitude)
    for f in freqs:
        f = abs(f)
        if f <= 0.0 or a <= 0.0 or cycles <= 0:
            continue
        segs.append(
            Segment("windowed_sine", cycles / f, f"inertia_sine_f{f:.2f}", amp=a, freq=f)
        )
        segs.append(Segment("hold", dwell_s, f"inertia_hold_f{f:.2f}", value=0.0))
    return segs


def build_trajectory(
    *,
    settle_s: float,
    friction_amplitude: float,
    friction_speeds: list[float],
    inertia_amplitude: float,
    inertia_freqs: list[float],
    inertia_cycles: int,
    dwell_s: float,
    friction_blend_s: float = 0.25,
    friction_dwell_s: float | None = None,
    inertia_dwell_s: float | None = None,
    do_friction: bool = True,
    do_inertia: bool = True,
) -> Trajectory:
    """Assemble the per-joint excitation: settle -> friction -> inertia -> zero."""
    # ``dwell_s`` remains the common default so the pure generator has a
    # compact API.  The node exposes independent dwell settings because the
    # hardware experiment benefits from a longer stop after velocity sweeps.
    friction_dwell_s = dwell_s if friction_dwell_s is None else friction_dwell_s
    inertia_dwell_s = dwell_s if inertia_dwell_s is None else inertia_dwell_s
    segs: list[Segment] = [Segment("hold", settle_s, "q0_settle", value=0.0)]
    if do_friction:
        segs += build_friction_sweep(
            friction_amplitude,
            friction_speeds,
            friction_dwell_s,
            friction_blend_s,
        )
    if do_inertia:
        segs += build_inertia_sine(
            inertia_amplitude,
            inertia_freqs,
            inertia_cycles,
            inertia_dwell_s,
        )
    final_dwell_s = inertia_dwell_s if do_inertia else friction_dwell_s
    segs.append(Segment("hold", final_dwell_s, "q0_final", value=0.0))
    return Trajectory(segs)


if __name__ == "__main__":
    # Self-check: continuity, endpoints at q0, and speed/accel bounds.
    traj = build_trajectory(
        settle_s=0.5,
        friction_amplitude=0.30,
        friction_speeds=[0.05, 0.1, 0.2, 0.4],
        inertia_amplitude=0.15,
        inertia_freqs=[0.5, 1.0, 2.0],
        inertia_cycles=4,
        dwell_s=0.5,
        friction_blend_s=0.25,
    )
    dt = 1.0e-3
    prev = None
    max_jump = 0.0
    for t in _timeline(traj.duration, dt):
        delta, vel, acc, phase, done = traj.eval(t)
        if prev is not None:
            max_jump = max(max_jump, abs(delta - prev))
        prev = delta
    d0, *_ = traj.eval(0.0)
    dend, vend, aend, *_ = traj.eval(traj.duration)
    print(f"duration          : {traj.duration:.3f} s")
    print(f"peak |speed|      : {traj.peak_speed(dt):.4f} rad/s")
    print(f"peak |accel|      : {traj.peak_accel(dt):.4f} rad/s^2")
    print(f"peak |delta|      : {traj.peak_displacement(dt):.4f} rad")
    print(f"max step (1 ms)   : {max_jump:.6f} rad  (continuity)")
    print(f"delta(0)          : {d0:.6f} rad  (should be 0)")
    print(f"delta(end)        : {dend:.6f} rad  (should be ~0)")
    print(f"vel(end)          : {vend:.6f} rad/s (should be ~0)")
