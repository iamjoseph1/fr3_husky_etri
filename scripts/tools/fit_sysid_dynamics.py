#!/usr/bin/env python3
"""Inverse-dynamics residual fit for dual-FR3 SysID data.

Given per-joint CSVs from ``fr3_sysid_node`` (or the older consistency-probe
CSVs, which share the ``q_* / qd_* / tau_meas_*`` schema), this tool computes

    residual_j = tau_meas_j - tau_rbd_j(q, qdot, qddot)

where ``tau_rbd`` is the full rigid-body inverse dynamics (inertia + Coriolis +
gravity) from a 7-DoF FR3 pinocchio model, evaluated per arm.  The residual is
then regressed against the active joint's motion to estimate friction and any
un-modelled (reflected) inertia:

    residual_j ~= Fc_j*sign(qdot_j) + Fv_j*qdot_j [+ Ia_j*qddot_j] [+ b_j]

Two things this script is deliberately good at *before* any rich excitation
data exists:

* **stationary-residual check** -- on truly stationary rows ``tau_meas`` should
  equal ``g(q)``.  ``tau_meas - rnea(q, 0, 0)`` near zero validates the torque
  sign convention, the gravity basis, and the URDF joint mapping in one shot.
* **plumbing** -- CSV parsing, arm mapping, filtering, and rnea evaluation, so
  the pipeline is debugged on existing data.

Friction/inertia *numbers* are only trustworthy on rich-excitation CSVs (real
constant-velocity intervals and windowed sines); on pulse/ramp consistency CSVs
they are reported but should not be believed.
"""

from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.signal import butter, filtfilt

DEFAULT_URDF = "/home/dyros/fr3_control_mujoco_template/franka_fr3/fr3.urdf"

JOINT_NAMES = [
    f"{side}_fr3_joint{i}"
    for side in ("left", "right")
    for i in range(1, 8)
]
LEFT = list(range(0, 7))
RIGHT = list(range(7, 14))


@dataclass
class Log:
    path: Path
    t: np.ndarray            # (N,)
    phase: np.ndarray        # (N,) str
    active_joint: int        # 0..13, or -1 if mixed/unknown
    q: np.ndarray            # (N, 14)
    qd: np.ndarray           # (N, 14)
    tau: np.ndarray          # (N, 14) measured joint torque
    fs: float                # sample rate (Hz)


def _col(header: list[str], name: str) -> int | None:
    return header.index(name) if name in header else None


def load_log(path: Path) -> Log | None:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 5:
        print(f"  skip {path.name}: too few rows")
        return None
    header, data = rows[0], rows[1:]
    it = _col(header, "t_sec")
    ip = _col(header, "phase")
    ia = _col(header, "active_joint")
    qcols = [_col(header, f"q_{n}") for n in JOINT_NAMES]
    qdcols = [_col(header, f"qd_{n}") for n in JOINT_NAMES]
    taucols = [_col(header, f"tau_meas_{n}") for n in JOINT_NAMES]
    if it is None or any(c is None for c in qcols + qdcols + taucols):
        print(f"  skip {path.name}: missing required q_/qd_/tau_meas_ columns")
        return None

    def grab(cols):
        return np.array(
            [[float(r[c]) for c in cols] for r in data], dtype=np.float64
        )

    t = np.array([float(r[it]) for r in data], dtype=np.float64)
    phase = np.array([r[ip] for r in data]) if ip is not None else np.full(len(data), "")
    active = -1
    if ia is not None:
        vals = {int(float(r[ia])) for r in data if r[ia] not in ("", "-1")}
        active = next(iter(vals)) if len(vals) == 1 else -1
    dt = np.median(np.diff(t)) if t.size > 1 else 0.0
    fs = 1.0 / dt if dt > 0 else 0.0
    return Log(path, t, phase, active, grab(qcols), grab(qdcols), grab(taucols), fs)


def make_model(urdf: str):
    model = pin.buildModelFromUrdf(urdf)
    if model.nq != 7:
        raise ValueError(f"expected a 7-DoF FR3 URDF, got nq={model.nq}")
    return model, model.createData()


def rnea_arm(model, data, q7, v7, a7):
    return np.asarray(pin.rnea(model, data, q7, v7, a7))


def rbd_torque(model, data, q, qd, qdd):
    """Full inverse-dynamics torque for all 14 joints, per arm."""
    n = q.shape[0]
    tau = np.zeros((n, 14))
    for k in range(n):
        tau[k, LEFT] = rnea_arm(model, data, q[k, LEFT], qd[k, LEFT], qdd[k, LEFT])
        tau[k, RIGHT] = rnea_arm(model, data, q[k, RIGHT], qd[k, RIGHT], qdd[k, RIGHT])
    return tau


def filtered_qdd(qd, fs, cutoff_hz):
    if fs <= 0 or qd.shape[0] < 30:
        return np.gradient(qd, axis=0) * fs
    wn = min(0.99, cutoff_hz / (0.5 * fs))
    b, a = butter(4, wn)
    qd_f = filtfilt(b, a, qd, axis=0)
    dt = 1.0 / fs
    return np.gradient(qd_f, dt, axis=0), qd_f


def stationary_check(logs, model, data, qd_thresh, gravity_left, gravity_right):
    """tau_meas - g(q) on stationary rows, aggregated across all logs."""
    res = {j: [] for j in range(14)}
    taum = {j: [] for j in range(14)}
    grav = {j: [] for j in range(14)}
    total = 0
    for log in logs:
        stat = np.all(np.abs(log.qd) < qd_thresh, axis=1)
        if not stat.any():
            continue
        total += int(stat.sum())
        zeros = np.zeros(7)
        for k in np.flatnonzero(stat):
            model.gravity.linear = gravity_left
            gl = rnea_arm(model, data, log.q[k, LEFT], zeros, zeros)
            model.gravity.linear = gravity_right
            gr = rnea_arm(model, data, log.q[k, RIGHT], zeros, zeros)
            g14 = np.concatenate([gl, gr])
            for j in range(14):
                if np.isfinite(log.tau[k, j]):
                    res[j].append(log.tau[k, j] - g14[j])
                    taum[j].append(log.tau[k, j])
                    grav[j].append(g14[j])
    print(f"\n=== Stationary-residual check ({total} stationary rows) ===")
    print(f"{'joint':<22}{'tau_meas':>10}{'g(q)=rnea':>12}{'residual':>12}{'std':>9}")
    worst = 0.0
    for j in range(14):
        if not res[j]:
            continue
        r = np.array(res[j])
        tm = float(np.mean(taum[j]))
        gm = float(np.mean(grav[j]))
        rm, rs = float(np.mean(r)), float(np.std(r))
        worst = max(worst, abs(rm))
        flag = "  <-- large" if abs(rm) > 2.0 else ""
        print(f"{JOINT_NAMES[j]:<22}{tm:>10.3f}{gm:>12.3f}{rm:>12.3f}{rs:>9.3f}{flag}")
    print(f"\nworst |mean residual| = {worst:.3f} Nm")
    print("interpretation: near 0 => tau sign/gravity/URDF mapping OK; large =>")
    print("  wrong sign, gravity-comp basis, base mounting orientation, or URDF.")


def friction_fit(logs, model, data, cutoff_hz, moving_qd, cruise_qdd):
    print("\n=== Friction / inertia fit (per active joint) ===")
    print("(numbers only trustworthy on rich-excitation CSVs)")
    for log in logs:
        j = log.active_joint
        if j < 0:
            continue
        out = filtered_qdd(log.qd, log.fs, cutoff_hz)
        qdd, qd_f = out if isinstance(out, tuple) else (out, log.qd)
        tau_rbd = rbd_torque(model, data, log.q, qd_f, qdd)
        resid = log.tau[:, j] - tau_rbd[:, j]
        v = qd_f[:, j]
        a = qdd[:, j]
        moving = np.abs(v) > moving_qd
        cruise = moving & (np.abs(a) < cruise_qdd)
        if cruise.sum() < 20:
            print(f"{log.path.name}: joint {JOINT_NAMES[j]} -- "
                  f"only {int(cruise.sum())} cruise samples (poor excitation)")
            continue
        A = np.column_stack([np.sign(v[cruise]), v[cruise], np.ones(cruise.sum())])
        (Fc, Fv, b), *_ = np.linalg.lstsq(A, resid[cruise], rcond=None)
        pred = A @ np.array([Fc, Fv, b])
        rmse = float(np.sqrt(np.mean((resid[cruise] - pred) ** 2)))
        print(f"{JOINT_NAMES[j]:<22} Fc={Fc:+.4f} Nm  Fv={Fv:+.4f} Nm*s/rad  "
              f"bias={b:+.4f}  RMSE={rmse:.4f} Nm  (n={int(cruise.sum())})")


def _rpy_gravity(rpy):
    r, p, y = rpy
    R = pin.rpy.rpyToMatrix(r, p, y)
    return R.T @ np.array([0.0, 0.0, -9.81])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, nargs="+",
                    help="CSV file(s) or glob(s), e.g. logs/sysid_raw/real/*.csv")
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--stationary-qd", type=float, default=0.01,
                    help="|qd| below which all 14 joints count as stationary (rad/s)")
    ap.add_argument("--cutoff-hz", type=float, default=25.0)
    ap.add_argument("--moving-qd", type=float, default=0.02)
    ap.add_argument("--cruise-qdd", type=float, default=0.5)
    ap.add_argument("--left-base-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--right-base-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--no-fit", action="store_true", help="only stationary check")
    args = ap.parse_args()

    paths: list[Path] = []
    for pattern in args.csv:
        paths += [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        raise SystemExit("no CSV files matched")
    print(f"loading {len(paths)} CSV file(s)...")
    logs = [lg for lg in (load_log(p) for p in paths) if lg is not None]
    if not logs:
        raise SystemExit("no usable logs")
    print(f"usable: {len(logs)}   sample rate ~ {logs[0].fs:.1f} Hz")

    model, data = make_model(args.urdf)
    gl = _rpy_gravity(args.left_base_rpy)
    gr = _rpy_gravity(args.right_base_rpy)
    stationary_check(logs, model, data, args.stationary_qd, gl, gr)
    if not args.no_fit:
        model.gravity.linear = np.array([0.0, 0.0, -9.81])
        friction_fit(logs, model, data, args.cutoff_hz, args.moving_qd, args.cruise_qdd)


if __name__ == "__main__":
    main()
