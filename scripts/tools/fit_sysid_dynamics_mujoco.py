#!/usr/bin/env python3
"""Inverse-dynamics residual fit for dual-FR3 SysID using MuJoCo as the engine.

This is the MuJoCo counterpart of ``fit_sysid_dynamics.py`` (which uses a
pinocchio URDF).  It is the preferred tool when the goal is to make the *MuJoCo*
plant match the real robot, because:

* MuJoCo's own ``mj_inverse`` provides the rigid-body inverse dynamics
  ``tau_rbd = M(q) qddot + C(q, qdot) qdot + g(q)`` for the exact model that the
  simulation uses -- including the real end-effector (plate + camera mount), the
  arm mounting pose (so per-arm gravity direction is correct automatically), and
  the same inertials you will later tune.
* residual = ``tau_meas - tau_rbd`` is then regressed for friction/armature, and
  the *same* MuJoCo parameters (``dof_frictionloss``, ``dof_damping``,
  ``dof_armature``) are what you set to close the gap.  No URDF conversion.

The reference model must match the physical robot during data collection.  For
the plate setup (gripper removed, plate + D405 mount attached) generate the MJCF
with the matching xacro args, e.g.:

    ros2 run xacro xacro \
      $(ros2 pkg prefix fr3_husky_description)/share/fr3_husky_description/mjcf/dual_fr3.xml.xacro \
      hand:=false with_realsense:=true mobile:=false \
      -o /tmp/dual_fr3_plate.xml

Then:

    python3 fit_sysid_dynamics_mujoco.py \
      --mjcf /tmp/dual_fr3_plate.xml \
      --csv 'logs/sysid_raw/real/*.csv'

By default the tool zeros ``dof_damping``, ``dof_frictionloss``, and
``dof_armature`` so ``mj_inverse`` returns the pure link rigid-body torque; the
residual then contains the *full* real friction and reflected inertia to be
identified.  It also forces gravity on (the Reach MuJoCo config runs at zero
gravity, which would be wrong for a torque comparison against the real robot).
"""

from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - depends on the experiment PC
    raise SystemExit(
        "mujoco python is required: pip install mujoco (run on the sim PC)"
    ) from exc

from scipy.signal import butter, filtfilt

JOINT_NAMES = [
    f"{side}_fr3_joint{i}"
    for side in ("left", "right")
    for i in range(1, 8)
]


@dataclass
class Log:
    path: Path
    t: np.ndarray
    phase: np.ndarray
    active_joint: int
    q: np.ndarray      # (N, 14)
    qd: np.ndarray     # (N, 14)
    tau: np.ndarray    # (N, 14)
    fs: float


def _col(header, name):
    return header.index(name) if name in header else None


def load_log(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 5:
        return None
    header, data = rows[0], rows[1:]
    it = _col(header, "t_sec")
    ip = _col(header, "phase")
    ia = _col(header, "active_joint")
    qcols = [_col(header, f"q_{n}") for n in JOINT_NAMES]
    qdcols = [_col(header, f"qd_{n}") for n in JOINT_NAMES]
    taucols = [_col(header, f"tau_meas_{n}") for n in JOINT_NAMES]
    if it is None or any(c is None for c in qcols + qdcols + taucols):
        print(f"  skip {path.name}: missing q_/qd_/tau_meas_ columns")
        return None

    def grab(cols):
        return np.array([[float(r[c]) for c in cols] for r in data], dtype=np.float64)

    t = np.array([float(r[it]) for r in data], dtype=np.float64)
    phase = np.array([r[ip] for r in data]) if ip is not None else np.full(len(data), "")
    active = -1
    if ia is not None:
        vals = {int(float(r[ia])) for r in data if r[ia] not in ("", "-1")}
        active = next(iter(vals)) if len(vals) == 1 else -1
    dt = np.median(np.diff(t)) if t.size > 1 else 0.0
    fs = 1.0 / dt if dt > 0 else 0.0
    return Log(path, t, phase, active, grab(qcols), grab(qdcols), grab(taucols), fs)


class MujocoEngine:
    """Map the 14 arm joints into a full MuJoCo model and run mj_inverse."""

    def __init__(self, mjcf: str, zero_passive: bool, gravity: float):
        self.model = mujoco.MjModel.from_xml_path(mjcf)
        self.data = mujoco.MjData(self.model)
        # Free-air inverse dynamics: no constraint/contact forces should leak
        # into qfrc_inverse.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        # The Reach MuJoCo config runs at zero gravity; a torque comparison
        # against the real robot must include gravity.
        self.model.opt.gravity[:] = np.array([0.0, 0.0, gravity])
        if zero_passive:
            # Pure rigid-body torque so the residual holds the *full* real
            # friction + reflected inertia to be identified.
            self.model.dof_damping[:] = 0.0
            self.model.dof_frictionloss[:] = 0.0
            self.model.dof_armature[:] = 0.0
        # Resolve the arm joints to qpos / dof addresses.
        self.qadr = np.empty(14, dtype=int)
        self.vadr = np.empty(14, dtype=int)
        for k, name in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise SystemExit(f"joint {name!r} not found in {mjcf}")
            self.qadr[k] = self.model.jnt_qposadr[jid]
            self.vadr[k] = self.model.jnt_dofadr[jid]
        # A neutral home for all non-arm DoFs (base pose, fingers if present).
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0) if self.model.nkey else None
        self._qpos0 = self.data.qpos.copy()

    def rbd_torque(self, q14, qd14, qdd14):
        """Rigid-body inverse-dynamics torque for the 14 arm joints."""
        n = q14.shape[0]
        tau = np.zeros((n, 14))
        for kk in range(n):
            self.data.qpos[:] = self._qpos0
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            self.data.qpos[self.qadr] = q14[kk]
            self.data.qvel[self.vadr] = qd14[kk]
            self.data.qacc[self.vadr] = qdd14[kk]
            mujoco.mj_inverse(self.model, self.data)
            tau[kk] = self.data.qfrc_inverse[self.vadr]
        return tau

    def gravity_torque(self, q14):
        return self.rbd_torque(q14, np.zeros_like(q14), np.zeros_like(q14))


def filtered_qdd(qd, fs, cutoff_hz):
    if fs <= 0 or qd.shape[0] < 30:
        return np.gradient(qd, axis=0) * max(fs, 1.0), qd
    wn = min(0.99, cutoff_hz / (0.5 * fs))
    b, a = butter(4, wn)
    qd_f = filtfilt(b, a, qd, axis=0)
    return np.gradient(qd_f, 1.0 / fs, axis=0), qd_f


def stationary_check(logs, engine, qd_thresh):
    res = {j: [] for j in range(14)}
    tm = {j: [] for j in range(14)}
    gm = {j: [] for j in range(14)}
    total = 0
    for log in logs:
        stat = np.all(np.abs(log.qd) < qd_thresh, axis=1)
        if not stat.any():
            continue
        idx = np.flatnonzero(stat)
        g = engine.gravity_torque(log.q[idx])
        total += idx.size
        for row, k in enumerate(idx):
            for j in range(14):
                if np.isfinite(log.tau[k, j]):
                    res[j].append(log.tau[k, j] - g[row, j])
                    tm[j].append(log.tau[k, j])
                    gm[j].append(g[row, j])
    print(f"\n=== Stationary-residual check ({total} stationary rows) ===")
    print(f"{'joint':<22}{'tau_meas':>10}{'g(q)=mj':>12}{'residual':>12}{'std':>9}")
    worst = 0.0
    for j in range(14):
        if not res[j]:
            continue
        r = np.array(res[j])
        worst = max(worst, abs(float(np.mean(r))))
        flag = "  <-- large" if abs(float(np.mean(r))) > 2.0 else ""
        print(f"{JOINT_NAMES[j]:<22}{np.mean(tm[j]):>10.3f}{np.mean(gm[j]):>12.3f}"
              f"{np.mean(r):>12.3f}{np.std(r):>9.3f}{flag}")
    print(f"\nworst |mean residual| = {worst:.3f} Nm")
    print("near 0 => MuJoCo mass/EE/gravity/mapping match the real robot at rest.")


def friction_fit(logs, engine, cutoff_hz, moving_qd, cruise_qdd):
    print("\n=== Friction / inertia fit (per active joint) ===")
    print("(trustworthy only on rich bidirectional-sweep CSVs)")
    for log in logs:
        j = log.active_joint
        if j < 0:
            continue
        qdd, qd_f = filtered_qdd(log.qd, log.fs, cutoff_hz)
        tau_rbd = engine.rbd_torque(log.q, qd_f, qdd)
        resid = log.tau[:, j] - tau_rbd[:, j]
        v, a = qd_f[:, j], qdd[:, j]
        cruise = (np.abs(v) > moving_qd) & (np.abs(a) < cruise_qdd)
        if cruise.sum() < 20:
            print(f"{JOINT_NAMES[j]:<22} only {int(cruise.sum())} cruise samples "
                  "(poor excitation)")
            continue
        A = np.column_stack([np.sign(v[cruise]), v[cruise], np.ones(int(cruise.sum()))])
        (Fc, Fv, b), *_ = np.linalg.lstsq(A, resid[cruise], rcond=None)
        rmse = float(np.sqrt(np.mean((resid[cruise] - A @ [Fc, Fv, b]) ** 2)))
        print(f"{JOINT_NAMES[j]:<22} Fc={Fc:+.4f} Nm  Fv={Fv:+.4f} Nm*s/rad  "
              f"bias={b:+.4f}  RMSE={rmse:.4f}  (n={int(cruise.sum())})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mjcf", required=True, help="compiled MJCF xml (xacro already expanded)")
    ap.add_argument("--csv", required=True, nargs="+")
    ap.add_argument("--stationary-qd", type=float, default=0.01)
    ap.add_argument("--cutoff-hz", type=float, default=25.0)
    ap.add_argument("--moving-qd", type=float, default=0.02)
    ap.add_argument("--cruise-qdd", type=float, default=0.5)
    ap.add_argument("--gravity", type=float, default=-9.81)
    ap.add_argument("--keep-passive", action="store_true",
                    help="do NOT zero damping/frictionloss/armature (residual = real - model)")
    ap.add_argument("--no-fit", action="store_true")
    args = ap.parse_args()

    paths = []
    for pattern in args.csv:
        paths += [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        raise SystemExit("no CSV files matched")
    logs = [lg for lg in (load_log(p) for p in paths) if lg is not None]
    if not logs:
        raise SystemExit("no usable logs")
    print(f"loaded {len(logs)} log(s), sample rate ~ {logs[0].fs:.1f} Hz")

    engine = MujocoEngine(args.mjcf, zero_passive=not args.keep_passive, gravity=args.gravity)
    print(f"MuJoCo model: nq={engine.model.nq} nv={engine.model.nv}, "
          f"passive {'kept' if args.keep_passive else 'zeroed'}, "
          f"gravity={args.gravity}")
    stationary_check(logs, engine, args.stationary_qd)
    if not args.no_fit:
        friction_fit(logs, engine, args.cutoff_hz, args.moving_qd, args.cruise_qdd)


if __name__ == "__main__":
    main()
