#!/usr/bin/env python3
"""Compare paired MuJoCo and real dual-FR3 consistency-probe CSV files.

The probe records one 14-DoF state stream for each excited joint.  This tool
matches CSVs by active-joint name, subtracts each trial's pre-pulse baseline,
and produces response overlays plus a machine-readable summary.  It does not
fit a dynamics model: it makes configuration and response mismatches visible
before an Isaac Lab/Isaac Sim identification run is attempted.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


JOINT_NAMES = tuple(
    f"{side}_fr3_joint{index}"
    for side in ("left", "right")
    for index in range(1, 8)
)
FILE_NAME = re.compile(r"^(left|right)_fr3_joint([1-7])_")
EPS = 1.0e-10


@dataclass(frozen=True)
class Trace:
    name: str
    source: str
    path: Path
    time_s: np.ndarray
    position_rad: np.ndarray
    all_position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    effort_nm: np.ndarray
    cmd_offset_rad: np.ndarray
    excitation_rad: np.ndarray
    cmd_tau_recon_nm: np.ndarray
    pulse_start_s: float
    pulse_end_s: float

    @property
    def direction(self) -> float:
        active = self.excitation_rad[np.abs(self.excitation_rad) > EPS]
        return float(np.sign(np.median(active))) if active.size else 1.0

    @property
    def sample_rate_hz(self) -> float:
        duration = self.time_s[-1] - self.time_s[0]
        return float((len(self.time_s) - 1) / duration) if duration > 0.0 else float("nan")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing mujoco/ and real/ probe CSV directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/consistency_probe_comparison"),
        help="Directory for PNGs, summary.csv, and report.md.",
    )
    return parser


def _trace_name(path: Path) -> str:
    match = FILE_NAME.match(path.name)
    if match is None:
        raise ValueError(f"Cannot determine active joint from {path.name}")
    return f"{match.group(1)}_fr3_joint{match.group(2)}"


def _column(rows: list[dict[str, str]], name: str, path: Path) -> np.ndarray:
    try:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"{path} is missing required column {name!r}") from error


def load_trace(source: str, path: Path) -> Trace:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) < 10:
        raise ValueError(f"{path} has too few samples ({len(rows)})")

    name = _trace_name(path)
    time_s = _column(rows, "t_sec", path)
    offset = _column(rows, "cmd_offset_rad", path)
    # q0-reference profiles keep a small measured-relative correction active
    # even during q0 holds.  The explicit q0-relative reference column is the
    # actual excitation input in those newer CSVs.  Older pulse/coast files only
    # have cmd_offset_rad, which remains their input signal.
    excitation = (
        _column(rows, "q_ref_delta_from_q0_rad", path)
        if "q_ref_delta_from_q0_rad" in rows[0]
        else offset
    )
    if not np.all(np.diff(time_s) >= 0.0):
        raise ValueError(f"{path} has non-monotonic timestamps")
    active = np.flatnonzero(np.abs(excitation) > EPS)
    if active.size == 0:
        raise ValueError(f"{path} contains no non-zero excitation command")

    return Trace(
        name=name,
        source=source,
        path=path,
        time_s=time_s,
        position_rad=_column(rows, f"q_{name}", path),
        all_position_rad=np.stack(
            [_column(rows, f"q_{joint_name}", path) for joint_name in JOINT_NAMES], axis=1
        ),
        velocity_rad_s=_column(rows, f"qd_{name}", path),
        effort_nm=_column(rows, f"tau_meas_{name}", path),
        cmd_offset_rad=offset,
        excitation_rad=excitation,
        cmd_tau_recon_nm=_column(rows, "tau_cmd_recon_nm", path),
        pulse_start_s=float(time_s[active[0]]),
        pulse_end_s=float(time_s[active[-1]]),
    )


def load_source(root: Path, source: str) -> dict[str, Trace]:
    directory = root / source
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {source} directory: {directory}")
    traces: dict[str, Trace] = {}
    for path in sorted(directory.glob("*.csv")):
        trace = load_trace(source, path)
        if trace.name in traces:
            raise ValueError(f"Duplicate {source} trial for {trace.name}")
        traces[trace.name] = trace
    if not traces:
        raise FileNotFoundError(f"No CSV files under {directory}")
    return traces


def _baseline_mask(trace: Trace) -> np.ndarray:
    # Discard the first 0.2 s of start-up jitter, then use all remaining
    # zero-input data before the command pulse.  The fallback supports shorter
    # profiles while retaining at least a few samples.
    start = trace.time_s[0] + min(0.2, 0.25 * trace.pulse_start_s)
    end = trace.pulse_start_s - 0.05
    mask = (trace.time_s >= start) & (trace.time_s <= end)
    if np.count_nonzero(mask) >= 10:
        return mask
    return trace.time_s < trace.pulse_start_s


def _tail_mask(trace: Trace) -> np.ndarray:
    # Leave 0.8 s after pulse fall for the zero-offset coast, then average the
    # final part of the recorded response.
    start = max(trace.pulse_end_s + 0.8, trace.time_s[-1] - 0.25)
    return trace.time_s >= start


def _baseline(trace: Trace, values: np.ndarray) -> float:
    return float(np.median(values[_baseline_mask(trace)]))


def _interp_relative(trace: Trace, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, trace.time_s, values - _baseline(trace, values))


def _trace_metrics(trace: Trace) -> dict[str, float]:
    q_rel = trace.position_rad - _baseline(trace, trace.position_rad)
    tau_rel = trace.effort_nm - _baseline(trace, trace.effort_nm)
    pulse = (trace.time_s >= trace.pulse_start_s) & (trace.time_s <= trace.pulse_end_s)
    tail = _tail_mask(trace)
    direction = trace.direction
    return {
        "samples": float(len(trace.time_s)),
        "sample_rate_hz": trace.sample_rate_hz,
        "pulse_start_s": trace.pulse_start_s,
        "pulse_end_s": trace.pulse_end_s,
        "pulse_width_s": trace.pulse_end_s - trace.pulse_start_s,
        "q_initial_rad": _baseline(trace, trace.position_rad),
        "q_pulse_peak_rad": float(np.max(direction * q_rel[pulse])),
        "q_tail_rad": float(np.median(direction * q_rel[tail])),
        "qd_pulse_peak_rad_s": float(np.max(np.abs(trace.velocity_rad_s[pulse]))),
        "tau_initial_nm": _baseline(trace, trace.effort_nm),
        "tau_delta_peak_nm": float(np.max(direction * tau_rel[pulse])),
        "tau_delta_min_nm": float(np.min(direction * tau_rel[pulse])),
        "tau_cmd_peak_nm": float(np.max(direction * trace.cmd_tau_recon_nm[pulse])),
    }


def paired_metrics(sim: Trace, real: Trace) -> dict[str, float]:
    sim_metrics = _trace_metrics(sim)
    real_metrics = _trace_metrics(real)
    end_s = min(sim.time_s[-1], real.time_s[-1])
    grid = np.linspace(0.0, end_s, int(np.floor(end_s * 1000.0)) + 1)
    sim_q = _interp_relative(sim, sim.position_rad, grid)
    real_q = _interp_relative(real, real.position_rad, grid)
    sim_peak = sim_metrics["q_pulse_peak_rad"]
    return {
        "initial_pose_delta_rad": real_metrics["q_initial_rad"] - sim_metrics["q_initial_rad"],
        "q_response_rmse_rad": float(np.sqrt(np.mean((sim_q - real_q) ** 2))),
        "peak_response_ratio_real_over_sim": real_metrics["q_pulse_peak_rad"] / sim_peak
        if abs(sim_peak) > EPS
        else float("nan"),
    }


def _shade_pulse(ax: plt.Axes, trace: Trace) -> None:
    ax.axvspan(trace.pulse_start_s, trace.pulse_end_s, color="#f4a261", alpha=0.16)


def _joint_axis(axes: np.ndarray, index: int) -> plt.Axes:
    return axes[index % 7, index // 7]


def plot_position_overlay(pairs: list[tuple[Trace, Trace]], out_dir: Path) -> None:
    fig, axes = plt.subplots(7, 2, figsize=(15, 20), sharex=True)
    for index, (sim, real) in enumerate(pairs):
        ax = _joint_axis(axes, index)
        _shade_pulse(ax, real)
        ax.plot(sim.time_s, np.degrees(sim.position_rad - _baseline(sim, sim.position_rad)),
                color="#2878b5", lw=1.2, label="MuJoCo")
        ax.plot(real.time_s, np.degrees(real.position_rad - _baseline(real, real.position_rad)),
                color="#d1495b", lw=1.2, label="real")
        ax.set_title(sim.name, fontsize=10)
        ax.grid(alpha=0.3)
        if index == 0:
            ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("time [s]")
        ax.tick_params(axis="x", labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("q - q₀ [deg]")
    fig.suptitle("Per-joint consistency-probe profile response (baseline-subtracted)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "position_overlay.png", dpi=160)
    plt.close(fig)


def plot_effort_overlay(pairs: list[tuple[Trace, Trace]], out_dir: Path) -> None:
    fig, axes = plt.subplots(7, 2, figsize=(15, 20), sharex=True)
    for index, (sim, real) in enumerate(pairs):
        ax = _joint_axis(axes, index)
        _shade_pulse(ax, real)
        ax.plot(sim.time_s, sim.effort_nm - _baseline(sim, sim.effort_nm),
                color="#2878b5", lw=1.1, label="MuJoCo τ_meas - τ₀")
        ax.plot(real.time_s, real.effort_nm - _baseline(real, real.effort_nm),
                color="#d1495b", lw=1.1, label="real τ_meas - τ₀")
        ax.plot(real.time_s, real.cmd_tau_recon_nm,
                color="#3a3a3a", lw=0.9, ls="--", label="raw PD reconstruction")
        ax.set_title(sim.name, fontsize=10)
        ax.grid(alpha=0.3)
        if index == 0:
            ax.legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel("time [s]")
        ax.tick_params(axis="x", labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Δτ [Nm]")
    fig.suptitle("Measured effort change and raw PD command", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "effort_overlay.png", dpi=160)
    plt.close(fig)


def plot_initial_pose(pairs: list[tuple[Trace, Trace]], out_dir: Path) -> None:
    labels = [sim.name.replace("_fr3_", " ").replace("_joint", " J") for sim, _ in pairs]
    deltas = np.asarray([
        _trace_metrics(real)["q_initial_rad"] - _trace_metrics(sim)["q_initial_rad"]
        for sim, real in pairs
    ])
    fig, ax = plt.subplots(figsize=(15, 5))
    color = np.where(deltas >= 0.0, "#457b9d", "#e76f51")
    ax.bar(np.arange(len(deltas)), np.degrees(deltas), color=color)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(deltas)), labels, rotation=35, ha="right")
    ax.set_ylabel("real q₀ - MuJoCo q₀ [deg]")
    ax.set_title("Initial-pose mismatch: current runs are not at the same operating point")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "initial_pose_delta.png", dpi=160)
    plt.close(fig)


def cross_coupling(traces: list[Trace]) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Return response of every measured joint to each single-joint input.

    Matrix row = measured joint and column = excited joint.  Values are the
    maximum absolute baseline-subtracted displacement normalized by the active
    joint's maximum displacement.  The diagonal is NaN because it is 1 by
    construction and would hide the off-axis scale in a heatmap.
    """
    matrix = np.full((len(JOINT_NAMES), len(JOINT_NAMES)), np.nan, dtype=np.float64)
    records: list[dict[str, float | str]] = []
    for trace in traces:
        active_index = JOINT_NAMES.index(trace.name)
        baseline = np.median(trace.all_position_rad[_baseline_mask(trace)], axis=0)
        peak = np.max(np.abs(trace.all_position_rad - baseline), axis=0)
        denominator = peak[active_index]
        if denominator <= EPS:
            raise ValueError(f"{trace.path} has zero active-joint response")
        ratios = peak / denominator
        ratios[active_index] = np.nan
        matrix[:, active_index] = ratios
        coupled_index = int(np.nanargmax(ratios))
        records.append(
            {
                "source": trace.source,
                "active_joint": trace.name,
                "largest_coupled_joint": JOINT_NAMES[coupled_index],
                "largest_coupled_displacement_rad": float(peak[coupled_index]),
                "active_displacement_rad": float(denominator),
                "largest_coupling_ratio": float(ratios[coupled_index]),
            }
        )
    return matrix, records


def plot_cross_coupling(pairs: list[tuple[Trace, Trace]], out_dir: Path) -> list[dict[str, float | str]]:
    sim_matrix, sim_records = cross_coupling([sim for sim, _ in pairs])
    real_matrix, real_records = cross_coupling([real for _, real in pairs])
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True, constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")
    for ax, matrix, title in (
        (axes[0], sim_matrix, "MuJoCo"),
        (axes[1], real_matrix, "real"),
    ):
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0.0, vmax=0.30, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(JOINT_NAMES)), JOINT_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(len(JOINT_NAMES)), JOINT_NAMES, fontsize=7)
        ax.set_xlabel("excited joint")
    axes[0].set_ylabel("measured joint")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.8)
    colorbar.set_label("max |Δq_measured| / max |Δq_active| (off-diagonal)")
    fig.suptitle("Off-axis joint coupling during each single-joint profile", fontweight="bold")
    fig.savefig(out_dir / "cross_coupling.png", dpi=160)
    plt.close(fig)
    return sim_records + real_records


def write_summary(
    pairs: list[tuple[Trace, Trace]],
    coupling_records: list[dict[str, float | str]],
    out_dir: Path,
) -> None:
    rows: list[dict[str, float | str]] = []
    for sim, real in pairs:
        row: dict[str, float | str] = {"joint": sim.name}
        row.update({f"mujoco_{key}": value for key, value in _trace_metrics(sim).items()})
        row.update({f"real_{key}": value for key, value in _trace_metrics(real).items()})
        row.update(paired_metrics(sim, real))
        rows.append(row)

    keys = list(rows[0])
    with (out_dir / "summary.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    response_ratio = np.asarray([
        float(row["peak_response_ratio_real_over_sim"]) for row in rows
    ])
    rmse_deg = np.degrees(np.asarray([float(row["q_response_rmse_rad"]) for row in rows]))
    initial_delta_deg = np.degrees(np.asarray([float(row["initial_pose_delta_rad"]) for row in rows]))
    with (out_dir / "cross_coupling.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(coupling_records[0]))
        writer.writeheader()
        writer.writerows(coupling_records)
    sim_coupling = np.asarray([
        float(row["largest_coupling_ratio"])
        for row in coupling_records
        if row["source"] == "mujoco"
    ])
    real_coupling = np.asarray([
        float(row["largest_coupling_ratio"])
        for row in coupling_records
        if row["source"] == "real"
    ])
    with (out_dir / "report.md").open("w") as report:
        report.write("# Dual-FR3 consistency-probe comparison\n\n")
        report.write(f"Matched trials: **{len(rows)}** / {len(JOINT_NAMES)} joints.\n\n")
        report.write("| Metric | Value |\n| --- | ---: |\n")
        report.write(f"| Median real / MuJoCo profile displacement | {np.median(response_ratio):.3f} |\n")
        report.write(f"| Mean relative-position RMSE | {np.mean(rmse_deg):.3f} deg |\n")
        report.write(f"| Maximum initial-pose mismatch | {np.max(np.abs(initial_delta_deg)):.3f} deg |\n")
        report.write(f"| Median largest off-axis coupling: MuJoCo | {np.median(sim_coupling):.3f} |\n")
        report.write(f"| Median largest off-axis coupling: real | {np.median(real_coupling):.3f} |\n")
        report.write("\nThe response ratio is descriptive only when both trials start at the same"
                     " 14-joint pose and use the same payload/contact/gravity conditions.\n")


def main() -> None:
    args = _parser().parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mujoco = load_source(root, "mujoco")
    real = load_source(root, "real")
    missing = sorted(set(mujoco).symmetric_difference(real))
    if missing:
        raise ValueError(f"Unpaired trials: {', '.join(missing)}")
    missing_expected = [name for name in JOINT_NAMES if name not in mujoco]
    if missing_expected:
        raise ValueError(f"Missing expected joint trials: {', '.join(missing_expected)}")

    pairs = [(mujoco[name], real[name]) for name in JOINT_NAMES]
    plot_position_overlay(pairs, out_dir)
    plot_effort_overlay(pairs, out_dir)
    plot_initial_pose(pairs, out_dir)
    coupling_records = plot_cross_coupling(pairs, out_dir)
    write_summary(pairs, coupling_records, out_dir)
    print(f"Wrote comparison artifacts to {out_dir}")


if __name__ == "__main__":
    main()
