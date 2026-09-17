from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACTION_COLUMNS = tuple(f"action_{index}" for index in range(14))
NOISE_COLUMNS = tuple(f"noise_{index}" for index in range(14))
TARGET_COLUMNS = ("target_center_x", "target_center_y", "target_center_z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReachActionTrajectory:
    """A finite sequence of 20 Hz raw Reach actions and logging targets."""

    path: Path
    source_elapsed_s: np.ndarray
    target_centers: np.ndarray
    actions: np.ndarray
    noise: np.ndarray
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "ReachActionTrajectory":
        trajectory_path = Path(path).expanduser().resolve()
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Reach trajectory does not exist: {trajectory_path}")

        with trajectory_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            required = {"step", "source_elapsed_s", *TARGET_COLUMNS, *ACTION_COLUMNS}
            missing = sorted(required - fieldnames)
            if missing:
                raise ValueError(
                    f"{trajectory_path} is missing columns: {', '.join(missing)}"
                )
            rows = list(reader)

        if not rows:
            raise ValueError(f"Reach trajectory is empty: {trajectory_path}")

        steps = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
        expected_steps = np.arange(len(rows), dtype=np.int64)
        if not np.array_equal(steps, expected_steps):
            raise ValueError("Reach trajectory step values must be contiguous and start at zero")

        source_elapsed_s = np.asarray(
            [float(row["source_elapsed_s"]) for row in rows], dtype=np.float64
        )
        target_centers = np.asarray(
            [[float(row[name]) for name in TARGET_COLUMNS] for row in rows],
            dtype=np.float64,
        )
        actions = np.asarray(
            [[float(row[name]) for name in ACTION_COLUMNS] for row in rows],
            dtype=np.float32,
        )
        if set(NOISE_COLUMNS).issubset(fieldnames):
            noise = np.asarray(
                [[float(row[name]) for name in NOISE_COLUMNS] for row in rows],
                dtype=np.float32,
            )
        else:
            noise = np.zeros_like(actions)

        arrays = (source_elapsed_s, target_centers, actions, noise)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError(f"Reach trajectory contains NaN or Inf: {trajectory_path}")
        if len(source_elapsed_s) > 1 and np.any(np.diff(source_elapsed_s) <= 0.0):
            raise ValueError("source_elapsed_s must be strictly increasing")
        if np.any(np.abs(actions) > 1.0 + 1.0e-6):
            raise ValueError("Reach trajectory raw actions must be within [-1, 1]")

        return cls(
            path=trajectory_path,
            source_elapsed_s=source_elapsed_s,
            target_centers=target_centers,
            actions=actions,
            noise=noise,
            sha256=sha256_file(trajectory_path),
        )

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def action_at(self, index: int, noise_scale: float = 1.0) -> np.ndarray:
        """Return one raw action with the stored perturbation rescaled."""
        if not np.isfinite(noise_scale) or noise_scale < 0.0:
            raise ValueError("noise_scale must be finite and non-negative")
        nominal = self.actions[index] - self.noise[index]
        return nominal + np.float32(noise_scale) * self.noise[index]

    @property
    def duration_s(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(self.source_elapsed_s[-1] - self.source_elapsed_s[0])
