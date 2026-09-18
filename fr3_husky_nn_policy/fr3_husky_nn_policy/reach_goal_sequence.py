"""Validation helpers for timed dual-FR3 Reach target sequences."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def load_reach_goal_sequence(path: str) -> list[dict[str, object]]:
    """Load the JSON format used by dual_fr3_lab's Reach sequence player."""

    sequence_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(sequence_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"failed to load goal sequence '{sequence_path}': {error}"
        ) from error

    goals = payload.get("goals") if isinstance(payload, dict) else payload
    if not isinstance(goals, list) or not goals:
        raise ValueError(
            "goal sequence must be a non-empty JSON list or an object with a 'goals' list"
        )

    validated: list[dict[str, object]] = []
    for index, goal in enumerate(goals):
        if not isinstance(goal, dict):
            raise ValueError(f"goal {index} must be a JSON object")
        position = goal.get("position")
        duration_s = goal.get("duration_s")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in position
            )
        ):
            raise ValueError(f"goal {index} position must contain three finite numbers")
        if (
            not isinstance(duration_s, (int, float))
            or isinstance(duration_s, bool)
            or not math.isfinite(duration_s)
            or duration_s <= 0.0
        ):
            raise ValueError(f"goal {index} duration_s must be a positive finite number")
        label = goal.get("label", f"goal_{index}")
        if not isinstance(label, str):
            raise ValueError(f"goal {index} label must be a string")
        validated.append(
            {
                "position": [float(value) for value in position],
                "duration_s": float(duration_s),
                "label": label,
            }
        )
    return validated


class ReachGoalSequencePlayer:
    """Advance validated Reach goals against an absolute monotonic deadline."""

    def __init__(self, goals: list[dict[str, object]], loop: bool = False):
        if not goals:
            raise ValueError("goal sequence player requires at least one goal")
        self.goals = goals
        self.loop = bool(loop)
        self.index = -1
        self.deadline_ns = 0
        self.completed = False

    @staticmethod
    def _duration_ns(goal: dict[str, object]) -> int:
        # The loader guarantees a positive finite duration. Keep even a
        # sub-nanosecond duration progressing instead of producing a zero-step
        # deadline and an infinite loop.
        return max(1, int(round(float(goal["duration_s"]) * 1.0e9)))

    @property
    def current_goal(self) -> Optional[dict[str, object]]:
        if self.index < 0 or self.completed:
            return None
        return self.goals[self.index]

    def reset(self):
        self.index = -1
        self.deadline_ns = 0
        self.completed = False

    def start(self, now_ns: int) -> dict[str, object]:
        self.index = 0
        self.completed = False
        self.deadline_ns = int(now_ns) + self._duration_ns(self.goals[0])
        return self.goals[0]

    def advance(self, now_ns: int) -> tuple[Optional[dict[str, object]], int]:
        """Return the active goal and number of elapsed goal boundaries.

        Deadlines are advanced from the preceding deadline rather than from
        ``now_ns`` so ROS timer jitter does not accumulate across the sequence.
        A ``None`` goal means a non-looping sequence has completed.
        """

        if self.index < 0:
            raise RuntimeError("goal sequence player has not been started")
        if self.completed:
            return None, 0

        transitions = 0
        now_ns = int(now_ns)
        while now_ns >= self.deadline_ns:
            next_index = self.index + 1
            if next_index >= len(self.goals):
                if not self.loop:
                    self.completed = True
                    return None, transitions + 1
                next_index = 0
            self.index = next_index
            self.deadline_ns += self._duration_ns(self.goals[self.index])
            transitions += 1
        return self.goals[self.index], transitions
