"""
Structured JSON-lines diagnostic logger for the instrumented simulation.

Logs every tick's NPC state plus discrete events (slot transitions,
subtask changes, goal progress, etc.) to a .jsonl file for post-run
analysis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.npc.models import NPC
    from tests.simulation.goals import NPCGoal


class DiagnosticLogger:
    """Writes structured log events to a JSONL file."""

    def __init__(self, output_path: str | Path):
        self._path = Path(output_path)
        self._fh = open(self._path, "w")  # noqa: SIM115
        self._event_count = 0

    def close(self) -> None:
        self._fh.close()

    @property
    def event_count(self) -> int:
        return self._event_count

    def _write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._event_count += 1

    def log_tick_state(
        self,
        tick: int,
        game_time: str,
        npc: NPC,
        goal: NPCGoal | None = None,
    ) -> None:
        """Log full NPC state snapshot for this tick."""
        step = goal.current_step if goal else None
        rng_hash = hashlib.md5(
            str(npc._rng.getstate()).encode(),
        ).hexdigest()[:8]

        self._write({
            "tick": tick,
            "game_time": game_time,
            "npc_id": npc.npc_id,
            "npc_name": npc.name,
            "event_type": "TICK_STATE",
            "data": {
                "activity": npc.activity.value,
                "position": [round(npc.x, 1), round(npc.z, 1)],
                "tile": [npc.tile_x, npc.tile_z],
                "subtask": (
                    npc.current_subtask.description
                    if npc.current_subtask else None
                ),
                "subtask_time_remaining": round(
                    npc.subtask_time_remaining, 1,
                ),
                "queue_depth": len(npc.subtask_queue),
                "path_length": len(npc.current_path),
                "action_desc": npc.current_action_description,
                "energy": round(npc.energy, 3),
                "hunger": round(npc.hunger, 3),
                "goal": goal.description if goal else None,
                "goal_step": goal.current_step_index if goal else None,
                "goal_step_desc": step.description if step else None,
                "goal_progress": (
                    round(goal.progress_fraction, 3) if goal else None
                ),
                "rng_state_hash": rng_hash,
            },
        })

    def log_event(
        self,
        tick: int,
        game_time: str,
        npc_id: str,
        npc_name: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Log a discrete event."""
        self._write({
            "tick": tick,
            "game_time": game_time,
            "npc_id": npc_id,
            "npc_name": npc_name,
            "event_type": event_type,
            "data": data or {},
        })
