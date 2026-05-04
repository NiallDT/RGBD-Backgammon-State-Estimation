from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time

from backgammon_types import BoardState, StateEvent


@dataclass
class EventInferenceConfig:
    emit_initial_state: bool = True
    emit_region_count_changes: bool = True
    emit_dice_changes: bool = True
    emit_cube_changes: bool = True


class BoardStateEventInferer:
    """
    Lightweight, non-CNN event inference layer.

    This does not attempt full legal move reconstruction yet. It turns changes
    between committed BoardState objects into coarse events that can be logged
    and inspected during experiments:
      - initial_state
      - region_count_changed
      - dice_changed
      - cube_changed

    A later dissertation extension can replace or extend this with a legal
    move/cube-action inference engine.
    """

    def __init__(self, config: Optional[EventInferenceConfig] = None) -> None:
        self.config = config or EventInferenceConfig()
        self.previous_state: Optional[BoardState] = None

    @staticmethod
    def _normalise_counts(counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for region, colours in counts.items():
            out[region] = {
                "light": int(colours.get("light", 0)),
                "dark": int(colours.get("dark", 0)),
            }
        return out

    @staticmethod
    def _region_count_deltas(previous: BoardState, current: BoardState) -> Dict[str, Dict[str, int]]:
        prev = BoardStateEventInferer._normalise_counts(previous.region_counts)
        cur = BoardStateEventInferer._normalise_counts(current.region_counts)
        regions = sorted(set(prev) | set(cur))
        deltas: Dict[str, Dict[str, int]] = {}
        for region in regions:
            light_delta = cur.get(region, {}).get("light", 0) - prev.get(region, {}).get("light", 0)
            dark_delta = cur.get(region, {}).get("dark", 0) - prev.get(region, {}).get("dark", 0)
            if light_delta or dark_delta:
                deltas[region] = {"light": light_delta, "dark": dark_delta}
        return deltas

    def infer(self, previous: Optional[BoardState], current: BoardState) -> List[StateEvent]:
        now = current.timestamp or time.time()
        events: List[StateEvent] = []

        if previous is None:
            if self.config.emit_initial_state:
                events.append(
                    StateEvent(
                        event_type="initial_state",
                        description="Initial committed board state observed.",
                        timestamp=now,
                        confidence=current.confidence,
                        payload={
                            "region_counts": current.region_counts,
                            "dice": list(current.dice),
                            "cube_value": current.cube_value,
                        },
                    )
                )
            return events

        if self.config.emit_region_count_changes:
            deltas = self._region_count_deltas(previous, current)
            if deltas:
                events.append(
                    StateEvent(
                        event_type="region_count_changed",
                        description="One or more checker region counts changed.",
                        timestamp=now,
                        confidence=current.confidence,
                        payload={"deltas": deltas},
                    )
                )

        if self.config.emit_dice_changes and tuple(previous.dice) != tuple(current.dice):
            events.append(
                StateEvent(
                    event_type="dice_changed",
                    description=f"Dice changed from {previous.dice} to {current.dice}.",
                    timestamp=now,
                    confidence=current.confidence,
                    payload={"previous": list(previous.dice), "current": list(current.dice)},
                )
            )

        if self.config.emit_cube_changes and previous.cube_value != current.cube_value:
            events.append(
                StateEvent(
                    event_type="cube_changed",
                    description=f"Cube value changed from {previous.cube_value} to {current.cube_value}.",
                    timestamp=now,
                    confidence=current.confidence,
                    payload={"previous": previous.cube_value, "current": current.cube_value},
                )
            )

        return events

    def update(self, current: BoardState) -> List[StateEvent]:
        events = self.infer(self.previous_state, current)
        self.previous_state = current
        return events

    def reset(self) -> None:
        self.previous_state = None


__all__ = ["EventInferenceConfig", "BoardStateEventInferer"]
