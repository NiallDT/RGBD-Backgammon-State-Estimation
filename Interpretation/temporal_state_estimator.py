from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from backgammon_types import BoardState, TemporalEstimate


def _mode_or_default(values: Iterable[Optional[int]], default: Optional[int] = None) -> Optional[int]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return default
    return Counter(filtered).most_common(1)[0][0]


def _state_signature(state: BoardState) -> Tuple:
    region_items = tuple(
        sorted((region, counts.get("light", 0), counts.get("dark", 0)) for region, counts in state.region_counts.items())
    )
    return region_items, state.dice, state.cube_value


class TemporalStateEstimator:
    """
    Fuses candidate board states over a short rolling window.

    The fusion strategy is intentionally simple: mode/median-like consensus over
    recent keyframes. This makes it easy to reason about the output during the
    data-collection phase and replace later with a learned temporal model.
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = max(1, window_size)
        self.history: Deque[BoardState] = deque(maxlen=self.window_size)
        self.last_committed_state: Optional[BoardState] = None

    def _fuse_region_counts(self) -> Dict[str, Dict[str, int]]:
        region_names = sorted({name for state in self.history for name in state.region_counts.keys()})
        fused: Dict[str, Dict[str, int]] = {}

        for region_name in region_names:
            light_values = [state.region_counts.get(region_name, {}).get("light", 0) for state in self.history]
            dark_values = [state.region_counts.get(region_name, {}).get("dark", 0) for state in self.history]

            light_mode = Counter(light_values).most_common(1)[0][0] if light_values else 0
            dark_mode = Counter(dark_values).most_common(1)[0][0] if dark_values else 0
            fused[region_name] = {"light": int(light_mode), "dark": int(dark_mode)}

        return fused

    def update(self, candidate_state: BoardState) -> TemporalEstimate:
        self.history.append(candidate_state)

        fused = BoardState(
            region_counts=self._fuse_region_counts(),
            dice=(
                _mode_or_default([s.dice[0] for s in self.history]),
                _mode_or_default([s.dice[1] for s in self.history]),
            ),
            cube_value=_mode_or_default([s.cube_value for s in self.history]),
            confidence=sum(s.confidence for s in self.history) / max(len(self.history), 1),
            timestamp=candidate_state.timestamp,
            metadata={"history_size": len(self.history)},
        )

        changed = True
        if self.last_committed_state is not None:
            changed = _state_signature(fused) != _state_signature(self.last_committed_state)

        return TemporalEstimate(
            candidate_state=candidate_state,
            fused_state=fused,
            changed_vs_last_commit=changed,
            history_size=len(self.history),
        )

    def commit(self, state: BoardState) -> None:
        self.last_committed_state = state
