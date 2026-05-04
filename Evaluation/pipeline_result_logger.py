from __future__ import annotations

from dataclasses import asdict, is_dataclass, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time

import numpy as np

from backgammon_types import PipelineResult, StateEvent


@dataclass
class PipelineLogConfig:
    root_dir: str = "pipeline_logs"
    session_name: Optional[str] = None
    write_intermediates_summary: bool = True


class PipelineResultLogger:
    """Writes PipelineResult summaries and StateEvent records to JSONL files."""

    def __init__(self, config: Optional[PipelineLogConfig] = None) -> None:
        self.config = config or PipelineLogConfig()
        session = self.config.session_name or time.strftime("pipeline_%Y%m%d_%H%M%S")
        self.session_dir = Path(self.config.root_dir) / session
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.session_dir / "pipeline_results.jsonl"
        self.events_path = self.session_dir / "events.jsonl"
        self.session_path = self.session_dir / "session.json"
        self._results_file = self.results_path.open("a", encoding="utf-8")
        self._events_file = self.events_path.open("a", encoding="utf-8")
        self.session_path.write_text(json.dumps({"session": session, "created": time.time()}, indent=2), encoding="utf-8")

    def close(self) -> None:
        self._results_file.close()
        self._events_file.close()

    def __enter__(self) -> "PipelineResultLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if is_dataclass(value):
            return PipelineResultLogger._safe(asdict(value))
        if isinstance(value, dict):
            return {str(k): PipelineResultLogger._safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PipelineResultLogger._safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def result_summary(result: PipelineResult) -> Dict[str, Any]:
        state = result.temporal.fused_state if result.temporal is not None else None
        return {
            "timestamp": time.time(),
            "committed": result.committed,
            "state_changed": result.state_changed,
            "board_lock": {
                "valid": bool(result.board_lock.valid),
                "method": result.board_lock.method,
                "confidence": result.board_lock.confidence,
            },
            "validation": None if result.validation is None else {
                "valid": result.validation.valid,
                "errors": result.validation.errors,
                "warnings": result.validation.warnings,
                "confidence": result.validation.confidence,
            },
            "state": None if state is None else {
                "region_counts": state.region_counts,
                "dice": list(state.dice),
                "cube_value": state.cube_value,
                "confidence": state.confidence,
                "timestamp": state.timestamp,
            },
            "piece_confidence": None if result.pieces is None else result.pieces.confidence,
            "dice_cube_confidence": None if result.dice_cube is None else result.dice_cube.confidence,
            "debug": result.debug,
        }

    def record(self, result: PipelineResult) -> None:
        summary = self._safe(self.result_summary(result))
        self._results_file.write(json.dumps(summary) + "\n")
        self._results_file.flush()
        for event in result.events:
            self._events_file.write(json.dumps(self._safe(event)) + "\n")
        self._events_file.flush()


__all__ = ["PipelineLogConfig", "PipelineResultLogger"]
