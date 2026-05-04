from __future__ import annotations

from pathlib import Path
from typing import Optional
import argparse
import sys

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent if _MODULE_DIR.name in {"Acquisition", "Geometric + Visual Preprocessing", "Interpretation", "Evaluation"} else _MODULE_DIR
for _rel in ("", "Acquisition", "Geometric + Visual Preprocessing", "Interpretation", "Evaluation"):
    _p = _PROJECT_ROOT / _rel if _rel else _PROJECT_ROOT
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from keyframe_gate import OakSRKeyframeGate
from board_state_pipeline import BoardStatePipeline
from pipeline_result_logger import PipelineLogConfig, PipelineResultLogger


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full board-state pipeline on live keyframes and optionally log results.")
    p.add_argument("--frames", type=int, default=300, help="Maximum camera frames to inspect.")
    p.add_argument("--process-all", action="store_true", help="Process every packet instead of accepted keyframes only.")
    p.add_argument("--log", action="store_true", help="Write pipeline_results.jsonl and events.jsonl.")
    p.add_argument("--log-dir", default="pipeline_logs", help="Output directory for logs.")
    p.add_argument("--print-every", type=int, default=1, help="Print every N processed results.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    gate = OakSRKeyframeGate().start()
    pipeline = BoardStatePipeline()
    logger = PipelineResultLogger(PipelineLogConfig(root_dir=args.log_dir)) if args.log else None
    processed = 0

    try:
        for _ in range(args.frames):
            packet = gate.get_packet(block=True)
            if packet is None:
                continue
            if not args.process_all and not packet.keyframe:
                continue

            result = pipeline.process_keyframe_packet(packet)
            processed += 1
            if logger is not None:
                logger.record(result)

            if processed % max(args.print_every, 1) == 0:
                print("-" * 72)
                print(f"frame={packet.frame_index} committed={result.committed} changed={result.state_changed} events={len(result.events)}")
                print(f"lock={result.board_lock.valid} method={result.board_lock.method} conf={result.board_lock.confidence:.2f}")
                if result.validation is not None:
                    print(f"validation={result.validation.valid} warnings={result.validation.warnings} errors={result.validation.errors}")
                if result.temporal is not None:
                    state = result.temporal.fused_state
                    print(f"state_conf={state.confidence:.2f} dice={state.dice} cube={state.cube_value}")
                    # Print only occupied regions to keep output readable.
                    occupied = {
                        r: c for r, c in state.region_counts.items()
                        if c.get("light", 0) or c.get("dark", 0)
                    }
                    print("occupied:", occupied)
                for event in result.events:
                    print(f"event: {event.event_type} | {event.description}")

    except KeyboardInterrupt:
        pass
    finally:
        gate.stop()
        if logger is not None:
            logger.close()
            print(f"Logs written under: {logger.session_dir}")


if __name__ == "__main__":
    main()
