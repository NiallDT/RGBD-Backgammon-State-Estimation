from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import time

import cv2
import numpy as np

from streams_module import OakSRStreams, depth_to_grayscale
from keyframe_gate import OakSRKeyframeGate


@dataclass
class RecorderConfig:
    root_dir: str = "recordings"
    session_name: Optional[str] = None
    fps: float = 30.0
    view_size: tuple[int, int] = (640, 400)
    save_right: bool = False
    save_depth_npy: bool = True
    save_depth_png: bool = True
    use_keyframe_gate: bool = False
    keyframes_only: bool = False


class RGBDRecorder:
    """
    Records RGB-D data for model training.

    Output layout:
      recordings/<session_name>/
        rgb/
        depth_raw_npy/
        depth_gray_png/
        right/                  # optional
        metadata/
          frames.jsonl
          session.json
    """

    def __init__(self, config: Optional[RecorderConfig] = None) -> None:
        self.config = config or RecorderConfig()
        session_name = self.config.session_name or time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = Path(self.config.root_dir) / session_name

        (self.session_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "metadata").mkdir(parents=True, exist_ok=True)

        if self.config.save_depth_npy:
            (self.session_dir / "depth_raw_npy").mkdir(parents=True, exist_ok=True)
        if self.config.save_depth_png:
            (self.session_dir / "depth_gray_png").mkdir(parents=True, exist_ok=True)
        if self.config.save_right:
            (self.session_dir / "right").mkdir(parents=True, exist_ok=True)

        self.frame_index = 0
        self.metadata_path = self.session_dir / "metadata" / "frames.jsonl"

        if self.config.use_keyframe_gate:
            self.gate = OakSRKeyframeGate(fps=self.config.fps, view_size=self.config.view_size)
            self.streams = None
        else:
            self.gate = None
            self.streams = OakSRStreams(
                enable_left=True,
                enable_right=self.config.save_right,
                enable_depth=True,
                fps=self.config.fps,
                view_size=self.config.view_size,
                stereo_size=self.config.view_size,
            )

    def start(self) -> "RGBDRecorder":
        if self.gate is not None:
            self.gate.start()
        if self.streams is not None:
            self.streams.start()

        session_info = {
            "root_dir": str(self.session_dir),
            "fps": self.config.fps,
            "view_size": list(self.config.view_size),
            "save_right": self.config.save_right,
            "save_depth_npy": self.config.save_depth_npy,
            "save_depth_png": self.config.save_depth_png,
            "use_keyframe_gate": self.config.use_keyframe_gate,
            "keyframes_only": self.config.keyframes_only,
            "created_at": time.time(),
        }
        with open(self.session_dir / "metadata" / "session.json", "w", encoding="utf-8") as f:
            json.dump(session_info, f, indent=2)

        return self

    def stop(self) -> None:
        if self.gate is not None:
            self.gate.stop()
        if self.streams is not None:
            self.streams.stop()

    def __enter__(self) -> "RGBDRecorder":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _write_metadata(self, record: dict) -> None:
        with open(self.metadata_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def record_next(self) -> Optional[dict]:
        timestamp = time.time()
        idx = self.frame_index

        if self.gate is not None:
            packet = self.gate.get_packet(block=True)
            if packet is None:
                return None
            if self.config.keyframes_only and not packet.keyframe:
                return None

            left = packet.left_raw
            depth_raw = packet.depth_raw
            depth_gray = packet.depth_gray
            right = None
            keyframe = packet.keyframe
        else:
            left = self.streams.get_left_frame(block=True, use_cached=False)
            depth_raw = self.streams.get_depth_frame(block=True, use_cached=False)
            right = self.streams.get_right_frame(block=False, use_cached=True) if self.config.save_right else None

            if left is None or depth_raw is None:
                return None

            depth_gray = depth_to_grayscale(depth_raw)
            keyframe = None

        stem = f"{idx:06d}"
        cv2.imwrite(str(self.session_dir / "rgb" / f"{stem}.png"), left)

        if self.config.save_depth_npy:
            np.save(self.session_dir / "depth_raw_npy" / f"{stem}.npy", depth_raw)
        if self.config.save_depth_png:
            cv2.imwrite(str(self.session_dir / "depth_gray_png" / f"{stem}.png"), depth_gray)
        if self.config.save_right and right is not None:
            cv2.imwrite(str(self.session_dir / "right" / f"{stem}.png"), right)

        record = {
            "frame_index": idx,
            "timestamp": timestamp,
            "rgb_path": f"rgb/{stem}.png",
            "depth_npy_path": f"depth_raw_npy/{stem}.npy" if self.config.save_depth_npy else None,
            "depth_png_path": f"depth_gray_png/{stem}.png" if self.config.save_depth_png else None,
            "right_path": f"right/{stem}.png" if self.config.save_right and right is not None else None,
            "keyframe": keyframe,
        }
        self._write_metadata(record)
        self.frame_index += 1
        return record

    def record_frames(self, max_frames: int) -> list[dict]:
        records = []
        while len(records) < max_frames:
            record = self.record_next()
            if record is not None:
                records.append(record)
        return records
