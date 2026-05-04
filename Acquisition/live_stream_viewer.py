from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple
import argparse
import json
import re
import sys
import time
from datetime import datetime

import cv2
import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent if _MODULE_DIR.name in {"Acquisition", "Geometric + Visual Preprocessing", "Interpretation"} else _MODULE_DIR
for _rel in ("", "Acquisition", "Geometric + Visual Preprocessing", "Interpretation"):
    _path = _PROJECT_ROOT / _rel if _rel else _PROJECT_ROOT
    if _path.exists():
        _path_str = str(_path)
        if _path_str not in sys.path:
            sys.path.insert(0, _path_str)

from streams_module import OakSRStreams, depth_to_grayscale
from keyframe_gate import OakSRKeyframeGate, KeyframePacket

try:
    from board_registration import BoardRegistrar
    from perspective_rectification import PerspectiveRectifier
    from image_normalisation import BoardNormaliser
    from point_tray_segmentation import PointTraySegmenter
except Exception:  # pragma: no cover - lets acquisition-only viewing still work
    BoardRegistrar = None
    PerspectiveRectifier = None
    BoardNormaliser = None
    PointTraySegmenter = None


ViewMode = Literal["streams", "gate", "preprocess", "all"]
Size = Tuple[int, int]


@dataclass
class LiveViewConfig:
    """Configuration for the live diagnostic stream viewer."""

    mode: ViewMode = "all"
    fps: float = 30.0
    view_size: Size = (640, 400)
    grid_cell_size: Size = (420, 260)
    grid_columns: int = 3
    window_name: str = "OAK-D SR live stream viewer"

    motion_threshold: float = 2.5
    hand_near_mm: int = 450
    max_near_ratio: float = 0.03
    max_invalid_ratio: float = 0.80

    enable_preprocessing: bool = True
    min_piece_height_mm: float = 3.0
    max_piece_height_mm: float = 40.0
    min_piece_area_px: int = 80
    height_max_mm: float = 35.0

    auto_baseline: bool = True
    auto_baseline_frames: int = 30
    baseline_min_valid_ratio: float = 0.20

    show_help: bool = True

    record: bool = False
    record_dir: Path = Path("troubleshooting_recordings")
    record_every: int = 1
    record_dashboard: bool = True
    record_views: bool = True
    record_raw_npz: bool = False
    record_codec: str = "mp4v"


def parse_size(text: str) -> Size:
    """Parse a size string such as '640x400' into (width, height)."""
    if "x" not in text.lower():
        raise argparse.ArgumentTypeError("Size must be written as WIDTHxHEIGHT, e.g. 640x400")
    w_str, h_str = text.lower().split("x", 1)
    return int(w_str), int(h_str)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/mask/float images into uint8 BGR for display."""
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    arr = image
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    elif np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        if not np.any(finite):
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            lo = float(np.percentile(arr[finite], 2))
            hi = float(np.percentile(arr[finite], 98))
            if hi <= lo:
                hi = lo + 1.0
            arr = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.copy()

    raise ValueError(f"Unsupported image shape for display: {arr.shape}")


def resize_letterbox(image_bgr: np.ndarray, size_wh: Size) -> np.ndarray:
    """Resize image to fit inside size while preserving aspect ratio."""
    target_w, target_h = size_wh
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def add_label(image_bgr: np.ndarray, label: str, *, ok: Optional[bool] = None) -> np.ndarray:
    """Add a small label strip to a display image."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    strip_h = 28
    cv2.rectangle(out, (0, 0), (w, strip_h), (0, 0, 0), -1)

    colour = (230, 230, 230)
    if ok is True:
        colour = (0, 220, 0)
    elif ok is False:
        colour = (0, 0, 255)

    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)
    return out


def sanitise_filename(text: str) -> str:
    """Make a stream/view label safe for use as a filename."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "view"


def render_view_cell(label: str, image: np.ndarray, ok: Optional[bool], cell_size: Size) -> np.ndarray:
    """Render one named view exactly like a dashboard cell."""
    bgr = ensure_bgr(image)
    cell = resize_letterbox(bgr, cell_size)
    return add_label(cell, label, ok=ok)


def make_grid(items: Iterable[Tuple[str, np.ndarray, Optional[bool]]], *, cell_size: Size, columns: int) -> np.ndarray:
    """Build a labelled image grid from named views."""
    cells: List[np.ndarray] = []
    for label, image, ok in items:
        cells.append(render_view_cell(label, image, ok, cell_size))

    if not cells:
        w, h = cell_size
        return np.zeros((h, w, 3), dtype=np.uint8)

    columns = max(1, int(columns))
    rows = int(np.ceil(len(cells) / columns))
    w, h = cell_size
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    padded = cells + [blank] * (rows * columns - len(cells))
    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.hstack(padded[r * columns : (r + 1) * columns]))
    return np.vstack(row_imgs)


def draw_gate_status(frame_bgr: np.ndarray, packet: KeyframePacket) -> np.ndarray:
    """Overlay keyframe-gate scores onto an RGB frame."""
    out = frame_bgr.copy()
    colour = (0, 255, 0) if packet.keyframe else ((0, 200, 255) if packet.stable else (0, 0, 255))

    lines = [
        f"frame={packet.frame_index} stable={packet.stable} keyframe={packet.keyframe}",
        f"occluded={packet.occluded} stable_count={packet.stable_count}",
        f"motion={packet.motion_score:.2f} rgb={packet.rgb_motion_score:.2f} depth={packet.depth_motion_score:.2f}",
        f"near={packet.near_ratio:.3f} invalid={packet.invalid_ratio:.3f} novelty={packet.novelty_score:.2f}",
    ]

    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 2, cv2.LINE_AA)
        y += 24
    return out


def draw_board_lock(frame_bgr: np.ndarray, lock) -> np.ndarray:
    """Draw board-registration corners on a copy of the input frame."""
    out = frame_bgr.copy()
    if lock is None or not getattr(lock, "valid", False):
        cv2.putText(out, "board lock: invalid", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return out

    corners = lock.corners_xy.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    for idx, pt in enumerate(corners.reshape(-1, 2)):
        cv2.circle(out, tuple(pt), 5, (0, 255, 255), -1)
        cv2.putText(out, str(idx), tuple(pt + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(
        out,
        f"board lock: {lock.method} conf={lock.confidence:.2f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def build_piece_height_mask(
    height_map_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_height_mm: float,
    max_height_mm: float,
    min_area_px: int,
) -> np.ndarray:
    """Create a simple black/white pre-CNN piece candidate mask from height above board."""
    mask = (
        (height_map_mm >= min_height_mm)
        & (height_map_mm <= max_height_mm)
        & (valid_mask > 0)
    ).astype(np.uint8) * 255

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            cleaned[labels == label] = 255
    return cleaned


def blend_overlay(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a sparse overlay onto a base image."""
    if overlay_bgr.shape[:2] != base_bgr.shape[:2]:
        overlay_bgr = cv2.resize(overlay_bgr, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.any(overlay_bgr > 0, axis=2)
    out = base_bgr.copy()
    out[mask] = cv2.addWeighted(base_bgr, 1.0 - alpha, overlay_bgr, alpha, 0)[mask]
    return out



class EmptyBoardBaseline:
    """Maintains an empty-board depth baseline and produces height/mask diagnostics."""

    def __init__(
        self,
        *,
        frames_required: int = 30,
        min_valid_ratio: float = 0.20,
        min_piece_height_mm: float = 3.0,
        max_piece_height_mm: float = 40.0,
        min_piece_area_px: int = 80,
        height_max_mm: float = 35.0,
    ) -> None:
        self.frames_required = max(1, int(frames_required))
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_piece_height_mm = float(min_piece_height_mm)
        self.max_piece_height_mm = float(max_piece_height_mm)
        self.min_piece_area_px = int(min_piece_area_px)
        self.height_max_mm = float(height_max_mm)
        self.samples: List[np.ndarray] = []
        self.baseline: Optional[np.ndarray] = None
        self.ready = False

    def reset(self) -> None:
        self.samples.clear()
        self.baseline = None
        self.ready = False

    def clear_samples_only(self) -> None:
        self.samples.clear()

    def add_empty_frame(self, depth_mm: np.ndarray) -> None:
        if self.ready:
            return
        valid_ratio = float(np.count_nonzero(depth_mm)) / float(depth_mm.size) if depth_mm.size else 0.0
        if valid_ratio < self.min_valid_ratio:
            return
        self.samples.append(depth_mm.copy())
        if len(self.samples) >= self.frames_required:
            stack = []
            for sample in self.samples:
                f = sample.astype(np.float32)
                f[f == 0] = np.nan
                stack.append(f)
            arr = np.stack(stack, axis=0)
            baseline = np.nanmedian(arr, axis=0)
            baseline[np.isnan(baseline)] = 0
            self.baseline = baseline.astype(np.float32)
            self.ready = True
            self.samples.clear()
            valid = 100.0 * float(np.count_nonzero(self.baseline)) / float(self.baseline.size)
            print(f"Empty-board baseline captured. Valid baseline pixels: {valid:.1f}%")

    def compute(self, depth_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.baseline is None:
            h, w = depth_mm.shape[:2]
            blank = np.zeros((h, w), dtype=np.uint8)
            return np.zeros((h, w), dtype=np.float32), blank, blank, blank

        current = depth_mm.astype(np.float32)
        baseline = self.baseline
        if current.shape != baseline.shape:
            baseline = cv2.resize(baseline, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_NEAREST)

        current_valid = current > 0
        baseline_valid = baseline > 0
        overlap = current_valid & baseline_valid

        height = baseline - current
        height[height < 0] = 0
        height[~overlap] = 0
        height[np.isnan(height)] = 0

        height_uint8 = np.clip(height, 0, max(1.0, self.height_max_mm)).astype(np.float32)
        height_uint8 = (height_uint8 * 255.0 / max(1.0, self.height_max_mm)).astype(np.uint8)
        height_uint8[~overlap] = 0

        piece_mask = build_piece_height_mask(
            height,
            overlap.astype(np.uint8) * 255,
            min_height_mm=self.min_piece_height_mm,
            max_height_mm=self.max_piece_height_mm,
            min_area_px=self.min_piece_area_px,
        )
        overlap_mask = overlap.astype(np.uint8) * 255
        return height.astype(np.float32), height_uint8, piece_mask, overlap_mask

    def build_views(self, depth_mm: np.ndarray) -> List[Tuple[str, np.ndarray, Optional[bool]]]:
        if not self.ready:
            h, w = depth_mm.shape[:2]
            status = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(status, "empty-board baseline not ready", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(status, f"captured {len(self.samples)}/{self.frames_required} frames", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(status, "leave board empty, or press b to restart", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            return [("baseline status", status, False)]

        _, height_uint8, piece_mask, overlap_mask = self.compute(depth_mm)
        return [
            ("baseline valid-overlap mask", overlap_mask, True),
            ("baseline height above board", height_uint8, True),
            ("baseline pre-CNN piece mask", piece_mask, True),
        ]


class DiagnosticRecorder:
    """Records the dashboard, individual named views, metadata and optional raw arrays."""

    def __init__(self, config: LiveViewConfig) -> None:
        self.config = config
        self.active = bool(config.record)
        self.session_dir: Optional[Path] = None
        self.view_dir: Optional[Path] = None
        self.raw_dir: Optional[Path] = None
        self.writers: Dict[str, cv2.VideoWriter] = {}
        self.frame_counter = 0
        self.recorded_counter = 0
        self.metadata_file = None
        self.fourcc = cv2.VideoWriter_fourcc(*config.record_codec)
        if self.active:
            self.start()

    def start(self) -> None:
        if self.active and self.session_dir is not None:
            return
        stamp = datetime.now().strftime("live_view_%Y%m%d_%H%M%S")
        self.session_dir = self.config.record_dir / stamp
        self.view_dir = self.session_dir / "views"
        self.raw_dir = self.session_dir / "raw_npz"
        self.view_dir.mkdir(parents=True, exist_ok=True)
        if self.config.record_raw_npz:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

        info = {
            "started_at": stamp,
            "mode": self.config.mode,
            "fps": self.config.fps,
            "view_size": list(self.config.view_size),
            "grid_cell_size": list(self.config.grid_cell_size),
            "grid_columns": self.config.grid_columns,
            "record_every": self.config.record_every,
            "record_dashboard": self.config.record_dashboard,
            "record_views": self.config.record_views,
            "record_raw_npz": self.config.record_raw_npz,
            "record_codec": self.config.record_codec,
            "notes": "Videos are 8-bit BGR diagnostic views. raw_npz contains exact arrays when enabled.",
        }
        (self.session_dir / "session_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        self.metadata_file = (self.session_dir / "metadata.jsonl").open("a", encoding="utf-8")
        self.active = True
        print(f"Recording started: {self.session_dir}")

    def stop(self) -> None:
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
        if self.metadata_file is not None:
            self.metadata_file.close()
            self.metadata_file = None
        if self.active and self.session_dir is not None:
            print(f"Recording stopped: {self.session_dir}")
        self.active = False

    def toggle(self) -> None:
        if self.active:
            self.stop()
        else:
            self.start()

    def _writer(self, name: str, frame_bgr: np.ndarray) -> cv2.VideoWriter:
        if self.session_dir is None:
            self.start()
        assert self.session_dir is not None
        if name in self.writers:
            return self.writers[name]
        h, w = frame_bgr.shape[:2]
        path = self.session_dir / f"{sanitise_filename(name)}.mp4" if name == "dashboard" else self.view_dir / f"{sanitise_filename(name)}.mp4"
        writer = cv2.VideoWriter(str(path), self.fourcc, max(1.0, self.config.fps / max(1, self.config.record_every)), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        self.writers[name] = writer
        return writer

    def record_frame(
        self,
        *,
        packet: KeyframePacket,
        views: List[Tuple[str, np.ndarray, Optional[bool]]],
        dashboard: np.ndarray,
        right_frame: Optional[np.ndarray],
    ) -> None:
        if not self.active:
            return
        self.frame_counter += 1
        if (self.frame_counter - 1) % max(1, self.config.record_every) != 0:
            return

        self.recorded_counter += 1
        if self.config.record_dashboard:
            self._writer("dashboard", ensure_bgr(dashboard)).write(ensure_bgr(dashboard))

        if self.config.record_views:
            for label, image, ok in views:
                cell = render_view_cell(label, image, ok, self.config.grid_cell_size)
                self._writer(label, cell).write(cell)

        if self.metadata_file is not None:
            row = {
                "recorded_index": self.recorded_counter,
                "frame_index": int(packet.frame_index),
                "timestamp": float(packet.timestamp),
                "stable": bool(packet.stable),
                "occluded": bool(packet.occluded),
                "keyframe": bool(packet.keyframe),
                "motion_score": float(packet.motion_score),
                "rgb_motion_score": float(packet.rgb_motion_score),
                "depth_motion_score": float(packet.depth_motion_score),
                "near_ratio": float(packet.near_ratio),
                "invalid_ratio": float(packet.invalid_ratio),
                "views": [label for label, _, _ in views],
            }
            self.metadata_file.write(json.dumps(row) + "\n")
            self.metadata_file.flush()

        if self.config.record_raw_npz and self.raw_dir is not None:
            raw_path = self.raw_dir / f"frame_{self.recorded_counter:06d}.npz"
            np.savez_compressed(
                raw_path,
                left_raw=packet.left_raw,
                left_norm=packet.left_norm,
                left_gray=packet.left_gray,
                right_raw=right_frame if right_frame is not None else np.array([], dtype=np.uint8),
                depth_raw_mm=packet.depth_raw,
                depth_gray=packet.depth_gray,
                depth_valid_mask=packet.depth_valid_mask,
                near_mask=packet.near_mask,
            )

    def save_snapshot(
        self,
        *,
        packet: KeyframePacket,
        views: List[Tuple[str, np.ndarray, Optional[bool]]],
        dashboard: np.ndarray,
        right_frame: Optional[np.ndarray],
    ) -> None:
        if self.session_dir is None:
            self.start()
        assert self.session_dir is not None
        snap_dir = self.session_dir / f"snapshot_{datetime.now().strftime('%H%M%S_%f')}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snap_dir / "dashboard.png"), ensure_bgr(dashboard))
        for label, image, ok in views:
            cv2.imwrite(str(snap_dir / f"{sanitise_filename(label)}.png"), render_view_cell(label, image, ok, self.config.grid_cell_size))
        np.savez_compressed(
            snap_dir / "raw_packet.npz",
            left_raw=packet.left_raw,
            left_norm=packet.left_norm,
            left_gray=packet.left_gray,
            right_raw=right_frame if right_frame is not None else np.array([], dtype=np.uint8),
            depth_raw_mm=packet.depth_raw,
            depth_gray=packet.depth_gray,
            depth_valid_mask=packet.depth_valid_mask,
            near_mask=packet.near_mask,
        )
        print(f"Snapshot saved: {snap_dir}")

class PreprocessingPreview:
    """Runs the pre-CNN geometric/visual pipeline for live diagnostics."""

    def __init__(self, config: LiveViewConfig) -> None:
        if any(x is None for x in (BoardRegistrar, PerspectiveRectifier, BoardNormaliser, PointTraySegmenter)):
            raise RuntimeError("Preprocessing modules are not importable. Check project paths.")

        self.config = config
        self.registrar = BoardRegistrar()
        self.rectifier = PerspectiveRectifier()
        self.normaliser = BoardNormaliser()
        self.segmenter = PointTraySegmenter()

    @staticmethod
    def _resize_depth_to_rgb(depth_mm: np.ndarray, rgb_bgr: np.ndarray) -> np.ndarray:
        if depth_mm.shape[:2] == rgb_bgr.shape[:2]:
            return depth_mm
        return cv2.resize(depth_mm.astype(np.float32), (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    def build_views(self, packet: KeyframePacket) -> List[Tuple[str, np.ndarray, Optional[bool]]]:
        views: List[Tuple[str, np.ndarray, Optional[bool]]] = []

        lock = self.registrar.update(packet.left_norm)
        views.append(("board registration", draw_board_lock(packet.left_norm, lock), bool(getattr(lock, "valid", False))))

        if not getattr(lock, "valid", False):
            return views

        try:
            depth_for_rgb = self._resize_depth_to_rgb(packet.depth_raw, packet.left_norm)
            rectified = self.rectifier.rectify(packet.left_norm, depth_for_rgb, lock)
            normalised = self.normaliser.normalize(rectified)
            regions = self.segmenter.segment(normalised.rgb_bgr.shape[:2])

            region_overlay = blend_overlay(normalised.rgb_bgr, regions.overlay_bgr, alpha=0.45)
            piece_mask = build_piece_height_mask(
                normalised.height_map_mm,
                normalised.valid_mask,
                min_height_mm=self.config.min_piece_height_mm,
                max_height_mm=self.config.max_piece_height_mm,
                min_area_px=self.config.min_piece_area_px,
            )

            views.extend(
                [
                    ("rectified RGB", rectified.rgb_bgr, True),
                    ("rectified raw depth BW", depth_to_grayscale(rectified.depth_mm), True),
                    ("normalised RGB", normalised.rgb_bgr, True),
                    ("normalised depth BW", normalised.depth_gray, True),
                    ("height above board", normalised.height_uint8, True),
                    ("point/tray masks", region_overlay, True),
                    ("pre-CNN piece height mask", piece_mask, True),
                ]
            )
        except Exception as exc:
            error_img = packet.left_norm.copy()
            cv2.putText(error_img, f"preprocess error: {exc}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            views.append(("preprocess error", error_img, False))

        return views


class LiveStreamViewer:
    """
    Live diagnostic viewer for OAK-D SR streams and pre-CNN processing stages.

    Recommended command-line use:
        python Acquisition/live_stream_viewer.py --mode all

    Recommended Jupyter use:
        %run Acquisition/live_stream_viewer.py --mode all
    """

    def __init__(self, config: Optional[LiveViewConfig] = None) -> None:
        self.config = config or LiveViewConfig()
        self._last_right_frame: Optional[np.ndarray] = None
        self._preprocessing_enabled = self.config.enable_preprocessing and self.config.mode in {"preprocess", "all"}
        self._preprocessor: Optional[PreprocessingPreview] = None
        self._baseline = EmptyBoardBaseline(
            frames_required=self.config.auto_baseline_frames,
            min_valid_ratio=self.config.baseline_min_valid_ratio,
            min_piece_height_mm=self.config.min_piece_height_mm,
            max_piece_height_mm=self.config.max_piece_height_mm,
            min_piece_area_px=self.config.min_piece_area_px,
            height_max_mm=self.config.height_max_mm,
        )
        self._recorder = DiagnosticRecorder(self.config)

        if self._preprocessing_enabled:
            try:
                self._preprocessor = PreprocessingPreview(self.config)
            except Exception as exc:
                print(f"Preprocessing preview disabled: {exc}")
                self._preprocessing_enabled = False

    def _make_gate(self) -> OakSRKeyframeGate:
        # Create exactly one DepthAI stream/pipeline. The earlier implementation
        # built a gate, then replaced gate.streams with a second OakSRStreams
        # instance. That meant two dai.Pipeline objects were constructed for one
        # viewer launch, unlike the user's working camtest.py.
        return OakSRKeyframeGate(
            fps=self.config.fps,
            view_size=self.config.view_size,
            motion_threshold=self.config.motion_threshold,
            hand_near_mm=self.config.hand_near_mm,
            max_near_ratio=self.config.max_near_ratio,
            max_invalid_ratio=self.config.max_invalid_ratio,
            enable_right=True,
        )

    def _build_views(self, packet: KeyframePacket, right_frame: Optional[np.ndarray]) -> List[Tuple[str, np.ndarray, Optional[bool]]]:
        mode = self.config.mode
        views: List[Tuple[str, np.ndarray, Optional[bool]]] = []

        if mode in {"streams", "gate", "preprocess", "all"}:
            views.append(("left RGB raw", packet.left_raw, None))
            if right_frame is not None:
                views.append(("right RGB raw", right_frame, None))
            views.append(("raw depth BW", depth_to_grayscale(packet.depth_raw), None))

        if mode in {"gate", "preprocess", "all"}:
            views.extend(
                [
                    ("keyframe gate status", draw_gate_status(packet.left_norm, packet), packet.keyframe),
                    ("left RGB normalised", packet.left_norm, packet.stable),
                    ("gate depth BW", packet.depth_gray, None),
                    ("valid depth mask", packet.depth_valid_mask * 255, packet.invalid_ratio <= self.config.max_invalid_ratio),
                    ("near/hand occlusion mask", packet.near_mask * 255, not packet.occluded),
                ]
            )
            if self.config.auto_baseline:
                self._baseline.add_empty_frame(packet.depth_raw)
                views.extend(self._baseline.build_views(packet.depth_raw))

        if mode in {"preprocess", "all"} and self._preprocessing_enabled and self._preprocessor is not None:
            views.extend(self._preprocessor.build_views(packet))

        return views

    def run(self) -> None:
        gate = self._make_gate().start()
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                packet = gate.get_packet(block=True)
                if packet is None:
                    continue

                right_frame = gate.streams.get_right_frame(block=False, use_cached=True)
                if right_frame is not None:
                    self._last_right_frame = right_frame
                else:
                    right_frame = self._last_right_frame

                views = self._build_views(packet, right_frame)
                grid = make_grid(views, cell_size=self.config.grid_cell_size, columns=self.config.grid_columns)

                if self.config.show_help:
                    rec_state = "REC" if self._recorder.active else "not rec"
                    help_text = "q/ESC quit | p preprocess | r record | s snapshot | b recapture baseline | c clear baseline | " + rec_state
                    cv2.putText(grid, help_text, (10, grid.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                self._recorder.record_frame(packet=packet, views=views, dashboard=grid, right_frame=right_frame)

                cv2.imshow(self.config.window_name, grid)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    self._preprocessing_enabled = not self._preprocessing_enabled
                    print(f"Preprocessing preview: {self._preprocessing_enabled}")
                if key == ord("r"):
                    self._recorder.toggle()
                if key == ord("s"):
                    self._recorder.save_snapshot(packet=packet, views=views, dashboard=grid, right_frame=right_frame)
                if key == ord("b"):
                    self._baseline.reset()
                    print("Baseline reset. Leave board empty while baseline is recaptured.")
                if key == ord("c"):
                    self._baseline.reset()
                    self.config.auto_baseline = False
                    print("Baseline cleared and auto-baseline disabled for this run.")

        finally:
            self._recorder.stop()
            gate.stop()
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live viewer for OAK-D SR raw streams, keyframe gate, and pre-CNN pipeline outputs.")
    parser.add_argument("--mode", choices=["streams", "gate", "preprocess", "all"], default="all", help="Which group of views to show.")
    parser.add_argument("--fps", type=float, default=30.0, help="Camera FPS.")
    parser.add_argument("--view-size", type=parse_size, default=(640, 400), help="Camera output size, e.g. 640x400.")
    parser.add_argument("--cell-size", type=parse_size, default=(420, 260), help="Each grid cell size, e.g. 420x260.")
    parser.add_argument("--columns", type=int, default=3, help="Number of columns in the display grid.")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing views even in preprocess/all mode.")
    parser.add_argument("--motion-threshold", type=float, default=2.5, help="Keyframe-gate motion threshold.")
    parser.add_argument("--hand-near-mm", type=int, default=450, help="Depth threshold for near-field hand/occlusion detection.")
    parser.add_argument("--max-near-ratio", type=float, default=0.03, help="Max ROI fraction allowed to be near/occluding.")
    parser.add_argument("--max-invalid-ratio", type=float, default=0.80, help="Max invalid-depth fraction allowed in gate ROI.")
    parser.add_argument("--min-piece-height-mm", type=float, default=3.0, help="Lower threshold for the pre-CNN piece height mask.")
    parser.add_argument("--max-piece-height-mm", type=float, default=40.0, help="Upper threshold for the pre-CNN piece height mask.")
    parser.add_argument("--min-piece-area-px", type=int, default=80, help="Minimum connected-component area for the pre-CNN piece mask.")
    parser.add_argument("--height-max-mm", type=float, default=35.0, help="Display scaling max for baseline height-above-board view.")
    parser.add_argument("--no-auto-baseline", action="store_true", help="Do not automatically capture an empty-board depth baseline from startup frames.")
    parser.add_argument("--auto-baseline-frames", type=int, default=30, help="Number of valid empty-board frames used for baseline capture.")
    parser.add_argument("--baseline-min-valid-ratio", type=float, default=0.20, help="Minimum valid depth fraction required for a frame to be used in baseline capture.")
    parser.add_argument("--record", action="store_true", help="Start recording diagnostic videos immediately.")
    parser.add_argument("--record-dir", type=Path, default=Path("troubleshooting_recordings"), help="Directory for diagnostic recording sessions.")
    parser.add_argument("--record-every", type=int, default=1, help="Save every Nth frame while recording.")
    parser.add_argument("--record-raw-npz", action="store_true", help="Also save exact raw arrays per recorded frame as compressed NPZ files.")
    parser.add_argument("--no-record-dashboard", action="store_true", help="Do not save the combined dashboard video.")
    parser.add_argument("--no-record-views", action="store_true", help="Do not save one video per individual view/stream.")
    parser.add_argument("--record-codec", type=str, default="mp4v", help="OpenCV video codec fourcc, e.g. mp4v or XVID.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = LiveViewConfig(
        mode=args.mode,
        fps=args.fps,
        view_size=args.view_size,
        grid_cell_size=args.cell_size,
        grid_columns=args.columns,
        motion_threshold=args.motion_threshold,
        hand_near_mm=args.hand_near_mm,
        max_near_ratio=args.max_near_ratio,
        max_invalid_ratio=args.max_invalid_ratio,
        enable_preprocessing=not args.no_preprocess,
        min_piece_height_mm=args.min_piece_height_mm,
        max_piece_height_mm=args.max_piece_height_mm,
        min_piece_area_px=args.min_piece_area_px,
        height_max_mm=args.height_max_mm,
        auto_baseline=not args.no_auto_baseline,
        auto_baseline_frames=args.auto_baseline_frames,
        baseline_min_valid_ratio=args.baseline_min_valid_ratio,
        record=args.record,
        record_dir=args.record_dir,
        record_every=args.record_every,
        record_dashboard=not args.no_record_dashboard,
        record_views=not args.no_record_views,
        record_raw_npz=args.record_raw_npz,
        record_codec=args.record_codec,
    )
    LiveStreamViewer(config).run()


if __name__ == "__main__":
    main()
