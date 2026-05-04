from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import importlib.util
import json
import sys
import time

import cv2
import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent if _MODULE_DIR.name in {"Acquisition", "Interpretation", "Geometric + Visual Preprocessing"} else _MODULE_DIR
for _rel in ("", "Acquisition", "Interpretation", "Geometric + Visual Preprocessing"):
    _p = _PROJECT_ROOT / _rel if _rel else _PROJECT_ROOT
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from streams_module import OakSRStreams

# Load the classifier from the project Interpretation folder explicitly. This avoids
# Jupyter accidentally reusing an older cached depth_stack_classifier module.
_classifier_path = _PROJECT_ROOT / "Interpretation" / "depth_stack_classifier.py"
if not _classifier_path.exists():
    _classifier_path = _MODULE_DIR / "depth_stack_classifier.py"
if not _classifier_path.exists():
    _classifier_path = Path("depth_stack_classifier.py").resolve()

_spec = importlib.util.spec_from_file_location("depth_stack_classifier_runtime", _classifier_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load depth_stack_classifier from {_classifier_path}")
_dsc = importlib.util.module_from_spec(_spec)
# Python 3.12 dataclasses expect the module being executed to already be
# registered in sys.modules. Without this, @dataclass can fail with:
# AttributeError: 'NoneType' object has no attribute '__dict__'.
sys.modules[_spec.name] = _dsc
_spec.loader.exec_module(_dsc)
print(f"Loaded depth_stack_classifier from: {_classifier_path}")

EmptyBoardBaseline = _dsc.EmptyBoardBaseline
TemporalMaskFilter = _dsc.TemporalMaskFilter
TemporalStackSmoother = _dsc.TemporalStackSmoother
classify_rgb_candidates_by_depth = _dsc.classify_rgb_candidates_by_depth
depth_to_gray_fixed = _dsc.depth_to_gray_fixed
height_to_gray = _dsc.height_to_gray

Size = Tuple[int, int]
RoiFrac = Optional[Tuple[float, float, float, float]]


@dataclass
class Config:
    fps: float = 15.0
    view_size: Size = (640, 400)
    cell_size: Size = (420, 260)
    columns: int = 3

    baseline_frames: int = 60
    chip_thickness_mm: float = 10.0
    min_piece_height_mm: float = 3.0
    max_piece_height_mm: float = 45.0
    height_stat: str = "top35"

    depth_min_mm: float = 360.0
    depth_max_mm: float = 435.0
    height_max_visual_mm: float = 30.0

    min_chip_radius_px: float = 7.0
    max_chip_radius_px: float = 30.0
    expected_chip_radius_px: Optional[float] = None

    split_touching: bool = True
    use_hough: bool = False
    draw_rejected_candidates: bool = False
    min_candidate_support_ratio: float = 0.025
    min_support_pixels: int = 6
    min_depth_pixels: int = 8
    min_depth_ratio: float = 0.025
    roi_frac: RoiFrac = (0.18, 0.00, 0.88, 1.00)

    temporal_enabled: bool = True
    temporal_window: int = 3
    temporal_require: int = 2
    temporal_open_px: int = 3
    temporal_close_px: int = 5
    temporal_min_area_px: int = 30
    noise_margin_mm: float = 1.5

    stack_temporal_enabled: bool = True
    stack_temporal_window: int = 5
    stack_promote_votes: int = 2
    stack_demote_votes: int = 4
    stack_match_distance_px: float = 24.0
    stack_hold_misses: int = 2
    edge_margin_px: float = 42.0
    edge_promote_height_factor: float = 1.30
    edge_demote_height_factor: float = 1.18

    record: bool = False
    record_every: int = 3
    record_raw_npz: bool = False
    out_dir: Path = Path("troubleshooting_recordings/depth_stack_viewer")


def parse_size(text: str) -> Size:
    try:
        w, h = text.lower().split("x", 1)
        return int(w), int(h)
    except Exception as exc:
        raise argparse.ArgumentTypeError("size must look like 640x400") from exc


def parse_roi_frac(text: str) -> RoiFrac:
    if text.lower().strip() in {"none", "off", "false", "0"}:
        return None
    try:
        vals = tuple(float(v.strip()) for v in text.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError("ROI must be x1,y1,x2,y2 or 'none'") from exc
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("ROI must contain four comma-separated floats: x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise argparse.ArgumentTypeError("ROI fractions must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
    return vals  # type: ignore[return-value]


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        return img.copy()
    raise ValueError(str(img.shape))


def letterbox(img: np.ndarray, size: Size) -> np.ndarray:
    img = ensure_bgr(img)
    tw, th = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def grid(items: List[Tuple[str, np.ndarray]], cell_size: Size, columns: int) -> np.ndarray:
    cells = [label(letterbox(img, cell_size), name) for name, img in items]
    if not cells:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    blank = np.zeros_like(cells[0])
    rows = int(np.ceil(len(cells) / max(columns, 1)))
    cells += [blank] * (rows * columns - len(cells))
    return np.vstack([np.hstack(cells[r * columns:(r + 1) * columns]) for r in range(rows)])


class VideoRecorder:
    def __init__(self, out_dir: Path, fps: float) -> None:
        self.out_dir = out_dir / time.strftime("depth_stack_%Y%m%d_%H%M%S")
        self.views_dir = self.out_dir / "views"
        self.raw_dir = self.out_dir / "raw_npz"
        self.views_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writers = {}
        self.metadata = open(self.out_dir / "metadata.jsonl", "w", encoding="utf-8")

    def _writer(self, name: str, shape) -> cv2.VideoWriter:
        if name in self.writers:
            return self.writers[name]
        h, w = shape[:2]
        path = self.views_dir / f"{name}.mp4"
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writers[name] = wr
        return wr

    def write(self, views: List[Tuple[str, np.ndarray]], meta: dict, raw: Optional[dict] = None) -> None:
        for name, img in views:
            safe = name.lower().replace(" ", "_").replace("/", "-").replace("+", "plus")
            bgr = ensure_bgr(img)
            self._writer(safe, bgr.shape).write(bgr)
        self.metadata.write(json.dumps(meta) + "\n")
        self.metadata.flush()
        if raw is not None:
            np.savez_compressed(self.raw_dir / f"frame_{meta['frame']:06d}.npz", **raw)

    def close(self) -> None:
        for wr in self.writers.values():
            wr.release()
        self.metadata.close()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Live stack-height diagnostic viewer: stable depth support + RGB-restored chip circles + stack classification."
    )
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--view-size", type=parse_size, default=(640, 400))
    ap.add_argument("--cell-size", type=parse_size, default=(420, 260))
    ap.add_argument("--columns", type=int, default=3)

    ap.add_argument("--baseline-frames", type=int, default=60)
    ap.add_argument("--chip-thickness-mm", type=float, default=10.0)
    ap.add_argument("--min-piece-height-mm", type=float, default=3.0)
    ap.add_argument("--max-piece-height-mm", type=float, default=45.0)
    ap.add_argument("--height-stat", choices=["median", "p75", "p90", "max", "top35", "top25"], default="top35")

    ap.add_argument("--depth-min-mm", type=float, default=360.0)
    ap.add_argument("--depth-max-mm", type=float, default=435.0)
    ap.add_argument("--height-max-visual-mm", type=float, default=30.0)

    ap.add_argument("--min-chip-radius-px", type=float, default=7.0)
    ap.add_argument("--max-chip-radius-px", type=float, default=30.0)
    ap.add_argument("--expected-chip-radius-px", type=float, default=None)

    ap.add_argument("--no-split-touching", dest="split_touching", action="store_false", default=True)
    ap.add_argument("--hough", dest="use_hough", action="store_true", default=False)
    ap.add_argument("--draw-rejected-candidates", action="store_true", default=False)
    ap.add_argument("--min-candidate-support-ratio", type=float, default=0.025)
    ap.add_argument("--min-support-pixels", type=int, default=6)
    ap.add_argument("--min-depth-pixels", type=int, default=8)
    ap.add_argument("--min-depth-ratio", type=float, default=0.025)
    ap.add_argument("--roi-frac", type=parse_roi_frac, default=(0.18, 0.00, 0.88, 1.00))

    ap.add_argument("--disable-temporal", dest="temporal_enabled", action="store_false", default=True)
    ap.add_argument("--temporal-window", type=int, default=3)
    ap.add_argument("--temporal-require", type=int, default=2)
    ap.add_argument("--temporal-open-px", type=int, default=3)
    ap.add_argument("--temporal-close-px", type=int, default=5)
    ap.add_argument("--temporal-min-area-px", type=int, default=30)
    ap.add_argument("--noise-margin-mm", type=float, default=1.5)

    ap.add_argument("--disable-stack-temporal", dest="stack_temporal_enabled", action="store_false", default=True)
    ap.add_argument("--stack-temporal-window", type=int, default=5)
    ap.add_argument("--stack-promote-votes", type=int, default=2)
    ap.add_argument("--stack-demote-votes", type=int, default=4)
    ap.add_argument("--stack-match-distance-px", type=float, default=24.0)
    ap.add_argument("--stack-hold-misses", type=int, default=2, help="Hold an edge/corner two-stack for this many missed frames to prevent brief corner dropouts.")
    ap.add_argument("--edge-margin-px", type=float, default=42.0, help="Pixels from ROI/image boundary treated as edge/corner for stack hysteresis.")
    ap.add_argument("--edge-promote-height-factor", type=float, default=1.30, help="Near-edge height/chip-thickness ratio that counts as two-stack evidence.")
    ap.add_argument("--edge-demote-height-factor", type=float, default=1.18, help="Near-edge height/chip-thickness ratio below which demotion may occur after enough votes.")

    ap.add_argument("--record", action="store_true")
    ap.add_argument("--record-every", type=int, default=3)
    ap.add_argument("--record-raw-npz", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("troubleshooting_recordings/depth_stack_viewer"))
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = Config(**vars(args))

    baseline = EmptyBoardBaseline(samples_required=cfg.baseline_frames)
    temporal_filter = None
    if cfg.temporal_enabled:
        temporal_filter = TemporalMaskFilter(
            window=cfg.temporal_window,
            required=cfg.temporal_require,
            open_px=cfg.temporal_open_px,
            close_px=cfg.temporal_close_px,
            min_area_px=cfg.temporal_min_area_px,
        )

    stack_smoother = None
    if cfg.stack_temporal_enabled:
        stack_smoother = TemporalStackSmoother(
            window=cfg.stack_temporal_window,
            promote_votes=cfg.stack_promote_votes,
            demote_votes=cfg.stack_demote_votes,
            max_match_distance_px=cfg.stack_match_distance_px,
            hold_misses=cfg.stack_hold_misses,
            edge_margin_px=cfg.edge_margin_px,
            edge_promote_height_factor=cfg.edge_promote_height_factor,
            edge_demote_height_factor=cfg.edge_demote_height_factor,
        )

    cam = OakSRStreams(
        enable_left=True,
        enable_right=True,
        enable_depth=True,
        fps=cfg.fps,
        view_size=cfg.view_size,
        stereo_size=cfg.view_size,
    ).start()
    recorder = VideoRecorder(cfg.out_dir, max(1.0, cfg.fps / max(cfg.record_every, 1))) if cfg.record else None
    cv2.namedWindow("Depth stack diagnostic viewer", cv2.WINDOW_NORMAL)

    frame_i = 0
    try:
        while True:
            left = cam.get_left_frame(block=True, use_cached=True)
            right = cam.get_right_frame(block=False, use_cached=True)
            depth = cam.get_depth_frame(block=True, use_cached=True)
            if left is None or depth is None:
                continue

            raw_depth_bw = depth_to_gray_fixed(depth, cfg.depth_min_mm, cfg.depth_max_mm)
            views: List[Tuple[str, np.ndarray]] = [("left RGB", left), ("raw depth BW", raw_depth_bw)]
            if right is not None:
                views.insert(1, ("right RGB", right))

            if not baseline.ready:
                baseline.add_sample(depth)
                if temporal_filter is not None:
                    temporal_filter.reset()
                if stack_smoother is not None:
                    stack_smoother.reset()
                status = np.zeros((*left.shape[:2], 3), dtype=np.uint8)
                cv2.putText(status, f"Capturing empty-board baseline: {len(baseline.samples)}/{cfg.baseline_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(status, "Keep board empty and still. Press b to restart baseline.", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                views.append(("baseline status", status))
            else:
                result = classify_rgb_candidates_by_depth(
                    left,
                    depth,
                    baseline,
                    chip_thickness_mm=cfg.chip_thickness_mm,
                    min_piece_height_mm=cfg.min_piece_height_mm,
                    max_piece_height_mm=cfg.max_piece_height_mm,
                    height_stat=cfg.height_stat,
                    min_depth_pixels=cfg.min_depth_pixels,
                    min_depth_ratio=cfg.min_depth_ratio,
                    height_max_visual_mm=cfg.height_max_visual_mm,
                    min_radius_px=cfg.min_chip_radius_px,
                    max_radius_px=cfg.max_chip_radius_px,
                    expected_radius_px=cfg.expected_chip_radius_px,
                    split_touching=cfg.split_touching,
                    use_hough=cfg.use_hough,
                    min_candidate_support_ratio=cfg.min_candidate_support_ratio,
                    min_support_pixels=cfg.min_support_pixels,
                    draw_rejected_candidates=cfg.draw_rejected_candidates,
                    roi_frac=cfg.roi_frac,
                    temporal_filter=temporal_filter,
                    stack_smoother=stack_smoother,
                    noise_margin_mm=cfg.noise_margin_mm,
                    edge_margin_px=cfg.edge_margin_px,
                )
                height_bw = height_to_gray(result.height_mm, cfg.height_max_visual_mm)
                valid = result.valid_overlap_mask * 255
                stable_support = result.stable_depth_support_mask
                restored = result.rgb_restored_circle_mask
                stack_overlay = result.overlay_bgr.copy()
                text = (
                    f"chips={int(result.diagnostics['candidate_count'])} "
                    f"classified={int(result.diagnostics['classified_count'])} "
                    f"raw={result.diagnostics['raw_support_ratio']:.4f} "
                    f"stable={result.diagnostics['stable_support_ratio']:.4f} "
                    f"T={int(result.diagnostics['temporal_history'])}/{cfg.temporal_window if cfg.temporal_enabled else 0} "
                    f"S={int(result.diagnostics.get('stack_temporal_tracks', 0))} "
                    f"held={int(result.diagnostics.get('held_candidate_count', 0))}"
                )
                cv2.putText(stack_overlay, text, (10, stack_overlay.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
                views.extend([
                    ("height above board BW", height_bw),
                    ("valid overlap mask", valid),
                    ("stable depth support mask", stable_support),
                    ("RGB candidate mask", result.rgb_candidate_mask),
                    ("RGB-restored circle mask", restored),
                    ("stack class mask", result.stack_class_bgr),
                    ("RGB+depth stack overlay", stack_overlay),
                ])

                if recorder and (frame_i % max(cfg.record_every, 1) == 0):
                    raw = None
                    if cfg.record_raw_npz:
                        raw = {
                            "left_rgb": left,
                            "right_rgb": right if right is not None else np.zeros_like(left),
                            "depth_raw_mm": depth,
                            "height_mm": result.height_mm,
                            "valid_overlap_mask": result.valid_overlap_mask,
                            "stable_depth_support_mask": result.stable_depth_support_mask,
                            "rgb_candidate_mask": result.rgb_candidate_mask,
                            "rgb_restored_circle_mask": result.rgb_restored_circle_mask,
                            "stack_class_bgr": result.stack_class_bgr,
                        }
                    recorder.write(views, {"frame": frame_i, "time": time.time(), **result.diagnostics}, raw=raw)

            dash = grid(views, cfg.cell_size, cfg.columns)
            cv2.putText(dash, "q/ESC quit | b recapture empty-board baseline", (10, dash.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("Depth stack diagnostic viewer", dash)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                baseline.clear()
                if temporal_filter is not None:
                    temporal_filter.reset()
                if stack_smoother is not None:
                    stack_smoother.reset()
                print("Baseline cleared. Keep board empty and still.")
            frame_i += 1
    finally:
        cam.stop()
        if recorder:
            recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
