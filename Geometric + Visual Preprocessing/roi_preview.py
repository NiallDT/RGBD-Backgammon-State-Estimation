from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import json
import sys

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

from streams_module import OakSRStreams
from backgammon_types import BoardLock
from board_registration import BoardRegistrar
from perspective_rectification import PerspectiveRectifier
from image_normalisation import BoardNormaliser
from point_tray_segmentation import PointTraySegmenter


WINDOW_NAME = "ROI preview"


@dataclass
class PreviewState:
    mode: str = "auto"  # auto, manual, full-frame
    manual_points: Optional[np.ndarray] = None
    pending_clicks: List[Tuple[int, int]] = None
    last_raw_frame: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.pending_clicks is None:
            self.pending_clicks = []


def colour_mask(mask: np.ndarray, colour_bgr: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    out[mask > 0] = colour_bgr
    return out


def add_text_panel(frame_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    out = frame_bgr.copy()
    y = 24
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return out


def make_manual_lock(corners_xy: np.ndarray, frame_shape_hw: Tuple[int, int]) -> BoardLock:
    return BoardLock(
        valid=True,
        corners_xy=corners_xy.astype(np.float32),
        confidence=1.0,
        method="manual",
        frame_size_hw=frame_shape_hw,
        debug={"source": "roi_preview_manual"},
    )


def make_full_frame_lock(frame_shape_hw: Tuple[int, int]) -> BoardLock:
    h, w = frame_shape_hw
    corners = np.array(
        [
            [0.0, 0.0],
            [float(w - 1), 0.0],
            [float(w - 1), float(h - 1)],
            [0.0, float(h - 1)],
        ],
        dtype=np.float32,
    )
    return BoardLock(
        valid=True,
        corners_xy=corners,
        confidence=1.0,
        method="full-frame",
        frame_size_hw=frame_shape_hw,
        debug={"source": "roi_preview_full_frame"},
    )


def draw_pending_points(frame_bgr: np.ndarray, state: PreviewState) -> np.ndarray:
    out = frame_bgr.copy()

    if state.manual_points is not None:
        pts = state.manual_points.astype(int)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(out, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i, (x, y) in enumerate(state.pending_clicks):
        cv2.circle(out, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(out, str(i + 1), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return out


def mouse_callback(event, x, y, flags, param) -> None:
    state: PreviewState = param

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state.mode != "manual":
        return

    state.pending_clicks.append((int(x), int(y)))

    if len(state.pending_clicks) == 4:
        state.manual_points = np.array(state.pending_clicks, dtype=np.float32)
        state.pending_clicks.clear()
        print("Manual board corners set. Order assumed: top-left, top-right, bottom-right, bottom-left.")


def save_manual_corners(path: Path, corners_xy: np.ndarray) -> None:
    payload = {
        "corner_order": "top-left, top-right, bottom-right, bottom-left",
        "corners_xy": corners_xy.astype(float).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved manual corners to: {path}")


def load_manual_corners(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        print(f"No saved manual corners found at: {path}")
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    corners = np.array(payload["corners_xy"], dtype=np.float32)

    if corners.shape != (4, 2):
        print(f"Invalid saved corners shape: {corners.shape}")
        return None

    print(f"Loaded manual corners from: {path}")
    return corners


def build_roi_overlay(base_bgr: np.ndarray, regions) -> np.ndarray:
    overlay = base_bgr.copy()

    named_colours = {
        "checker_detection_area": (0, 255, 0),          # green
        "dice_area": (0, 255, 255),                     # yellow/cyan
        "cube_area": (255, 0, 255),                     # magenta
        "middle_strip_exclusion": (0, 0, 255),          # red
        "checker_detection_area_rect": (255, 255, 0),   # cyan
        "depth_analysis_area": (255, 120, 0),           # orange/blueish
    }

    # First blend filled masks.
    fill = np.zeros_like(base_bgr)
    for name, colour in named_colours.items():
        mask = regions.masks.get(name)
        if mask is None:
            continue
        fill[mask.astype(bool)] = colour

    overlay = cv2.addWeighted(overlay, 0.72, fill, 0.28, 0)

    # Then draw outlines and labels.
    for name, colour in named_colours.items():
        mask = regions.masks.get(name)
        if mask is None:
            continue

        mask_u8 = (mask.astype(np.uint8) * 255)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            # The checker_detection_area can contain many grid cells; drawing all
            # contours makes the preview noisy. Show it as a filled green overlay
            # with a simple bounding rectangle instead. Other named ROIs still get
            # their exact contour/rectangle outlines.
            if name == "checker_detection_area":
                cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
            else:
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, colour, 2)

            cv2.putText(
                overlay,
                name,
                (x1 + 5, max(22, y1 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colour,
                2,
                cv2.LINE_AA,
            )

    return overlay


def main() -> None:
    state = PreviewState()
    corners_path = _PROJECT_ROOT / "manual_board_corners.json"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)

    cam = OakSRStreams(enable_left=True, enable_depth=True, fps=15).start()

    registrar = BoardRegistrar()
    rectifier = PerspectiveRectifier()
    normaliser = BoardNormaliser()
    segmenter = PointTraySegmenter()

    print("ROI preview controls:")
    print("  a = auto board lock")
    print("  m = manual corner mode; click 4 corners: TL, TR, BR, BL")
    print("  f = full-frame fallback lock")
    print("  c = clear manual points")
    print("  s = save manual corners")
    print("  l = load manual corners")
    print("  q/Esc = quit")

    try:
        while True:
            rgb = cam.get_left_frame()
            depth = cam.get_depth_frame()

            if rgb is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            state.last_raw_frame = rgb.copy()
            h, w = rgb.shape[:2]

            lock: Optional[BoardLock] = None
            status_lines: List[str] = []

            if state.mode == "full-frame":
                lock = make_full_frame_lock((h, w))
                status_lines.append("Mode: FULL-FRAME fallback lock")
                status_lines.append("Useful for rough ROI tuning only.")
            elif state.mode == "manual":
                if state.manual_points is not None:
                    lock = make_manual_lock(state.manual_points, (h, w))
                    status_lines.append("Mode: MANUAL board lock")
                else:
                    view = draw_pending_points(rgb, state)
                    status_lines = [
                        "Mode: MANUAL corner selection",
                        "Click board corners in order: TL, TR, BR, BL",
                        f"Clicked: {len(state.pending_clicks)}/4",
                        "a=auto  f=full-frame  c=clear  l=load  q=quit",
                    ]
                    cv2.imshow(WINDOW_NAME, add_text_panel(view, status_lines))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("a"):
                        state.mode = "auto"
                    elif key == ord("f"):
                        state.mode = "full-frame"
                    elif key == ord("c"):
                        state.pending_clicks.clear()
                        state.manual_points = None
                    elif key == ord("l"):
                        loaded = load_manual_corners(corners_path)
                        if loaded is not None:
                            state.manual_points = loaded
                            state.mode = "manual"
                    continue
            else:
                lock = registrar.update(rgb)
                status_lines.append("Mode: AUTO BoardRegistrar")

            if lock is None or not lock.valid:
                view = draw_pending_points(rgb, state)
                status_lines.extend(
                    [
                        "Board lock invalid.",
                        "Press m to click manual board corners.",
                        "Press f to use full-frame fallback for rough ROI tuning.",
                        "Press l to load saved manual corners.",
                        "q/Esc=quit",
                    ]
                )
                cv2.imshow(WINDOW_NAME, add_text_panel(view, status_lines))
            else:
                if depth is None:
                    # Use a zero depth frame so the rectifier can still render RGB ROI overlay in fallback cases.
                    depth = np.zeros((h, w), dtype=np.uint16)

                # If a cropped-depth stream comes back different from RGB, the streams module should
                # normally expand it to full frame. This resize is just a defensive fallback.
                if depth.shape[:2] != rgb.shape[:2]:
                    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

                try:
                    rectified = rectifier.rectify(rgb, depth, lock)
                    normalised = normaliser.normalize(rectified)
                    regions = segmenter.segment(normalised.rgb_bgr.shape[:2])
                    preview = build_roi_overlay(normalised.rgb_bgr, regions)

                    status_lines.extend(
                        [
                            f"Lock: {lock.method} conf={lock.confidence:.2f}",
                            "green=checker_detection_area  yellow=dice_area",
                            "magenta=cube_area  red=middle_strip_exclusion",
                            "a=auto  m=manual  f=full-frame  s=save  l=load  q=quit",
                        ]
                    )
                    preview = add_text_panel(preview, status_lines)
                    cv2.imshow(WINDOW_NAME, preview)
                except Exception as exc:
                    view = draw_pending_points(rgb, state)
                    status_lines.extend(
                        [
                            f"ROI preview error: {type(exc).__name__}: {exc}",
                            "Try f for full-frame, or m to set manual corners.",
                        ]
                    )
                    cv2.imshow(WINDOW_NAME, add_text_panel(view, status_lines))

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                state.mode = "auto"
                print("Mode set to auto.")
            elif key == ord("m"):
                state.mode = "manual"
                state.pending_clicks.clear()
                print("Manual mode: click TL, TR, BR, BL board corners.")
            elif key == ord("f"):
                state.mode = "full-frame"
                print("Mode set to full-frame fallback.")
            elif key == ord("c"):
                state.pending_clicks.clear()
                state.manual_points = None
                print("Cleared manual corners.")
            elif key == ord("s"):
                if state.manual_points is not None:
                    save_manual_corners(corners_path, state.manual_points)
                else:
                    print("No manual corners to save.")
            elif key == ord("l"):
                loaded = load_manual_corners(corners_path)
                if loaded is not None:
                    state.manual_points = loaded
                    state.mode = "manual"

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
