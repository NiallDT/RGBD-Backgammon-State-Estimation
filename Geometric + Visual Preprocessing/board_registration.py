from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from backgammon_types import BoardLock


@dataclass
class BoardRegistrationConfig:
    method: str = "auto"
    use_aruco: bool = True
    smoothing: float = 0.75
    min_area_ratio: float = 0.12
    min_confidence: float = 0.15
    canny_low: int = 50
    canny_high: int = 150
    debug: bool = False


def order_corners(corners_xy: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2)
    s = corners.sum(axis=1)
    d = np.diff(corners, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = corners[np.argmin(s)]  # top-left
    ordered[2] = corners[np.argmax(s)]  # bottom-right
    ordered[1] = corners[np.argmin(d)]  # top-right
    ordered[3] = corners[np.argmax(d)]  # bottom-left
    return ordered


def polygon_area(corners_xy: np.ndarray) -> float:
    corners = np.asarray(corners_xy, dtype=np.float32).reshape(-1, 2)
    x = corners[:, 0]
    y = corners[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def contour_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    return approx.reshape(4, 2).astype(np.float32)


def detect_board_corners_contour(frame_bgr: np.ndarray, cfg: BoardRegistrationConfig) -> Tuple[Optional[np.ndarray], float, dict]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    frame_area = float(h * w)
    best_quad = None
    best_area = 0.0

    for contour in contours:
        quad = contour_to_quad(contour)
        if quad is None:
            continue

        area = polygon_area(quad)
        if area < cfg.min_area_ratio * frame_area:
            continue

        if area > best_area:
            best_area = area
            best_quad = quad

    confidence = 0.0 if best_quad is None else min(1.0, best_area / (0.55 * frame_area + 1e-6))
    debug = {"edge_nonzero": int(np.count_nonzero(edges)), "best_area": best_area}
    return best_quad, confidence, debug


def detect_board_corners_aruco(frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float, dict]:
    if not hasattr(cv2, "aruco"):
        return None, 0.0, {"aruco_available": False}

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame_bgr)

    if ids is None or len(corners) < 2:
        return None, 0.0, {"aruco_markers": 0}

    points = np.concatenate([c.reshape(-1, 2) for c in corners], axis=0).astype(np.float32)
    hull = cv2.convexHull(points).reshape(-1, 2)
    rect = cv2.minAreaRect(hull)
    quad = cv2.boxPoints(rect).astype(np.float32)

    confidence = min(1.0, 0.2 + 0.15 * len(corners))
    return quad, confidence, {"aruco_markers": int(len(corners))}


class BoardRegistrar:
    """
    Initial board-registration / recovery stage.

    This is a classical implementation intended to get you to the point where
    you can lock the board and start collecting aligned training data. Later,
    the same interface can be backed by a dedicated corner-regression CNN.
    """

    def __init__(self, config: Optional[BoardRegistrationConfig] = None) -> None:
        self.config = config or BoardRegistrationConfig()
        self._previous_corners: Optional[np.ndarray] = None
        self._manual_corners: Optional[np.ndarray] = None

    def set_manual_corners(self, corners_xy: np.ndarray) -> None:
        self._manual_corners = order_corners(corners_xy)

    def clear_manual_corners(self) -> None:
        self._manual_corners = None

    def _smooth(self, corners_xy: np.ndarray) -> np.ndarray:
        if self._previous_corners is None:
            return corners_xy
        alpha = float(np.clip(self.config.smoothing, 0.0, 1.0))
        return alpha * self._previous_corners + (1.0 - alpha) * corners_xy

    def _build_lock(
        self,
        frame_bgr: np.ndarray,
        corners_xy: Optional[np.ndarray],
        confidence: float,
        method: str,
        debug: dict,
    ) -> BoardLock:
        h, w = frame_bgr.shape[:2]
        if corners_xy is None:
            return BoardLock(
                valid=False,
                corners_xy=np.zeros((4, 2), dtype=np.float32),
                confidence=0.0,
                method=method,
                frame_size_hw=(h, w),
                debug=debug,
            )

        ordered = order_corners(corners_xy)
        ordered = self._smooth(ordered).astype(np.float32)
        self._previous_corners = ordered.copy()

        valid = confidence >= self.config.min_confidence
        return BoardLock(
            valid=valid,
            corners_xy=ordered,
            confidence=float(confidence),
            method=method,
            frame_size_hw=(h, w),
            debug=debug,
        )

    def update(self, frame_bgr: np.ndarray) -> BoardLock:
        if self._manual_corners is not None:
            return self._build_lock(
                frame_bgr,
                self._manual_corners,
                confidence=1.0,
                method="manual",
                debug={"source": "manual"},
            )

        if self.config.use_aruco:
            corners_xy, confidence, debug = detect_board_corners_aruco(frame_bgr)
            if corners_xy is not None:
                return self._build_lock(frame_bgr, corners_xy, confidence, "aruco", debug)

        corners_xy, confidence, debug = detect_board_corners_contour(frame_bgr, self.config)
        if corners_xy is not None:
            return self._build_lock(frame_bgr, corners_xy, confidence, "contour", debug)

        if self._previous_corners is not None:
            return self._build_lock(
                frame_bgr,
                self._previous_corners,
                confidence=0.05,
                method="previous",
                debug={"source": "previous"},
            )

        return self._build_lock(frame_bgr, None, 0.0, "none", {"source": "none"})
