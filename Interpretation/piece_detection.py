from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from backgammon_types import (
    NormalisedBoard,
    PieceDetectionResult,
    PieceInstance,
    RegionMasks,
    make_empty_region_counts,
)


@dataclass
class PieceDetectionConfig:
    # Height model. Chips are only expected to be either 1-high or 2-high.
    min_piece_height_mm: float = 3.0
    max_piece_height_mm: float = 35.0
    # Stack classification is deliberately binary: 1-high or 2-high only.
    # Promote/demote hysteresis avoids flickering around the boundary.
    stack_split_height_mm: float = 16.5  # backwards-compatible alias / default promote value
    stack_promote_height_mm: float = 16.5
    stack_demote_height_mm: float = 14.5
    chip_thickness_mm: float = 10.0
    max_stack_count: int = 2

    # Slot geometry. If expected_piece_radius_px is None, radius is estimated
    # from each point's mask width in the rectified board view.
    expected_piece_radius_px: Optional[float] = None
    slot_radius_from_point_width: float = 0.43
    slot_spacing_radius_mult: float = 1.75
    max_slots_per_point: int = 5

    # Evidence thresholds. Keep min_support_pixels permissive; temporal smoothing
    # should stabilise the result rather than filtering real edge/corner chips out.
    min_support_pixels: int = 6
    min_height_pixels: int = 6
    min_valid_depth_ratio: float = 0.01

    # Temporal support mask for depth flicker removal.
    temporal_support_window: int = 3
    temporal_support_required: int = 2

    # Slot-level hysteresis. This stops 2-high stacks flickering to 1-high when
    # passive stereo briefly under-reads the stack top.
    slot_history_window: int = 5
    slot_promote_votes: int = 2
    slot_demote_votes: int = 4
    slot_hold_misses: int = 1

    # Region policy.
    checker_detection_mask_name: str = "checker_detection_area"
    include_bar: bool = True
    include_bearoff: bool = False

    # Mask cleanup for the height support mask.
    morph_open_ksize: int = 3
    morph_close_ksize: int = 5


class RGBDPieceDetector:
    """
    Geometry-constrained RGB-D checker detector.

    The detector no longer searches for circles globally. It only samples legal
    checker slots inside the board's point masks (plus the bar if enabled), uses
    a temporally stable height-above-board support mask, and classifies each
    occupied slot as either a 1-high chip or a 2-high stack.

    This remains a classical baseline. The slot decision could be swapped for
    a learned classifier later if the dataset supports it.
    """

    def __init__(self, config: Optional[PieceDetectionConfig] = None) -> None:
        self.config = config or PieceDetectionConfig()
        self._support_history: Deque[np.ndarray] = deque(maxlen=self.config.temporal_support_window)
        self._slot_histories: Dict[Tuple[str, int], Deque[int]] = {}
        self._slot_stable: Dict[Tuple[str, int], int] = {}
        self._slot_misses: Dict[Tuple[str, int], int] = {}

    @staticmethod
    def _mean_lab(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab[mask > 0]
        if len(pixels) == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return pixels.mean(axis=0).astype(np.float32)

    @staticmethod
    def _circle_mask(shape_hw: Tuple[int, int], x: float, y: float, r: float) -> np.ndarray:
        mask = np.zeros(shape_hw, dtype=np.uint8)
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(max(1.0, r))), 1, -1)
        return mask

    @staticmethod
    def _circle_contour(x: float, y: float, r: float) -> np.ndarray:
        pts = []
        for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
            pts.append([int(round(x + np.cos(a) * r)), int(round(y + np.sin(a) * r))])
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    def _region_names_to_process(self, regions: RegionMasks) -> List[str]:
        names = list(regions.point_names)
        if self.config.include_bar and "bar" in regions.masks:
            names.append("bar")
        if self.config.include_bearoff:
            for name in ("bearoff_left", "bearoff_right"):
                if name in regions.masks:
                    names.append(name)
        return names

    def _checker_area_mask(self, shape_hw: Tuple[int, int], regions: RegionMasks) -> np.ndarray:
        if self.config.checker_detection_mask_name in regions.masks:
            return regions.masks[self.config.checker_detection_mask_name].astype(np.uint8)
        out = np.zeros(shape_hw, dtype=np.uint8)
        for name in self._region_names_to_process(regions):
            out = cv2.bitwise_or(out, regions.masks[name].astype(np.uint8))
        return out

    def _raw_support_mask(self, board: NormalisedBoard, checker_area: np.ndarray) -> np.ndarray:
        cfg = self.config
        raw = (
            (board.height_map_mm >= cfg.min_piece_height_mm)
            & (board.height_map_mm <= cfg.max_piece_height_mm)
            & (board.valid_mask > 0)
            & (checker_area > 0)
        ).astype(np.uint8)
        if cfg.morph_open_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_open_ksize, cfg.morph_open_ksize))
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
        if cfg.morph_close_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close_ksize, cfg.morph_close_ksize))
            raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)
        return raw.astype(np.uint8)

    def _stable_support_mask(self, raw: np.ndarray) -> np.ndarray:
        self._support_history.append(raw.copy())
        stack = np.stack(list(self._support_history), axis=0)
        required = min(self.config.temporal_support_required, len(self._support_history))
        return (stack.sum(axis=0) >= required).astype(np.uint8)

    def _slot_radius(self, region_mask: np.ndarray, image_width: int) -> float:
        cfg = self.config
        if cfg.expected_piece_radius_px is not None:
            return float(cfg.expected_piece_radius_px)
        ys, xs = np.where(region_mask > 0)
        if xs.size == 0:
            return max(5.0, 0.025 * image_width)
        width = float(xs.max() - xs.min() + 1)
        return float(np.clip(width * cfg.slot_radius_from_point_width, 5.0, 0.055 * image_width))

    def _slot_centres_for_point(self, region_name: str, region_mask: np.ndarray, shape_hw: Tuple[int, int]) -> List[Tuple[float, float, float]]:
        h, w = shape_hw
        ys, xs = np.where(region_mask > 0)
        if xs.size == 0:
            return []
        cx = float(np.mean(xs))
        radius = self._slot_radius(region_mask, w)
        spacing = radius * self.config.slot_spacing_radius_mult
        is_top = float(np.mean(ys)) < h / 2.0
        base_y = float(ys.min()) if is_top else float(ys.max())

        centres: List[Tuple[float, float, float]] = []
        for slot_idx in range(self.config.max_slots_per_point):
            if is_top:
                cy = base_y + radius + slot_idx * spacing
            else:
                cy = base_y - radius - slot_idx * spacing
            if cy < radius * 0.25 or cy > h - radius * 0.25:
                break
            centres.append((cx, cy, radius))
        return centres

    def _slot_centres_for_rect_region(self, region_mask: np.ndarray, shape_hw: Tuple[int, int]) -> List[Tuple[float, float, float]]:
        # Bar fallback. This is intentionally simpler than points because bar use
        # is secondary in the current pipeline.
        h, w = shape_hw
        ys, xs = np.where(region_mask > 0)
        if xs.size == 0:
            return []
        radius = self._slot_radius(region_mask, w)
        cx = float(np.mean(xs))
        y0, y1 = float(ys.min()), float(ys.max())
        centres = []
        y = y0 + radius
        idx = 0
        while y <= y1 - radius and idx < self.config.max_slots_per_point * 2:
            centres.append((cx, y, radius))
            y += radius * self.config.slot_spacing_radius_mult
            idx += 1
        return centres

    def _height_for_slot(self, board: NormalisedBoard, slot_mask: np.ndarray) -> Tuple[float, int, int]:
        vals = board.height_map_mm[(slot_mask > 0) & (board.valid_mask > 0)]
        vals = vals[(vals >= self.config.min_piece_height_mm * 0.5) & (vals <= self.config.max_piece_height_mm)]
        if vals.size == 0:
            return 0.0, 0, 0
        # Use an upper-band statistic to estimate the top face of a chip/stack,
        # but avoid using a single noisy maximum.
        threshold = np.percentile(vals, 65)
        top_vals = vals[vals >= threshold]
        if top_vals.size < 3:
            top_vals = vals
        height = float(np.median(top_vals))
        high_pixels = int(np.count_nonzero(vals >= self.config.stack_promote_height_mm))
        return height, int(vals.size), high_pixels

    def _smooth_slot_count(self, key: Tuple[str, int], raw_count: int, height_mm: float) -> int:
        cfg = self.config
        hist = self._slot_histories.setdefault(key, deque(maxlen=cfg.slot_history_window))
        # Store the raw count, but encode absence as 0. Height evidence is used
        # directly below so a one-frame under-read does not demote a true stack.
        hist.append(int(raw_count))
        counts = Counter(hist)
        prev = self._slot_stable.get(key, 0)

        # Track recent height evidence in a companion deque stored under a
        # sentinel key. This keeps the public structures simple while allowing
        # promote/demote hysteresis by actual mm values.
        height_key = (key[0], key[1] + 10_000)
        height_hist = self._slot_histories.setdefault(height_key, deque(maxlen=cfg.slot_history_window))
        if raw_count > 0:
            height_hist.append(2 if height_mm >= cfg.stack_promote_height_mm else 1 if height_mm <= cfg.stack_demote_height_mm else raw_count)
        else:
            height_hist.append(0)
        hcounts = Counter(height_hist)

        if prev == 0:
            if hcounts[2] >= cfg.slot_promote_votes or counts[2] >= cfg.slot_promote_votes:
                new = 2
            elif counts[1] >= cfg.slot_promote_votes:
                new = 1
            else:
                new = 0
        elif prev == 1:
            if hcounts[2] >= cfg.slot_promote_votes or counts[2] >= cfg.slot_promote_votes:
                new = 2
            elif counts[0] >= cfg.slot_demote_votes:
                new = 0
            else:
                new = 1
        else:  # prev == 2
            # A two-stack only drops to one-high after repeated low-height
            # evidence. This is the key fix for green/yellow flicker.
            if counts[0] >= cfg.slot_demote_votes:
                new = 0
            elif hcounts[1] >= cfg.slot_demote_votes and counts[1] >= max(1, cfg.slot_demote_votes - 1):
                new = 1
            else:
                new = 2

        self._slot_stable[key] = int(new)
        return int(new)

    def _raw_slot_count(self, board: NormalisedBoard, slot_mask: np.ndarray, support_pixels: int) -> Tuple[int, float, int, int]:
        height_mm, height_pixels, high_pixels = self._height_for_slot(board, slot_mask)
        if support_pixels < self.config.min_support_pixels and height_pixels < self.config.min_height_pixels:
            return 0, height_mm, height_pixels, high_pixels
        if height_mm <= 0:
            return 0, height_mm, height_pixels, high_pixels
        # Binary 1-vs-2 classification. Use a lower promote threshold than the
        # ideal 20 mm because passive stereo often under-reads stack height,
        # especially near edges. Temporal hysteresis handles borderline values.
        strong_high = high_pixels >= max(2, self.config.min_support_pixels // 2)
        raw_count = 2 if (height_mm >= self.config.stack_promote_height_mm or strong_high) else 1
        raw_count = int(np.clip(raw_count, 1, self.config.max_stack_count))
        return raw_count, height_mm, height_pixels, high_pixels

    def detect(self, board: NormalisedBoard, regions: RegionMasks) -> PieceDetectionResult:
        h, w = board.rgb_bgr.shape[:2]
        overlay = board.rgb_bgr.copy()
        region_order = self._region_names_to_process(regions)
        region_counts = make_empty_region_counts(regions.point_names + regions.auxiliary_names)
        pieces: List[PieceInstance] = []
        raw_records = []

        checker_area = self._checker_area_mask((h, w), regions)
        raw_support = self._raw_support_mask(board, checker_area)
        stable_support = self._stable_support_mask(raw_support)
        slot_candidate_mask = np.zeros((h, w), dtype=np.uint8)
        stack_class_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        for region_name in region_order:
            if region_name not in regions.masks:
                continue
            region_mask = cv2.bitwise_and(regions.masks[region_name].astype(np.uint8), checker_area)
            if np.count_nonzero(region_mask) == 0:
                continue
            if region_name.startswith("point_"):
                centres = self._slot_centres_for_point(region_name, region_mask, (h, w))
            else:
                centres = self._slot_centres_for_rect_region(region_mask, (h, w))

            for slot_idx, (cx, cy, radius) in enumerate(centres):
                full_slot_mask = self._circle_mask((h, w), cx, cy, radius)
                slot_mask = cv2.bitwise_and(full_slot_mask, checker_area)
                support_pixels = int(np.count_nonzero((stable_support > 0) & (slot_mask > 0)))
                raw_count, height_mm, height_pixels, high_pixels = self._raw_slot_count(board, slot_mask, support_pixels)
                stable_count = self._smooth_slot_count((region_name, slot_idx), raw_count, height_mm)
                if stable_count <= 0:
                    continue

                slot_candidate_mask[slot_mask > 0] = 255
                colour = (0, 255, 0) if stable_count == 1 else (0, 220, 255)
                stack_class_bgr[slot_mask > 0] = colour
                mean_lab = self._mean_lab(board.rgb_bgr, slot_mask)
                raw_records.append(
                    {
                        "region_name": region_name,
                        "slot_idx": slot_idx,
                        "centroid_xy": (cx, cy),
                        "radius_px": radius,
                        "area_px": float(np.count_nonzero(slot_mask)),
                        "height_mm": height_mm,
                        "stack_count": stable_count,
                        "raw_count": raw_count,
                        "support_pixels": support_pixels,
                        "height_pixels": height_pixels,
                        "high_pixels": high_pixels,
                        "mean_lab": mean_lab,
                        "mask": slot_mask,
                        "contour": self._circle_contour(cx, cy, radius),
                    }
                )

        # Assign light/dark using LAB lightness over detected slots. This works
        # best once non-checker areas have already been removed by RegionMasks.
        if raw_records:
            lightness = np.float32([rec["mean_lab"][0] for rec in raw_records]).reshape(-1, 1)
            if len(raw_records) >= 2:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                _, labels, centers = cv2.kmeans(lightness, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
                centers = centers.ravel()
                light_cluster = int(np.argmax(centers))
                colour_labels = ["light" if int(label[0]) == light_cluster else "dark" for label in labels]
            else:
                colour_labels = ["light" if float(lightness[0, 0]) >= 128.0 else "dark"]

            for rec, colour_name in zip(raw_records, colour_labels):
                confidence = float(np.clip(0.45 + 0.05 * rec["stack_count"] + 0.02 * min(rec["support_pixels"], 10), 0.0, 1.0))
                piece = PieceInstance(
                    region_name=rec["region_name"],
                    colour_name=colour_name,
                    centroid_xy=rec["centroid_xy"],
                    area_px=rec["area_px"],
                    radius_px=rec["radius_px"],
                    height_mm=rec["height_mm"],
                    stack_count=rec["stack_count"],
                    confidence=confidence,
                    contour=rec["contour"],
                )
                pieces.append(piece)
                if piece.region_name in region_counts:
                    region_counts[piece.region_name][piece.colour_name] += piece.stack_count

                draw_colour = (40, 255, 40) if colour_name == "light" else (40, 40, 255)
                cv2.drawContours(overlay, [piece.contour], -1, draw_colour, 2)
                label = f"{piece.region_name}:{colour_name[0]}x{piece.stack_count}"
                cv2.putText(
                    overlay,
                    label,
                    (int(piece.centroid_xy[0] - piece.radius_px), int(piece.centroid_xy[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    draw_colour,
                    1,
                    cv2.LINE_AA,
                )

        confidence = float(np.mean([p.confidence for p in pieces])) if pieces else 0.0
        return PieceDetectionResult(
            pieces=pieces,
            region_counts=region_counts,
            overlay_bgr=overlay,
            confidence=confidence,
            stable_depth_support_mask=(stable_support * 255).astype(np.uint8),
            slot_candidate_mask=slot_candidate_mask,
            checker_detection_area_mask=(checker_area * 255).astype(np.uint8),
            stack_class_bgr=stack_class_bgr,
            debug={
                "raw_support_ratio": float(np.mean(raw_support)),
                "stable_support_ratio": float(np.mean(stable_support)),
                "piece_count": len(pieces),
                "config": self.config.__dict__.copy(),
            },
        )


__all__ = ["PieceDetectionConfig", "RGBDPieceDetector"]
