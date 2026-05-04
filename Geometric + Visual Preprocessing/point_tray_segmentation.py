from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backgammon_types import RegionMasks

RectFrac = Tuple[float, float, float, float]


@dataclass
class SegmentationConfig:
    outer_margin_frac_x: float = 0.045
    outer_margin_frac_y: float = 0.05
    bar_frac: float = 0.07
    point_tip_frac: float = 0.16
    point_base_frac: float = 0.15
    bearoff_frac: float = 0.09

    # Manually tunable rectified-board regions. Fractions are (x1, y1, x2, y2)
    # in the rectified image coordinate frame, not raw camera coordinates.
    # These defaults are based on the most stable live-viewer crop reported so far.
    checker_detection_area_frac: Optional[RectFrac] = (0.10, 0.00, 1.00, 1.00)
    middle_strip_exclusion_frac: Optional[RectFrac] = (0.00, 0.38, 1.00, 0.60)
    dice_area_frac: Optional[RectFrac] = (0.10, 0.38, 1.00, 0.60)
    cube_area_frac: Optional[RectFrac] = (0.01, 0.45, 0.08, 0.52)
    depth_analysis_area_frac: Optional[RectFrac] = (0.10, 0.00, 1.00, 1.00)

    # Borne-off checker trays are deliberately excluded from normal point-stack
    # detection because they behave differently from playable points. Add a
    # separate tray counter later if needed.
    include_bar_in_checker_area: bool = True
    include_bearoff_in_checker_area: bool = False

    # Visualisation opacity for ROI overlays.
    roi_overlay_alpha: float = 0.28

    # The original version used triangular point masks matching the printed
    # board graphics. For stable checker detection it is often better to use
    # grid/rectangular point columns instead, because checker occupancy is
    # constrained by board regions rather than by the printed triangle art.
    #
    # False = grid/rectangular point regions.
    # True  = old triangular point regions.
    use_triangular_point_masks: bool = False

    # Draw the final checker_detection_area contour into RegionMasks.overlay_bgr.
    # This is off by default to keep ROI preview uncluttered.
    draw_checker_detection_contour: bool = False


def polygon_mask(shape_hw: Tuple[int, int], polygon_xy: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon_xy.astype(np.int32), 1)
    return mask


def rect_to_pixels(shape_hw: Tuple[int, int], frac: Optional[RectFrac]) -> Optional[Tuple[int, int, int, int]]:
    if frac is None:
        return None
    h, w = shape_hw
    x1f, y1f, x2f, y2f = frac
    x1 = int(round(np.clip(x1f, 0.0, 1.0) * w))
    y1 = int(round(np.clip(y1f, 0.0, 1.0) * h))
    x2 = int(round(np.clip(x2f, 0.0, 1.0) * w))
    y2 = int(round(np.clip(y2f, 0.0, 1.0) * h))
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def rect_mask(shape_hw: Tuple[int, int], frac: Optional[RectFrac]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    rect = rect_to_pixels(shape_hw, frac)
    if rect is None:
        return mask
    x1, y1, x2, y2 = rect
    mask[y1:y2, x1:x2] = 1
    return mask


def union_masks(shape_hw: Tuple[int, int], masks: List[np.ndarray]) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    for mask in masks:
        out = cv2.bitwise_or(out, mask.astype(np.uint8))
    return out


def subtract_masks(base: np.ndarray, exclusions: List[np.ndarray]) -> np.ndarray:
    out = base.copy().astype(np.uint8)
    for exclusion in exclusions:
        out[exclusion.astype(bool)] = 0
    return out


class PointTraySegmenter:
    """
    Produces canonical masks for the 24 points, central bar, bear-off trays,
    dice/cube areas, middle-strip exclusion, and checker-only detection area.

    All manually tuned ROIs are defined after perspective rectification, so the
    numbers are stable fractions of the canonical board image rather than raw
    camera pixel coordinates.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self.config = config or SegmentationConfig()

    @staticmethod
    def _label_rect(overlay: np.ndarray, rect: Optional[Tuple[int, int, int, int]], name: str, colour: Tuple[int, int, int]) -> None:
        if rect is None:
            return
        x1, y1, x2, y2 = rect
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            overlay,
            name,
            (x1 + 6, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

    def segment(self, shape_hw: Tuple[int, int]) -> RegionMasks:
        h, w = shape_hw
        cfg = self.config

        mx = int(cfg.outer_margin_frac_x * w)
        my = int(cfg.outer_margin_frac_y * h)
        bearoff_w = int(cfg.bearoff_frac * w)
        bar_w = int(cfg.bar_frac * w)

        playable_left = mx + bearoff_w
        playable_right = w - mx - bearoff_w
        playable_width = playable_right - playable_left
        half_playable = (playable_width - bar_w) // 2

        left_half_x0 = playable_left
        left_half_x1 = playable_left + half_playable
        bar_x0 = left_half_x1
        bar_x1 = bar_x0 + bar_w
        right_half_x0 = bar_x1
        right_half_x1 = playable_right

        top_y0 = my
        top_y1 = h // 2 - my // 2
        bot_y0 = h // 2 + my // 2
        bot_y1 = h - my

        masks: Dict[str, np.ndarray] = {}
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        def tint(mask: np.ndarray, colour: Tuple[int, int, int], alpha: float = 0.35) -> None:
            colour_img = np.full_like(overlay, colour)
            blended = cv2.addWeighted(overlay, 1.0 - alpha, colour_img, alpha, 0)
            overlay[mask > 0] = blended[mask > 0]

        def make_half_points(x0: int, x1: int, is_top: bool, point_numbers: List[int]) -> None:
            span = x1 - x0
            point_w = span / 6.0

            for i in range(6):
                px0 = int(round(x0 + i * point_w))
                px1 = int(round(x0 + (i + 1) * point_w))
                point_name = f"point_{point_numbers[i]:02d}"

                if cfg.use_triangular_point_masks:
                    # Legacy visual/geometry mode: mask the printed triangular point.
                    if is_top:
                        poly = np.array(
                            [[px0, top_y0], [px1, top_y0], [(px0 + px1) // 2, top_y1]],
                            dtype=np.int32,
                        )
                    else:
                        poly = np.array(
                            [[px0, bot_y1], [px1, bot_y1], [(px0 + px1) // 2, bot_y0]],
                            dtype=np.int32,
                        )
                    mask = polygon_mask((h, w), poly)
                else:
                    # Default detection mode: grid/rectangular point columns.
                    # This removes the dependency on the printed triangle contours.
                    mask = np.zeros((h, w), dtype=np.uint8)
                    if is_top:
                        mask[top_y0:top_y1, px0:px1] = 1
                    else:
                        mask[bot_y0:bot_y1, px0:px1] = 1

                masks[point_name] = mask
                tint(mask, (0, 160, 255) if is_top else (255, 180, 0), alpha=0.18)

        # Standard numbering in a canonical rectified view:
        # top-left half 13..18, top-right half 19..24,
        # bottom-left half 12..7, bottom-right half 6..1.
        make_half_points(left_half_x0, left_half_x1, True, [13, 14, 15, 16, 17, 18])
        make_half_points(right_half_x0, right_half_x1, True, [19, 20, 21, 22, 23, 24])
        make_half_points(left_half_x0, left_half_x1, False, [12, 11, 10, 9, 8, 7])
        make_half_points(right_half_x0, right_half_x1, False, [6, 5, 4, 3, 2, 1])

        bar_mask = np.zeros((h, w), dtype=np.uint8)
        bar_mask[:, bar_x0:bar_x1] = 1
        masks["bar"] = bar_mask

        masks["bearoff_left"] = np.zeros((h, w), dtype=np.uint8)
        masks["bearoff_left"][:, mx:playable_left] = 1
        masks["bearoff_right"] = np.zeros((h, w), dtype=np.uint8)
        masks["bearoff_right"][:, playable_right:w - mx] = 1

        # Manual named ROIs.
        masks["checker_detection_area_rect"] = rect_mask((h, w), cfg.checker_detection_area_frac)
        masks["middle_strip_exclusion"] = rect_mask((h, w), cfg.middle_strip_exclusion_frac)
        masks["dice_area"] = rect_mask((h, w), cfg.dice_area_frac)
        masks["cube_area"] = rect_mask((h, w), cfg.cube_area_frac)
        masks["depth_analysis_area"] = rect_mask((h, w), cfg.depth_analysis_area_frac)

        point_names = [f"point_{i:02d}" for i in range(1, 25)]
        auxiliary_names = ["bar", "bearoff_left", "bearoff_right"]

        checker_sources = [masks[name] for name in point_names]
        if cfg.include_bar_in_checker_area:
            checker_sources.append(masks["bar"])
        if cfg.include_bearoff_in_checker_area:
            checker_sources.extend([masks["bearoff_left"], masks["bearoff_right"]])

        checker_area = union_masks((h, w), checker_sources)

        # Restrict to manual overall checker rectangle/depth area.
        rect_area = masks["checker_detection_area_rect"]
        if np.count_nonzero(rect_area) > 0:
            checker_area = cv2.bitwise_and(checker_area, rect_area)
        depth_area = masks["depth_analysis_area"]
        if np.count_nonzero(depth_area) > 0:
            checker_area = cv2.bitwise_and(checker_area, depth_area)

        # Remove non-checker areas before piece detection can see them.
        checker_area = subtract_masks(
            checker_area,
            [
                masks["dice_area"],
                masks["cube_area"],
                masks["middle_strip_exclusion"],
            ],
        )
        masks["checker_detection_area"] = checker_area

        # Visual overlays.
        tint(masks["bar"], (60, 60, 60), alpha=0.40)
        tint(masks["bearoff_left"], (40, 0, 160), alpha=0.35)
        tint(masks["bearoff_right"], (40, 160, 0), alpha=0.35)
        tint(masks["dice_area"], (255, 255, 0), alpha=0.45)
        tint(masks["cube_area"], (255, 0, 255), alpha=0.45)
        tint(masks["middle_strip_exclusion"], (0, 0, 255), alpha=0.20)

        # Draw named ROI rectangles and checker detection contour.
        self._label_rect(overlay, rect_to_pixels((h, w), cfg.checker_detection_area_frac), "checker_rect", (255, 255, 0))
        self._label_rect(overlay, rect_to_pixels((h, w), cfg.dice_area_frac), "dice_area", (0, 255, 255))
        self._label_rect(overlay, rect_to_pixels((h, w), cfg.cube_area_frac), "cube_area", (255, 0, 255))
        self._label_rect(overlay, rect_to_pixels((h, w), cfg.middle_strip_exclusion_frac), "middle_exclusion", (0, 0, 255))

        if cfg.draw_checker_detection_contour:
            contours, _ = cv2.findContours((checker_area * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        return RegionMasks(
            masks=masks,
            overlay_bgr=overlay,
            point_names=point_names,
            auxiliary_names=auxiliary_names,
        )


__all__ = [
    "SegmentationConfig",
    "PointTraySegmenter",
    "polygon_mask",
    "rect_to_pixels",
    "rect_mask",
    "union_masks",
    "subtract_masks",
]
