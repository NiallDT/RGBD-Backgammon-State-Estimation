from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backgammon_types import RegionMasks


@dataclass
class SegmentationConfig:
    outer_margin_frac_x: float = 0.045
    outer_margin_frac_y: float = 0.05
    bar_frac: float = 0.07
    point_tip_frac: float = 0.16
    point_base_frac: float = 0.15
    bearoff_frac: float = 0.09


def polygon_mask(shape_hw: Tuple[int, int], polygon_xy: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon_xy.astype(np.int32), 1)
    return mask


class PointTraySegmenter:
    """
    Produces canonical masks for the 24 points, central bar, and bear-off trays.

    This assumes the board has already been rectified into a stable canonical
    coordinate frame.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self.config = config or SegmentationConfig()

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

        tip_top = int(top_y0 + cfg.point_tip_frac * (top_y1 - top_y0))
        tip_bot = int(bot_y1 - cfg.point_tip_frac * (bot_y1 - bot_y0))

        masks: Dict[str, np.ndarray] = {}
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        def make_half_points(x0: int, x1: int, is_top: bool, point_numbers: List[int]) -> None:
            span = x1 - x0
            point_w = span / 6.0

            for i in range(6):
                px0 = int(round(x0 + i * point_w))
                px1 = int(round(x0 + (i + 1) * point_w))
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

                point_name = f"point_{point_numbers[i]:02d}"
                mask = polygon_mask((h, w), poly)
                masks[point_name] = mask
                colour = (0, 160, 255) if is_top else (255, 180, 0)
                overlay[mask > 0] = cv2.addWeighted(overlay, 1.0, np.full_like(overlay, colour), 0.25, 0)[mask > 0]

        # Standard numbering in a canonical rectified view:
        # top-left half 13..18, top-right half 19..24, bottom-left half 12..7, bottom-right half 6..1
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

        auxiliary_names = ["bar", "bearoff_left", "bearoff_right"]
        point_names = [f"point_{i:02d}" for i in range(1, 25)]

        overlay[masks["bar"] > 0] = (60, 60, 60)
        overlay[masks["bearoff_left"] > 0] = (40, 0, 160)
        overlay[masks["bearoff_right"] > 0] = (40, 160, 0)

        return RegionMasks(
            masks=masks,
            overlay_bgr=overlay,
            point_names=point_names,
            auxiliary_names=auxiliary_names,
        )
