from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    min_piece_height_mm: float = 3.0
    max_piece_height_mm: float = 80.0
    chip_thickness_mm: float = 10.0
    min_component_area_px: int = 40
    max_component_area_px: int = 20000
    morph_open_ksize: int = 3
    morph_close_ksize: int = 5
    expected_piece_diameter_px: Optional[float] = None


class RGBDPieceDetector:
    """
    Classical RGB-D detector that estimates checker stacks per region.

    This intentionally leaves a clean interface for later swapping in a CNN
    or Mask-RCNN-style segmenter while still giving you something usable for
    prototyping and data collection now.
    """

    def __init__(self, config: Optional[PieceDetectionConfig] = None) -> None:
        self.config = config or PieceDetectionConfig()

    def _extract_components(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def _mean_lab(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab[mask > 0]
        if len(pixels) == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return pixels.mean(axis=0).astype(np.float32)

    def detect(self, board: NormalisedBoard, regions: RegionMasks) -> PieceDetectionResult:
        h, w = board.rgb_bgr.shape[:2]
        overlay = board.rgb_bgr.copy()
        region_counts = make_empty_region_counts(regions.point_names + regions.auxiliary_names)
        pieces: List[PieceInstance] = []

        open_kernel = np.ones((self.config.morph_open_ksize, self.config.morph_open_ksize), np.uint8)
        close_kernel = np.ones((self.config.morph_close_ksize, self.config.morph_close_ksize), np.uint8)

        raw_piece_records = []

        region_order = regions.point_names + regions.auxiliary_names
        for region_name in region_order:
            region_mask = regions.masks[region_name].astype(np.uint8)
            region_height = np.where(region_mask > 0, board.height_map_mm, 0.0)

            binary = ((region_height >= self.config.min_piece_height_mm) & (region_height <= self.config.max_piece_height_mm)).astype(np.uint8) * 255
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

            contours = self._extract_components(binary)

            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.config.min_component_area_px or area > self.config.max_component_area_px:
                    continue

                contour_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 1, thickness=-1)

                if np.count_nonzero(contour_mask & region_mask) == 0:
                    continue

                moments = cv2.moments(contour)
                if abs(moments["m00"]) < 1e-6:
                    continue

                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
                (_, _), radius = cv2.minEnclosingCircle(contour)

                heights = board.height_map_mm[contour_mask > 0]
                if heights.size == 0:
                    continue

                height_mm = float(np.percentile(heights, 95))
                stack_count = max(1, int(round(height_mm / max(self.config.chip_thickness_mm, 1e-6))))
                mean_lab = self._mean_lab(board.rgb_bgr, contour_mask)

                raw_piece_records.append(
                    {
                        "region_name": region_name,
                        "centroid_xy": (cx, cy),
                        "area_px": area,
                        "radius_px": float(radius),
                        "height_mm": height_mm,
                        "stack_count": stack_count,
                        "confidence": float(min(1.0, 0.4 + 0.015 * stack_count + 0.0002 * area)),
                        "contour": contour,
                        "mean_lab": mean_lab,
                    }
                )

        if raw_piece_records:
            lightness = np.float32([rec["mean_lab"][0] for rec in raw_piece_records]).reshape(-1, 1)
            if len(raw_piece_records) >= 2:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
                _, labels, centers = cv2.kmeans(lightness, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
                centers = centers.ravel()
                light_cluster = int(np.argmax(centers))
                colour_labels = ["light" if int(label[0]) == light_cluster else "dark" for label in labels]
            else:
                colour_labels = ["light" if float(lightness[0, 0]) >= 128.0 else "dark"]

            for rec, colour_name in zip(raw_piece_records, colour_labels):
                piece = PieceInstance(
                    region_name=rec["region_name"],
                    colour_name=colour_name,
                    centroid_xy=rec["centroid_xy"],
                    area_px=rec["area_px"],
                    radius_px=rec["radius_px"],
                    height_mm=rec["height_mm"],
                    stack_count=rec["stack_count"],
                    confidence=rec["confidence"],
                    contour=rec["contour"],
                )
                pieces.append(piece)
                region_counts[piece.region_name][piece.colour_name] += piece.stack_count

                draw_colour = (0, 255, 0) if colour_name == "light" else (0, 0, 255)
                cv2.drawContours(overlay, [piece.contour], -1, draw_colour, 2)
                cv2.putText(
                    overlay,
                    f"{piece.region_name}:{piece.colour_name[0]}x{piece.stack_count}",
                    (int(piece.centroid_xy[0]) - 30, int(piece.centroid_xy[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
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
        )
