from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backgammon_types import CubeObservation, DiceCubeObservation, DiceObservation, NormalisedBoard, RegionMasks


try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


@dataclass
class DiceCubeConfig:
    min_square_area_px: int = 150
    max_square_area_px: int = 20000
    pip_min_area_px: int = 5
    pip_max_area_px: int = 500
    cube_min_area_px: int = 250
    cube_max_area_px: int = 30000


class DiceCubeReader:
    """
    Classical dice / doubling-cube observation step.

    Dice and cube searches can be restricted to RegionMasks['dice_area'] and
    RegionMasks['cube_area'] when those masks are available. This avoids the
    checker detector and dice/cube reader fighting over the same visual objects.
    """

    def __init__(self, config: Optional[DiceCubeConfig] = None) -> None:
        self.config = config or DiceCubeConfig()

    @staticmethod
    def _aspect_ratio_ok(w: int, h: int, tol: float = 0.35) -> bool:
        ratio = w / max(h, 1)
        return abs(ratio - 1.0) <= tol

    @staticmethod
    def _masked_gray(gray: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None or np.count_nonzero(mask) == 0:
            return gray
        return cv2.bitwise_and(gray, gray, mask=(mask.astype(np.uint8) * 255))

    @staticmethod
    def _box_center_in_mask(box: Tuple[int, int, int, int], mask: Optional[np.ndarray]) -> bool:
        if mask is None or np.count_nonzero(mask) == 0:
            return True
        x1, y1, x2, y2 = box
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        h, w = mask.shape[:2]
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return False
        return bool(mask[cy, cx] > 0)

    def _find_square_candidates(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_square_area_px or area > self.config.max_square_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not self._aspect_ratio_ok(w, h):
                continue
            boxes.append((x, y, x + w, y + h))
        return boxes

    def _count_pips(self, patch_bgr: np.ndarray) -> Tuple[Optional[int], float]:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        pip_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.pip_min_area_px <= area <= self.config.pip_max_area_px:
                pip_count += 1
        if 1 <= pip_count <= 6:
            return pip_count, min(1.0, 0.3 + 0.1 * pip_count)
        return None, 0.0

    def _read_cube_value(self, patch_bgr: np.ndarray) -> Tuple[Optional[int], float, Optional[str]]:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if pytesseract is None:
            return None, 0.0, None
        text = pytesseract.image_to_string(thresh, config="--psm 8 -c tessedit_char_whitelist=012468")
        text = "".join(ch for ch in text if ch.isdigit())
        if not text:
            return None, 0.0, None
        try:
            value = int(text)
        except ValueError:
            return None, 0.0, text
        if value in {2, 4, 8, 16, 32, 64}:
            return value, 0.8, text
        return None, 0.2, text

    def read(self, board: NormalisedBoard, regions: Optional[RegionMasks] = None) -> DiceCubeObservation:
        overlay = board.rgb_bgr.copy()
        dice_mask = regions.masks.get("dice_area") if regions is not None else None
        cube_mask = regions.masks.get("cube_area") if regions is not None else None

        dice_search = self._masked_gray(board.rgb_gray, dice_mask)
        cube_search = self._masked_gray(board.rgb_gray, cube_mask)
        dice_boxes = [b for b in self._find_square_candidates(dice_search) if self._box_center_in_mask(b, dice_mask)]
        cube_boxes = [b for b in self._find_square_candidates(cube_search) if self._box_center_in_mask(b, cube_mask)]

        dice: List[DiceObservation] = []
        cube: Optional[CubeObservation] = None

        for x1, y1, x2, y2 in dice_boxes:
            patch = board.rgb_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            dice_value, dice_conf = self._count_pips(patch)
            if dice_value is not None:
                obs = DiceObservation((x1, y1, x2, y2), dice_value, dice_conf)
                dice.append(obs)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 255, 40), 2)
                cv2.putText(overlay, f"d{dice_value}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1, cv2.LINE_AA)

        dice = sorted(dice, key=lambda d: d.confidence, reverse=True)[:2]

        for x1, y1, x2, y2 in cube_boxes:
            patch = board.rgb_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            area = (x2 - x1) * (y2 - y1)
            if not (self.config.cube_min_area_px <= area <= self.config.cube_max_area_px):
                continue
            cube_value, cube_conf, cube_text = self._read_cube_value(patch)
            cube = CubeObservation((x1, y1, x2, y2), cube_value, cube_conf, cube_text)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(overlay, f"cube:{cube_value if cube_value is not None else '?'}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1, cv2.LINE_AA)
            break

        confidence_parts = [d.confidence for d in dice]
        if cube is not None:
            confidence_parts.append(cube.confidence)
        return DiceCubeObservation(
            dice=dice,
            cube=cube,
            overlay_bgr=overlay,
            confidence=float(np.mean(confidence_parts)) if confidence_parts else 0.0,
        )


__all__ = ["DiceCubeConfig", "DiceCubeReader"]
