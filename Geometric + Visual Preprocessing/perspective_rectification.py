from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from backgammon_types import BoardLock, RectifiedBoard


@dataclass
class RectificationConfig:
    target_size_wh: Tuple[int, int] = (1200, 900)
    plane_fit_upper_quantile: float = 0.70
    fit_stride: int = 8
    min_plane_points: int = 500


def destination_corners(target_size_wh: Tuple[int, int]) -> np.ndarray:
    width, height = target_size_wh
    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )


def fit_depth_plane(depth_mm: np.ndarray, valid_mask: np.ndarray, cfg: RectificationConfig) -> Tuple[Tuple[float, float, float], np.ndarray]:
    ys, xs = np.indices(depth_mm.shape)
    valid = (valid_mask > 0) & np.isfinite(depth_mm) & (depth_mm > 0)

    if np.count_nonzero(valid) < cfg.min_plane_points:
        plane = (0.0, 0.0, float(np.nanmedian(depth_mm[valid])) if np.count_nonzero(valid) else 0.0)
        z_plane = np.full(depth_mm.shape, plane[2], dtype=np.float32)
        return plane, z_plane

    depth_valid = depth_mm[valid]
    threshold = float(np.quantile(depth_valid, cfg.plane_fit_upper_quantile))
    board_like = valid & (depth_mm >= threshold)

    xs_s = xs[board_like][:: cfg.fit_stride].astype(np.float32)
    ys_s = ys[board_like][:: cfg.fit_stride].astype(np.float32)
    zs_s = depth_mm[board_like][:: cfg.fit_stride].astype(np.float32)

    if len(xs_s) < 3:
        plane = (0.0, 0.0, float(np.nanmedian(depth_valid)))
        z_plane = np.full(depth_mm.shape, plane[2], dtype=np.float32)
        return plane, z_plane

    A = np.column_stack([xs_s, ys_s, np.ones_like(xs_s)])
    coeffs, _, _, _ = np.linalg.lstsq(A, zs_s, rcond=None)

    a, b, c = [float(v) for v in coeffs]
    z_plane = a * xs.astype(np.float32) + b * ys.astype(np.float32) + c
    return (a, b, c), z_plane


class PerspectiveRectifier:
    """
    Homography-based rectification plus a simple depth-plane fit.

    The depth-plane model makes it easy to convert raw depth into an
    approximate height-above-board map for downstream segmentation.
    """

    def __init__(self, config: Optional[RectificationConfig] = None) -> None:
        self.config = config or RectificationConfig()

    def rectify(self, rgb_bgr: np.ndarray, depth_mm: np.ndarray, board_lock: BoardLock) -> RectifiedBoard:
        if not board_lock.valid:
            raise ValueError("Board lock is not valid.")

        dst = destination_corners(self.config.target_size_wh)
        H = cv2.getPerspectiveTransform(board_lock.corners_xy.astype(np.float32), dst)
        Hinv = cv2.getPerspectiveTransform(dst, board_lock.corners_xy.astype(np.float32))

        width, height = self.config.target_size_wh
        rgb_rectified = cv2.warpPerspective(
            rgb_bgr,
            H,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        depth_rectified = cv2.warpPerspective(
            depth_mm.astype(np.float32),
            H,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        valid_mask = (depth_rectified > 0).astype(np.uint8)
        board_mask = np.ones((height, width), dtype=np.uint8)

        plane_coeffs, plane_depth = fit_depth_plane(depth_rectified, valid_mask, self.config)
        height_map_mm = np.maximum(plane_depth - depth_rectified, 0.0).astype(np.float32)
        height_map_mm[valid_mask == 0] = 0.0

        return RectifiedBoard(
            rgb_bgr=rgb_rectified,
            depth_mm=depth_rectified.astype(np.float32),
            valid_mask=valid_mask,
            height_map_mm=height_map_mm,
            homography=H.astype(np.float32),
            inverse_homography=Hinv.astype(np.float32),
            board_mask=board_mask,
            plane_coeffs=plane_coeffs,
        )
