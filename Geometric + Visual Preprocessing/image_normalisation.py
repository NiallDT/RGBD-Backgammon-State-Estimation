from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from backgammon_types import NormalisedBoard, RectifiedBoard
from streams_module import depth_to_grayscale


@dataclass
class NormalisationConfig:
    wb_strength: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    rgb_blur_ksize: int = 3
    depth_median_ksize: int = 5
    depth_bilateral_d: int = 5
    depth_bilateral_sigma_color: float = 25.0
    depth_bilateral_sigma_space: float = 25.0
    max_height_mm: float = 40.0


def gray_world_white_balance(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gray_mean = float(means.mean()) + 1e-6
    scale = gray_mean / (means + 1e-6)
    balanced = img * scale.reshape(1, 1, 3)
    mixed = img * (1.0 - strength) + balanced * strength
    return np.clip(mixed, 0, 255).astype(np.uint8)


def normalize_rgb(rgb_bgr: np.ndarray, cfg: NormalisationConfig) -> Tuple[np.ndarray, np.ndarray]:
    wb = gray_world_white_balance(rgb_bgr, strength=cfg.wb_strength)

    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid,
    )
    l_eq = clahe.apply(l_chan)

    merged = cv2.merge((l_eq, a_chan, b_chan))
    rgb_norm = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    rgb_norm = cv2.GaussianBlur(rgb_norm, (cfg.rgb_blur_ksize | 1, cfg.rgb_blur_ksize | 1), 0)

    gray = cv2.cvtColor(rgb_norm, cv2.COLOR_BGR2GRAY)
    return rgb_norm, gray


def normalize_depth(depth_mm: np.ndarray, valid_mask: np.ndarray, cfg: NormalisationConfig) -> Tuple[np.ndarray, np.ndarray]:
    depth = depth_mm.astype(np.float32).copy()
    depth[valid_mask == 0] = 0.0

    if cfg.depth_median_ksize >= 3:
        depth = cv2.medianBlur(depth, cfg.depth_median_ksize | 1)

    depth = cv2.bilateralFilter(
        depth,
        d=cfg.depth_bilateral_d,
        sigmaColor=cfg.depth_bilateral_sigma_color,
        sigmaSpace=cfg.depth_bilateral_sigma_space,
    )
    depth[valid_mask == 0] = 0.0

    depth_gray = depth_to_grayscale(depth, near_percentile=3.0, far_percentile=95.0, invert=False)
    return depth.astype(np.float32), depth_gray


def normalize_height_map(height_map_mm: np.ndarray, valid_mask: np.ndarray, max_height_mm: float = 40.0) -> np.ndarray:
    height = np.clip(height_map_mm.astype(np.float32), 0.0, max_height_mm)
    norm = (height * (255.0 / max_height_mm)).astype(np.uint8)
    norm[valid_mask == 0] = 0
    return norm


class BoardNormaliser:
    def __init__(self, config: Optional[NormalisationConfig] = None) -> None:
        self.config = config or NormalisationConfig()

    def normalize(self, rectified: RectifiedBoard) -> NormalisedBoard:
        rgb_norm, gray = normalize_rgb(rectified.rgb_bgr, self.config)
        depth_norm, depth_gray = normalize_depth(rectified.depth_mm, rectified.valid_mask, self.config)
        height_uint8 = normalize_height_map(
            rectified.height_map_mm,
            rectified.valid_mask,
            max_height_mm=self.config.max_height_mm,
        )

        return NormalisedBoard(
            rgb_bgr=rgb_norm,
            rgb_gray=gray,
            depth_mm=depth_norm,
            depth_gray=depth_gray,
            valid_mask=rectified.valid_mask.copy(),
            height_map_mm=rectified.height_map_mm.copy(),
            height_uint8=height_uint8,
        )
