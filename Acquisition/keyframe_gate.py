from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import time

import cv2
import numpy as np

from streams_module import OakSRStreams, depth_to_grayscale


RoiFrac = Tuple[float, float, float, float]


@dataclass
class KeyframePacket:
    timestamp: float
    frame_index: int

    left_raw: np.ndarray
    left_norm: np.ndarray
    left_gray: np.ndarray

    depth_raw: np.ndarray
    depth_gray: np.ndarray
    depth_valid_mask: np.ndarray
    near_mask: np.ndarray

    motion_score: float
    rgb_motion_score: float
    depth_motion_score: float
    novelty_score: float

    near_ratio: float
    invalid_ratio: float

    stable: bool
    occluded: bool
    keyframe: bool
    stable_count: int


class OakSRKeyframeGate:
    """
    Wraps streams_module.OakSRStreams and outputs processed frame packets
    plus keyframe decisions.

    Intended use:
      - Iterate all processed packets with iter_packets()
      - Or iterate only accepted keyframes with iter_keyframes()

    Main idea:
      - RGB stability gate via frame differencing
      - Depth hand/occlusion gate via near-field ratio + invalid depth ratio
      - Keyframe only after N stable frames and enough change from the last keyframe
    """

    def __init__(
        self,
        *,
        fps: float = 30.0,
        view_size: Tuple[int, int] = (640, 400),
        roi_frac: RoiFrac = (0.08, 0.08, 0.92, 0.92),
        motion_threshold: float = 2.5,
        min_stable_frames: int = 5,
        min_keyframe_gap_frames: int = 10,
        min_keyframe_delta: float = 1.5,
        hand_near_mm: int = 450,
        min_valid_mm: int = 80,
        max_invalid_ratio: float = 0.80,
        max_near_ratio: float = 0.03,
        ema_alpha: float = 0.2,
        wb_strength: float = 1.0,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        depth_blur_ksize: int = 5,
        enable_right: bool = False,
        depth_roi: Optional[Tuple[int, int, int, int]] = (120, 0, 550, 400),
        depth_pad: Tuple[int, int] = (40, 0),
        streams: Optional[OakSRStreams] = None,
    ) -> None:
        # Accept an externally-created stream object so callers such
        # as live_stream_viewer.py do not accidentally construct two DepthAI
        # pipelines before starting one. The earlier version did that, which
        # could leave the OAK-D SR in a failed boot state.
        self.streams = streams if streams is not None else OakSRStreams(
            enable_left=True,
            enable_right=enable_right,
            enable_depth=True,
            fps=fps,
            view_size=view_size,
            stereo_size=view_size,
            depth_roi=depth_roi,
            depth_pad=depth_pad,
            lr_check=True,
            subpixel=True,
            extended_disparity=False,
        )

        self.roi_frac = roi_frac
        self.motion_threshold = motion_threshold
        self.min_stable_frames = min_stable_frames
        self.min_keyframe_gap_frames = min_keyframe_gap_frames
        self.min_keyframe_delta = min_keyframe_delta

        self.hand_near_mm = hand_near_mm
        self.min_valid_mm = min_valid_mm
        self.max_invalid_ratio = max_invalid_ratio
        self.max_near_ratio = max_near_ratio

        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 0.99))
        self.wb_strength = float(np.clip(wb_strength, 0.0, 1.0))
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.depth_blur_ksize = max(3, depth_blur_ksize | 1)  # force odd

        self._frame_index = 0
        self._stable_count = 0
        self._last_keyframe_index = -10_000

        self._prev_left_gray_roi: Optional[np.ndarray] = None
        self._prev_depth_gray_roi: Optional[np.ndarray] = None
        self._last_keyframe_gray_roi: Optional[np.ndarray] = None

        self._running = False

    def start(self) -> "OakSRKeyframeGate":
        self.streams.start()
        self._running = True
        return self

    def stop(self) -> None:
        self.streams.stop()
        self._running = False

    def __enter__(self) -> "OakSRKeyframeGate":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @staticmethod
    def _gray_world_white_balance(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
        img = img_bgr.astype(np.float32)
        means = img.reshape(-1, 3).mean(axis=0)
        gray_mean = float(means.mean()) + 1e-6
        scale = gray_mean / (means + 1e-6)
        balanced = img * scale.reshape(1, 1, 3)
        mixed = img * (1.0 - strength) + balanced * strength
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def _normalize_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        wb = self._gray_world_white_balance(frame_bgr, strength=self.wb_strength)

        lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )
        l_eq = clahe.apply(l)

        merged = cv2.merge((l_eq, a, b))
        out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        out = cv2.GaussianBlur(out, (3, 3), 0)
        return out

    def _normalize_depth_gray(self, depth_raw: np.ndarray) -> np.ndarray:
        gray = depth_to_grayscale(
            depth_raw,
            near_percentile=3.0,
            far_percentile=95.0,
            invert=False,
        )
        gray = cv2.medianBlur(gray, self.depth_blur_ksize)
        return gray

    @staticmethod
    def _resize_to_match(src: np.ndarray, target_shape_hw: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
        h, w = target_shape_hw
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(src, (w, h), interpolation=interp)

    def _roi_bounds(self, h: int, w: int) -> Tuple[int, int, int, int]:
        x1f, y1f, x2f, y2f = self.roi_frac
        x1 = int(np.clip(x1f, 0.0, 1.0) * w)
        y1 = int(np.clip(y1f, 0.0, 1.0) * h)
        x2 = int(np.clip(x2f, 0.0, 1.0) * w)
        y2 = int(np.clip(y2f, 0.0, 1.0) * h)

        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)

        return x1, y1, x2, y2

    def _extract_roi(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = self._roi_bounds(h, w)
        return img[y1:y2, x1:x2]

    @staticmethod
    def _mean_absdiff(a: np.ndarray, b: np.ndarray) -> float:
        diff = cv2.absdiff(a, b)
        return float(diff.mean())

    def _compute_motion_scores(
        self,
        left_gray_roi: np.ndarray,
        depth_gray_roi: np.ndarray,
    ) -> Tuple[float, float, float]:
        if self._prev_left_gray_roi is None or self._prev_depth_gray_roi is None:
            rgb_motion = 999.0
            depth_motion = 999.0
        else:
            rgb_motion = self._mean_absdiff(left_gray_roi, self._prev_left_gray_roi)
            depth_motion = self._mean_absdiff(depth_gray_roi, self._prev_depth_gray_roi)

        combined = 0.75 * rgb_motion + 0.25 * depth_motion
        return combined, rgb_motion, depth_motion

    def _compute_novelty_score(self, left_gray_roi: np.ndarray) -> float:
        if self._last_keyframe_gray_roi is None:
            return 999.0
        return self._mean_absdiff(left_gray_roi, self._last_keyframe_gray_roi)

    def _compute_occlusion_metrics(
        self,
        depth_raw: np.ndarray,
        target_shape_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        valid_mask = (depth_raw >= self.min_valid_mm).astype(np.uint8)

        # Use a dynamic near-field threshold relative to the current board depth.
        # The earlier fixed threshold (e.g. 450 mm) treated most of the board as
        # an occluder when the board itself was around 390-430 mm from camera.
        # Here, only things substantially closer than the median board/ROI depth
        # are classed as near occlusions. This should catch hands while not
        # rejecting the board or normal checker stacks.
        roi_depth = self._extract_roi(depth_raw)
        roi_valid = roi_depth[roi_depth >= self.min_valid_mm]
        if roi_valid.size > 0:
            board_median_mm = float(np.median(roi_valid))
            dynamic_near_mm = min(float(self.hand_near_mm), board_median_mm - 40.0)
        else:
            dynamic_near_mm = float(self.hand_near_mm)

        near_mask = (
            (depth_raw >= self.min_valid_mm)
            & (depth_raw < dynamic_near_mm)
        ).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        near_mask = cv2.morphologyEx(near_mask * 255, cv2.MORPH_OPEN, kernel)
        near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_CLOSE, kernel)
        near_mask = (near_mask > 0).astype(np.uint8)

        valid_mask = self._resize_to_match(valid_mask, target_shape_hw, is_mask=True)
        near_mask = self._resize_to_match(near_mask, target_shape_hw, is_mask=True)

        valid_roi = self._extract_roi(valid_mask)
        near_roi = self._extract_roi(near_mask)

        invalid_ratio = 1.0 - float(valid_roi.mean())
        near_ratio = float(near_roi.mean())

        occluded = (near_ratio > self.max_near_ratio) or (invalid_ratio > self.max_invalid_ratio)
        return valid_mask, near_mask, near_ratio, invalid_ratio, occluded

    def get_packet(self, block: bool = True) -> Optional[KeyframePacket]:
        if not self._running:
            raise RuntimeError("Call start() before reading packets.")

        left_raw = self.streams.get_left_frame(block=block, use_cached=True)
        depth_raw = self.streams.get_depth_frame(block=block, use_cached=True)

        if left_raw is None or depth_raw is None:
            return None

        left_norm = self._normalize_rgb(left_raw)
        left_gray = cv2.cvtColor(left_norm, cv2.COLOR_BGR2GRAY)

        depth_gray = self._normalize_depth_gray(depth_raw)

        if depth_gray.shape[:2] != left_gray.shape[:2]:
            depth_gray = self._resize_to_match(depth_gray, left_gray.shape[:2], is_mask=False)

        left_gray_roi = self._extract_roi(left_gray)
        depth_gray_roi = self._extract_roi(depth_gray)

        motion_score, rgb_motion_score, depth_motion_score = self._compute_motion_scores(
            left_gray_roi,
            depth_gray_roi,
        )

        valid_mask, near_mask, near_ratio, invalid_ratio, occluded = self._compute_occlusion_metrics(
            depth_raw,
            left_gray.shape[:2],
        )

        stable = (motion_score <= self.motion_threshold) and not occluded

        if stable:
            self._stable_count += 1
        else:
            self._stable_count = 0

        novelty_score = self._compute_novelty_score(left_gray_roi)

        enough_gap = (self._frame_index - self._last_keyframe_index) >= self.min_keyframe_gap_frames
        keyframe = (
            self._stable_count >= self.min_stable_frames
            and enough_gap
            and not occluded
            and invalid_ratio <= self.max_invalid_ratio
            and novelty_score >= self.min_keyframe_delta
        )

        if keyframe:
            self._last_keyframe_gray_roi = left_gray_roi.copy()
            self._last_keyframe_index = self._frame_index

        self._prev_left_gray_roi = left_gray_roi.copy()
        self._prev_depth_gray_roi = depth_gray_roi.copy()

        packet = KeyframePacket(
            timestamp=time.time(),
            frame_index=self._frame_index,
            left_raw=left_raw,
            left_norm=left_norm,
            left_gray=left_gray,
            depth_raw=depth_raw,
            depth_gray=depth_gray,
            depth_valid_mask=valid_mask,
            near_mask=near_mask,
            motion_score=motion_score,
            rgb_motion_score=rgb_motion_score,
            depth_motion_score=depth_motion_score,
            novelty_score=novelty_score,
            near_ratio=near_ratio,
            invalid_ratio=invalid_ratio,
            stable=stable,
            occluded=occluded,
            keyframe=keyframe,
            stable_count=self._stable_count,
        )

        self._frame_index += 1
        return packet

    def iter_packets(self, block: bool = True) -> Iterator[KeyframePacket]:
        while self._running:
            packet = self.get_packet(block=block)
            if packet is not None:
                yield packet

    def iter_keyframes(self, block: bool = True) -> Iterator[KeyframePacket]:
        while self._running:
            packet = self.get_packet(block=block)
            if packet is not None and packet.keyframe:
                yield packet


if __name__ == "__main__":
    gate = OakSRKeyframeGate(
        motion_threshold=2.5,
        min_stable_frames=5,
        min_keyframe_gap_frames=10,
        min_keyframe_delta=1.5,
        hand_near_mm=450,
        max_near_ratio=0.03,
        max_invalid_ratio=0.80,
    ).start()

    try:
        for packet in gate.iter_packets():
            rgb_vis = packet.left_norm.copy()

            h, w = rgb_vis.shape[:2]
            x1 = int(0.08 * w)
            y1 = int(0.08 * h)
            x2 = int(0.92 * w)
            y2 = int(0.92 * h)
            cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

            status = f"stable={packet.stable} occluded={packet.occluded} keyframe={packet.keyframe}"
            stats = (
                f"motion={packet.motion_score:.2f} "
                f"near={packet.near_ratio:.3f} "
                f"invalid={packet.invalid_ratio:.3f} "
                f"stable_count={packet.stable_count}"
            )

            cv2.putText(rgb_vis, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(rgb_vis, stats, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            near_vis = (packet.near_mask * 255).astype(np.uint8)
            valid_vis = (packet.depth_valid_mask * 255).astype(np.uint8)

            cv2.imshow("Left Normalised", rgb_vis)
            cv2.imshow("Depth BW", packet.depth_gray)
            cv2.imshow("Near/Occlusion Mask", near_vis)
            cv2.imshow("Depth Valid Mask", valid_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        gate.stop()
        cv2.destroyAllWindows()
