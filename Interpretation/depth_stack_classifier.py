from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

HeightStat = Literal["median", "p75", "p90", "max", "top35", "top25"]
RoiFrac = Tuple[float, float, float, float]
RectFracs = Tuple[RoiFrac, ...]


@dataclass
class ChipCandidate:
    x: float
    y: float
    radius: float
    area_px: float
    circularity: float
    source: str = "unknown"
    support_ratio: float = 0.0
    support_pixels: int = 0
    valid_depth_ratio: float = 0.0
    height_median_mm: float = 0.0
    height_p75_mm: float = 0.0
    height_p90_mm: float = 0.0
    height_max_mm: float = 0.0
    height_top35_mm: float = 0.0
    height_top25_mm: float = 0.0
    height_used_mm: float = 0.0
    raw_stack_count: int = 0
    stack_count: int = 0
    track_id: int = -1
    confidence: float = 0.0


@dataclass
class StackClassificationResult:
    candidates: List[ChipCandidate]
    height_mm: np.ndarray
    valid_overlap_mask: np.ndarray
    stable_depth_support_mask: np.ndarray
    rgb_restored_circle_mask: np.ndarray
    rgb_candidate_mask: np.ndarray
    piece_presence_mask: np.ndarray
    stack_class_bgr: np.ndarray
    overlay_bgr: np.ndarray
    diagnostics: Dict[str, float]


class EmptyBoardBaseline:
    """Median empty-board depth reference plus per-pixel depth-noise estimate."""

    def __init__(
        self,
        samples_required: int = 60,
        min_valid_ratio: float = 0.15,
        noise_percentile: float = 90.0,
        min_noise_floor_mm: float = 0.5,
    ) -> None:
        self.samples_required = int(samples_required)
        self.min_valid_ratio = float(min_valid_ratio)
        self.noise_percentile = float(noise_percentile)
        self.min_noise_floor_mm = float(min_noise_floor_mm)
        self.samples: List[np.ndarray] = []
        self.reference_depth_mm: Optional[np.ndarray] = None
        self.noise_floor_mm: Optional[np.ndarray] = None

    @property
    def ready(self) -> bool:
        return self.reference_depth_mm is not None

    def clear(self) -> None:
        self.samples.clear()
        self.reference_depth_mm = None
        self.noise_floor_mm = None

    def add_sample(self, depth_mm: np.ndarray) -> bool:
        valid_ratio = float(np.count_nonzero(depth_mm)) / float(depth_mm.size)
        if valid_ratio < self.min_valid_ratio:
            return False
        self.samples.append(depth_mm.copy())
        if len(self.samples) < self.samples_required:
            return False

        stack = np.stack([f.astype(np.float32) for f in self.samples], axis=0)
        stack[stack <= 0] = np.nan
        baseline_nan = np.nanmedian(stack, axis=0)
        dev = np.abs(stack - baseline_nan[None, :, :])
        noise = np.nanpercentile(dev, self.noise_percentile, axis=0)
        noise[~np.isfinite(noise)] = 0
        noise = np.maximum(noise, self.min_noise_floor_mm)

        baseline = baseline_nan.copy()
        baseline[~np.isfinite(baseline)] = 0
        self.reference_depth_mm = baseline.astype(np.float32)
        self.noise_floor_mm = noise.astype(np.float32)
        self.samples.clear()
        return True

    def height_above_board(self, depth_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.reference_depth_mm is None:
            raise RuntimeError("No empty-board baseline has been captured yet.")
        if depth_mm.shape != self.reference_depth_mm.shape:
            depth_mm = cv2.resize(
                depth_mm.astype(np.float32),
                (self.reference_depth_mm.shape[1], self.reference_depth_mm.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        current = depth_mm.astype(np.float32)
        baseline = self.reference_depth_mm
        valid = (current > 0) & (baseline > 0)
        height = baseline - current
        height[~valid] = 0
        height[height < 0] = 0
        return height.astype(np.float32), valid.astype(np.uint8)


def depth_to_gray_fixed(depth_mm: np.ndarray, dmin: float = 360.0, dmax: float = 435.0) -> np.ndarray:
    d = depth_mm.astype(np.float32)
    out = np.clip((d - dmin) * 255.0 / max(dmax - dmin, 1.0), 0, 255).astype(np.uint8)
    out[depth_mm <= 0] = 0
    return out


def height_to_gray(height_mm: np.ndarray, height_max_mm: float = 30.0) -> np.ndarray:
    h = np.clip(height_mm.astype(np.float32), 0, max(height_max_mm, 1.0))
    return (h * 255.0 / max(height_max_mm, 1.0)).astype(np.uint8)


def _rect_mask(shape_hw: Tuple[int, int], frac: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if frac is None:
        return mask
    x1f, y1f, x2f, y2f = frac
    x1 = int(np.clip(x1f, 0.0, 1.0) * w)
    y1 = int(np.clip(y1f, 0.0, 1.0) * h)
    x2 = int(np.clip(x2f, 0.0, 1.0) * w)
    y2 = int(np.clip(y2f, 0.0, 1.0) * h)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def _roi_mask(shape_hw: Tuple[int, int], roi_frac: Optional[RoiFrac]) -> np.ndarray:
    if roi_frac is None:
        return np.ones(shape_hw, dtype=np.uint8) * 255
    return _rect_mask(shape_hw, roi_frac)


def _allowed_mask(shape_hw: Tuple[int, int], roi_frac: Optional[RoiFrac], exclusion_fracs: RectFracs = ()) -> np.ndarray:
    mask = _roi_mask(shape_hw, roi_frac)
    for frac in exclusion_fracs:
        ex = _rect_mask(shape_hw, frac)
        mask[ex > 0] = 0
    return mask


def clean_mask(mask: np.ndarray, open_px: int = 3, close_px: int = 5, min_area_px: int = 40) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    if open_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px | 1, open_px | 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px | 1, close_px | 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for lab in range(1, n):
        if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area_px:
            out[labels == lab] = 255
    return out


class TemporalMaskFilter:
    """Pixel-level temporal persistence filter for unstable depth support masks."""

    def __init__(
        self,
        window: int = 3,
        required: int = 2,
        *,
        open_px: int = 3,
        close_px: int = 5,
        min_area_px: int = 30,
    ) -> None:
        self.window = max(1, int(window))
        self.required = max(1, min(int(required), self.window))
        self.open_px = int(open_px)
        self.close_px = int(close_px)
        self.min_area_px = int(min_area_px)
        self.history: Deque[np.ndarray] = deque(maxlen=self.window)

    @property
    def history_size(self) -> int:
        return len(self.history)

    def reset(self) -> None:
        self.history.clear()

    def update(self, mask: np.ndarray) -> np.ndarray:
        current = (mask > 0).astype(np.uint8)
        self.history.append(current)
        stack = np.stack(list(self.history), axis=0).astype(np.uint8)
        count = np.sum(stack, axis=0)
        req = min(self.required, len(self.history))
        persistent = (count >= req).astype(np.uint8) * 255
        return clean_mask(persistent, open_px=self.open_px, close_px=self.close_px, min_area_px=self.min_area_px)



class StackCountSmoother:
    """
    Temporally smooths the 1-high vs 2-high stack label for RGB-restored chip candidates.

    This is intentionally separate from the detection gate. A candidate can be
    detected with permissive depth support, while the stack label itself changes
    only after repeated height evidence. That prevents true two-high stacks from
    flickering between 1 and 2 when passive stereo under-reads a few frames.
    """

    def __init__(
        self,
        *,
        window: int = 5,
        promote_votes: int = 2,
        demote_votes: int = 4,
        hold_misses: int = 2,
        promote_height_mm: float = 13.0,
        demote_height_mm: float = 11.0,
        match_distance_px: float = 28.0,
    ) -> None:
        self.window = max(1, int(window))
        self.promote_votes = max(1, int(promote_votes))
        self.demote_votes = max(1, int(demote_votes))
        self.hold_misses = max(0, int(hold_misses))
        self.promote_height_mm = float(promote_height_mm)
        self.demote_height_mm = float(demote_height_mm)
        self.match_distance_px = float(match_distance_px)
        self._tracks: Dict[int, Dict[str, object]] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _match_track(self, cand: ChipCandidate, used: set[int]) -> int:
        best_id = -1
        best_d = float("inf")
        max_d = max(self.match_distance_px, cand.radius * 1.8)
        for tid, tr in self._tracks.items():
            if tid in used:
                continue
            dx = float(tr["x"]) - cand.x
            dy = float(tr["y"]) - cand.y
            d = float((dx * dx + dy * dy) ** 0.5)
            if d < best_d and d <= max_d:
                best_d = d
                best_id = tid
        if best_id >= 0:
            return best_id
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "x": cand.x,
            "y": cand.y,
            "stable": max(1, int(cand.stack_count)),
            "raw_hist": deque(maxlen=self.window),
            "height_hist": deque(maxlen=self.window),
            "misses": 0,
        }
        return tid

    def update(self, candidates: List[ChipCandidate]) -> List[ChipCandidate]:
        used: set[int] = set()
        active_ids: set[int] = set()

        for cand in candidates:
            raw = int(np.clip(cand.stack_count, 1, 2))
            cand.raw_stack_count = raw
            tid = self._match_track(cand, used)
            used.add(tid)
            active_ids.add(tid)
            tr = self._tracks[tid]

            raw_hist: Deque[int] = tr["raw_hist"]  # type: ignore[assignment]
            height_hist: Deque[float] = tr["height_hist"]  # type: ignore[assignment]
            raw_hist.append(raw)
            height_hist.append(float(cand.height_used_mm))

            prev = int(tr.get("stable", raw))
            high_votes = sum(1 for r, h in zip(raw_hist, height_hist) if int(r) >= 2 or float(h) >= self.promote_height_mm)
            low_votes = sum(1 for h in height_hist if float(h) <= self.demote_height_mm)
            one_votes = sum(1 for r in raw_hist if int(r) <= 1)

            if prev >= 2:
                # Once a chip has become a two-stack, require repeated low-height
                # evidence before demoting. This fixes 2 -> 1 flicker on edges.
                if low_votes >= self.demote_votes and one_votes >= self.demote_votes:
                    stable = 1
                else:
                    stable = 2
            else:
                if high_votes >= self.promote_votes:
                    stable = 2
                else:
                    stable = 1

            tr["stable"] = int(stable)
            # Slowly update the track centre so it follows real candidate movement
            # but does not jump wildly on a noisy frame.
            tr["x"] = 0.65 * float(tr["x"]) + 0.35 * cand.x
            tr["y"] = 0.65 * float(tr["y"]) + 0.35 * cand.y
            tr["misses"] = 0

            cand.track_id = tid
            cand.stack_count = int(stable)

        # Age unmatched tracks and remove old ones. Tracks are only used to smooth
        # labels for candidates that are actually detected; held tracks are not
        # hallucinated into the output.
        for tid in list(self._tracks.keys()):
            if tid not in active_ids:
                self._tracks[tid]["misses"] = int(self._tracks[tid].get("misses", 0)) + 1
                if int(self._tracks[tid]["misses"]) > self.hold_misses:
                    del self._tracks[tid]

        return candidates

def build_height_support_mask(
    height_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_piece_height_mm: float,
    max_piece_height_mm: float,
    roi_frac: Optional[RoiFrac],
    exclusion_fracs: RectFracs = (),
    noise_floor_mm: Optional[np.ndarray] = None,
    noise_margin_mm: float = 1.5,
    min_area_px: int = 24,
) -> np.ndarray:
    """Depth-only temporal support. This is only an anchor, not the final piece shape."""
    height = height_mm.astype(np.float32).copy()
    height[valid_mask == 0] = 0
    try:
        height_smooth = cv2.medianBlur(height, 3)
    except cv2.error:
        height_smooth = height

    if noise_floor_mm is not None:
        noise = noise_floor_mm.astype(np.float32)
        if noise.shape != height_smooth.shape:
            noise = cv2.resize(noise, (height_smooth.shape[1], height_smooth.shape[0]), interpolation=cv2.INTER_NEAREST)
        min_height_map = np.maximum(float(min_piece_height_mm), noise + float(noise_margin_mm))
    else:
        min_height_map = np.full_like(height_smooth, float(min_piece_height_mm), dtype=np.float32)

    raw = ((height_smooth >= min_height_map) & (height_smooth <= float(max_piece_height_mm)) & (valid_mask > 0)).astype(np.uint8) * 255
    raw = cv2.bitwise_and(raw, _allowed_mask(raw.shape[:2], roi_frac, exclusion_fracs))
    raw = clean_mask(raw, open_px=1, close_px=3, min_area_px=max(4, int(min_area_px)))
    return raw


def _top_band_stat(values: np.ndarray, keep_fraction: float) -> float:
    if values.size == 0:
        return 0.0
    keep_fraction = float(np.clip(keep_fraction, 0.05, 1.0))
    cutoff = np.percentile(values, 100.0 * (1.0 - keep_fraction))
    top = values[values >= cutoff]
    if top.size == 0:
        top = values
    return float(np.median(top))


def classify_stack_count(height_mm: float, chip_thickness_mm: float, min_piece_height_mm: float) -> int:
    if height_mm < max(1.0, min_piece_height_mm):
        return 0
    return max(1, int(round(height_mm / max(chip_thickness_mm, 1.0))))


def _circle_mask(shape_hw: Tuple[int, int], x: float, y: float, r: float, scale: float = 1.0) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.circle(mask, (int(round(x)), int(round(y))), max(1, int(round(r * scale))), 255, -1)
    return mask


def _support_stats_for_circle(support: np.ndarray, valid: np.ndarray, cand: ChipCandidate, inner_scale: float = 0.92) -> Tuple[float, int, float]:
    cmask = _circle_mask(support.shape[:2], cand.x, cand.y, cand.radius, inner_scale)
    area = max(1, int(np.count_nonzero(cmask)))
    support_px = int(np.count_nonzero((support > 0) & (cmask > 0)))
    valid_px = int(np.count_nonzero((valid > 0) & (cmask > 0)))
    return support_px / float(area), support_px, valid_px / float(area)


def _candidate_key(c: ChipCandidate) -> Tuple[float, float, float, float]:
    source_bonus = 2.0 if c.source.startswith("rgb") else 0.0
    return (source_bonus + c.support_ratio, float(c.support_pixels), c.circularity, -abs(c.radius))


def _nms_candidates(candidates: List[ChipCandidate], min_center_dist_factor: float = 1.18) -> List[ChipCandidate]:
    """Keep one centre per physical chip, but allow touching chips about 2 radii apart."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=_candidate_key, reverse=True)
    kept: List[ChipCandidate] = []
    for cand in ordered:
        duplicate = False
        for old in kept:
            dist = float(np.hypot(cand.x - old.x, cand.y - old.y))
            # Same physical stack fragments often land within about one chip radius;
            # touching chips should be closer to two radii apart and survive this.
            min_dist = min(cand.radius, old.radius) * min_center_dist_factor
            if dist < min_dist:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def _build_rgb_chip_mask(rgb_bgr: np.ndarray, roi_frac: Optional[RoiFrac], exclusion_fracs: RectFracs = ()) -> np.ndarray:
    """Bright low-saturation mask for the current pale/white diagnostic chips."""
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # White/off-white chips: bright, relatively unsaturated, roughly neutral in Lab.
    mask = ((v >= 125) & (s <= 95) & (l >= 120) & (np.abs(a.astype(np.int16) - 128) <= 28) & (np.abs(b.astype(np.int16) - 128) <= 38)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, _allowed_mask(mask.shape[:2], roi_frac, exclusion_fracs))
    mask = clean_mask(mask, open_px=3, close_px=5, min_area_px=30)
    return mask


def _component_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def _split_component_by_distance(
    comp_mask: np.ndarray,
    *,
    expected_radius_px: float,
    min_radius_px: float,
    max_radius_px: float,
) -> List[ChipCandidate]:
    dist = cv2.distanceTransform((comp_mask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []
    max_k = max(5, int(round(expected_radius_px * 1.05)) | 1)
    dil = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_k, max_k)))
    peaks = (dist == dil) & (dist > max(2.0, expected_radius_px * 0.42)) & (comp_mask > 0)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
    out: List[ChipCandidate] = []
    area = float(np.count_nonzero(comp_mask))
    for lab in range(1, n):
        x, y = cents[lab]
        r = float(np.clip(expected_radius_px, min_radius_px, max_radius_px))
        out.append(ChipCandidate(float(x), float(y), r, area, 1.0, source="rgb-split"))
    return out


def _rgb_circle_candidates(
    rgb_bgr: np.ndarray,
    stable_support: np.ndarray,
    *,
    roi_frac: Optional[RoiFrac],
    exclusion_fracs: RectFracs = (),
    expected_radius_px: float,
    min_radius_px: float,
    max_radius_px: float,
    split_touching: bool,
) -> Tuple[List[ChipCandidate], np.ndarray]:
    rgb_mask = _build_rgb_chip_mask(rgb_bgr, roi_frac, exclusion_fracs)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(rgb_mask, connectivity=8)
    out: List[ChipCandidate] = []
    expected_area = float(np.pi * expected_radius_px * expected_radius_px)
    min_area = max(20.0, np.pi * min_radius_px * min_radius_px * 0.40)
    max_area_single = np.pi * max_radius_px * max_radius_px * 1.35

    for lab in range(1, n):
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == lab).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        circ = _component_circularity(contour)
        x, y = cents[lab]
        (_, _), rr = cv2.minEnclosingCircle(contour)
        bbox_w = float(stats[lab, cv2.CC_STAT_WIDTH])
        bbox_h = float(stats[lab, cv2.CC_STAT_HEIGHT])
        elong = max(bbox_w, bbox_h) / max(1.0, min(bbox_w, bbox_h))

        should_split = split_touching and (area > expected_area * 1.35 or elong > 1.45 or area > max_area_single)
        if should_split:
            split = _split_component_by_distance(
                comp,
                expected_radius_px=expected_radius_px,
                min_radius_px=min_radius_px,
                max_radius_px=max_radius_px,
            )
            if len(split) >= 2:
                out.extend(split)
                continue

        # Use the expected physical radius when available so small depth/RGB holes do
        # not shrink the diagnostic footprint.
        r_est = expected_radius_px if expected_radius_px > 0 else np.sqrt(area / np.pi)
        r = float(np.clip(max(min_radius_px, min(max_radius_px, r_est, rr * 1.05)), min_radius_px, max_radius_px))
        out.append(ChipCandidate(float(x), float(y), r, area, circ, source="rgb-component"))
    return out, rgb_mask


def _support_seed_candidates(
    support: np.ndarray,
    *,
    expected_radius_px: float,
    min_radius_px: float,
    max_radius_px: float,
    min_component_area_px: int,
) -> List[ChipCandidate]:
    n, labels, stats, cents = cv2.connectedComponentsWithStats((support > 0).astype(np.uint8) * 255, connectivity=8)
    out: List[ChipCandidate] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_component_area_px:
            continue
        x, y = cents[lab]
        r = float(np.clip(expected_radius_px, min_radius_px, max_radius_px))
        out.append(ChipCandidate(float(x), float(y), r, float(area), 0.5, source="depth-seed"))
    return out


def _hough_candidates_constrained_by_rgb_and_support(
    rgb_bgr: np.ndarray,
    rgb_mask: np.ndarray,
    stable_support: np.ndarray,
    *,
    min_radius_px: float,
    max_radius_px: float,
    min_support_pixels: int,
) -> List[ChipCandidate]:
    search = cv2.bitwise_and(rgb_mask, cv2.dilate((stable_support > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))))
    if np.count_nonzero(search) == 0:
        return []
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    masked = cv2.bitwise_and(gray, gray, mask=cv2.dilate(search, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))))
    circles = cv2.HoughCircles(masked, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(8.0, min_radius_px * 1.8), param1=80, param2=18, minRadius=int(min_radius_px), maxRadius=int(max_radius_px))
    if circles is None:
        return []
    out: List[ChipCandidate] = []
    for x, y, r in np.round(circles[0]).astype(np.float32):
        cand = ChipCandidate(float(x), float(y), float(r), float(np.pi * r * r), 1.0, source="rgb-hough")
        _, support_px, _ = _support_stats_for_circle(stable_support, np.ones_like(stable_support, dtype=np.uint8), cand, inner_scale=0.92)
        if support_px >= min_support_pixels:
            out.append(cand)
    return out


def generate_restored_circle_candidates(
    rgb_bgr: np.ndarray,
    stable_support: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_radius_px: float,
    max_radius_px: float,
    expected_radius_px: Optional[float],
    min_candidate_support_ratio: float,
    min_support_pixels: int,
    use_hough: bool,
    split_touching: bool,
    roi_frac: Optional[RoiFrac],
    exclusion_fracs: RectFracs = (),
) -> Tuple[List[ChipCandidate], np.ndarray, np.ndarray, int]:
    if expected_radius_px is None or expected_radius_px <= 0:
        expected_radius_px = (float(min_radius_px) + float(max_radius_px)) * 0.5

    proposals: List[ChipCandidate] = []
    rgb_props, rgb_mask = _rgb_circle_candidates(
        rgb_bgr,
        stable_support,
        roi_frac=roi_frac,
        exclusion_fracs=exclusion_fracs,
        expected_radius_px=float(expected_radius_px),
        min_radius_px=min_radius_px,
        max_radius_px=max_radius_px,
        split_touching=split_touching,
    )
    proposals.extend(rgb_props)

    if use_hough:
        proposals.extend(_hough_candidates_constrained_by_rgb_and_support(rgb_bgr, rgb_mask, stable_support, min_radius_px=min_radius_px, max_radius_px=max_radius_px, min_support_pixels=min_support_pixels))

    # Fallback: if RGB misses a chip due blur/reflection, still allow a depth seed,
    # but it is lower priority and gets suppressed by RGB candidates in NMS.
    proposals.extend(_support_seed_candidates(stable_support, expected_radius_px=float(expected_radius_px), min_radius_px=min_radius_px, max_radius_px=max_radius_px, min_component_area_px=max(4, min_support_pixels)))
    proposed_count = len(proposals)

    accepted: List[ChipCandidate] = []
    for cand in proposals:
        support_ratio, support_px, valid_ratio = _support_stats_for_circle(stable_support, valid_mask, cand, inner_scale=0.92)
        cand.support_ratio = float(support_ratio)
        cand.support_pixels = int(support_px)
        cand.valid_depth_ratio = float(valid_ratio)

        # RGB candidates are already shape-valid, so a small amount of temporally
        # stable depth support is enough. Depth-only seeds remain stricter.
        if cand.source.startswith("rgb"):
            if support_px >= min_support_pixels and valid_ratio >= 0.01:
                accepted.append(cand)
        else:
            if support_px >= max(min_support_pixels, 8) and support_ratio >= min_candidate_support_ratio:
                accepted.append(cand)

    accepted = _nms_candidates(accepted, min_center_dist_factor=1.18)

    candidate_mask = np.zeros_like(stable_support)
    for cand in accepted:
        cv2.circle(candidate_mask, (int(round(cand.x)), int(round(cand.y))), max(2, int(round(cand.radius))), 255, -1)
    return accepted, candidate_mask, rgb_mask, proposed_count


def _height_values_in_candidate(
    height_mm: np.ndarray,
    valid_mask: np.ndarray,
    cand: ChipCandidate,
    *,
    min_depth_pixels: int,
    min_depth_ratio: float,
    sample_scale: float = 0.72,
) -> Tuple[np.ndarray, float]:
    mask = _circle_mask(height_mm.shape[:2], cand.x, cand.y, cand.radius, sample_scale)
    total = max(1, int(np.count_nonzero(mask)))
    good = (mask > 0) & (valid_mask > 0) & (height_mm > 0)
    ratio = float(np.count_nonzero(good)) / float(total)
    vals = height_mm[good].astype(np.float32)
    if vals.size < min_depth_pixels or ratio < min_depth_ratio:
        mask = _circle_mask(height_mm.shape[:2], cand.x, cand.y, cand.radius, 0.92)
        total = max(1, int(np.count_nonzero(mask)))
        good = (mask > 0) & (valid_mask > 0) & (height_mm > 0)
        ratio = float(np.count_nonzero(good)) / float(total)
        vals = height_mm[good].astype(np.float32)
    return vals, ratio


def _estimate_stack_from_distribution(vals: np.ndarray, chip_thickness_mm: float, min_piece_height_mm: float, max_piece_height_mm: float, requested_height: float) -> Tuple[int, float]:
    """Classify stack height from the distribution, not from a few high spikes."""
    t = max(float(chip_thickness_mm), 1.0)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= max(1.0, min_piece_height_mm * 0.65)) & (vals <= max_piece_height_mm)]
    if vals.size == 0:
        return 0, 0.0

    max_stack = max(1, min(8, int(np.ceil(max_piece_height_mm / t))))
    band_counts = []
    band_medians = []
    for k in range(1, max_stack + 1):
        centre = k * t
        lo = max(max(1.0, min_piece_height_mm * 0.65), centre - 0.45 * t)
        hi = centre + 0.55 * t
        band = vals[(vals >= lo) & (vals <= hi)]
        band_counts.append(int(band.size))
        band_medians.append(float(np.median(band)) if band.size else centre)

    best_idx = int(np.argmax(band_counts))
    best_count = band_counts[best_idx]
    best_stack = best_idx + 1
    n = int(vals.size)

    if best_count >= max(3, int(0.18 * n)):
        return best_stack, band_medians[best_idx]

    # Fallback: use the requested robust statistic, but round conservatively. This
    # helps very sparse edge stacks while avoiding single-chip overestimation.
    est = max(0, classify_stack_count(requested_height, t, min_piece_height_mm))
    if est > 1:
        # Require at least some evidence in the estimated band.
        idx = min(est, len(band_counts)) - 1
        if band_counts[idx] < max(2, int(0.08 * n)) and band_counts[0] >= max(2, band_counts[idx]):
            est = 1
            requested_height = band_medians[0] if band_counts[0] else float(np.median(vals))
    return est, float(requested_height)



def _estimate_stack_1_or_2(
    vals: np.ndarray,
    *,
    min_piece_height_mm: float,
    max_piece_height_mm: float,
    promote_height_mm: float,
    demote_height_mm: float,
    requested_height: float,
) -> Tuple[int, float]:
    """Return only 0, 1, or 2 for this project.

    The measured depth of a two-chip stack is often lower than the ideal 20 mm,
    especially near board edges. Instead of rounding by chip thickness, this uses
    a robust upper-band statistic and a lower promote threshold.
    """
    vals = vals[np.isfinite(vals)].astype(np.float32)
    vals = vals[(vals >= max(1.0, min_piece_height_mm * 0.6)) & (vals <= max_piece_height_mm)]
    if vals.size == 0:
        return 0, 0.0

    p50 = float(np.percentile(vals, 50))
    p75 = float(np.percentile(vals, 75))
    p90 = float(np.percentile(vals, 90))
    top35 = _top_band_stat(vals, 0.35)
    top25 = _top_band_stat(vals, 0.25)
    used = max(float(requested_height), top35)

    high_pixels = int(np.count_nonzero(vals >= promote_height_mm))
    high_ratio = high_pixels / max(float(vals.size), 1.0)

    # Promote on a robust top-face estimate, not a single spike. The p75/top35
    # tests handle dense centre-board readings; p90 plus a few high pixels helps
    # sparse edge/corner readings.
    is_two = (
        top35 >= promote_height_mm
        or p75 >= promote_height_mm
        or (p90 >= promote_height_mm and high_pixels >= max(2, int(0.05 * vals.size)))
        or high_ratio >= 0.12
    )

    # A genuinely low top band is one-high. Borderline values are handled by the
    # temporal smoother; without a smoother they remain conservative.
    if not is_two and top25 <= demote_height_mm:
        return 1, used
    return (2 if is_two else 1), used

def classify_rgb_candidates_by_depth(
    rgb_bgr: np.ndarray,
    depth_mm: np.ndarray,
    baseline: EmptyBoardBaseline,
    *,
    chip_thickness_mm: float = 10.0,
    min_piece_height_mm: float = 3.0,
    max_piece_height_mm: float = 45.0,
    height_stat: HeightStat = "top35",
    min_depth_pixels: int = 8,
    min_depth_ratio: float = 0.025,
    height_max_visual_mm: float = 30.0,
    min_radius_px: float = 7.0,
    max_radius_px: float = 30.0,
    expected_radius_px: Optional[float] = None,
    split_touching: bool = True,
    use_hough: bool = False,
    min_candidate_support_ratio: float = 0.025,
    min_support_pixels: int = 6,
    draw_rejected_candidates: bool = False,
    roi_frac: Optional[RoiFrac] = (0.18, 0.00, 0.88, 1.00),
    middle_exclusion_frac: Optional[Tuple[float, float, float, float]] = (0.00, 0.40, 1.00, 0.60),
    temporal_filter: Optional[TemporalMaskFilter] = None,
    stack_smoother: Optional[StackCountSmoother] = None,
    stack_promote_height_mm: float = 16.5,
    stack_demote_height_mm: float = 14.5,
    noise_margin_mm: float = 1.5,
) -> StackClassificationResult:
    height_mm, valid_overlap = baseline.height_above_board(depth_mm)

    if height_mm.shape[:2] != rgb_bgr.shape[:2]:
        height_for_rgb = cv2.resize(height_mm, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_for_rgb = cv2.resize(valid_overlap, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        noise_floor = cv2.resize(baseline.noise_floor_mm, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) if baseline.noise_floor_mm is not None else None
    else:
        height_for_rgb = height_mm
        valid_for_rgb = valid_overlap
        noise_floor = baseline.noise_floor_mm

    exclusion_fracs: RectFracs = tuple(frac for frac in (middle_exclusion_frac,) if frac is not None)

    raw_support = build_height_support_mask(
        height_for_rgb,
        valid_for_rgb,
        min_piece_height_mm=min_piece_height_mm,
        max_piece_height_mm=max_piece_height_mm,
        roi_frac=roi_frac,
        exclusion_fracs=exclusion_fracs,
        noise_floor_mm=noise_floor,
        noise_margin_mm=noise_margin_mm,
        min_area_px=max(3, min_support_pixels // 2),
    )
    stable_support = temporal_filter.update(raw_support) if temporal_filter is not None else raw_support

    candidates, restored_circle_mask, rgb_mask, proposed_count = generate_restored_circle_candidates(
        rgb_bgr,
        stable_support,
        valid_for_rgb.astype(np.uint8),
        min_radius_px=min_radius_px,
        max_radius_px=max_radius_px,
        expected_radius_px=expected_radius_px,
        min_candidate_support_ratio=min_candidate_support_ratio,
        min_support_pixels=min_support_pixels,
        use_hough=use_hough,
        split_touching=split_touching,
        roi_frac=roi_frac,
        exclusion_fracs=exclusion_fracs,
    )

    stat_getters = {
        "median": lambda v: float(np.median(v)),
        "p75": lambda v: float(np.percentile(v, 75)),
        "p90": lambda v: float(np.percentile(v, 90)),
        "max": lambda v: float(np.max(v)),
        "top35": lambda v: _top_band_stat(v, 0.35),
        "top25": lambda v: _top_band_stat(v, 0.25),
    }

    class_bgr = np.zeros((*rgb_bgr.shape[:2], 3), dtype=np.uint8)
    overlay = rgb_bgr.copy()
    classified: List[ChipCandidate] = []
    rejected = 0
    vals_min = max(1.0, min_piece_height_mm * 0.60)

    # First classify each candidate for this frame without drawing. Then apply
    # stack-count smoothing across frames. Drawing after smoothing ensures the
    # mask and overlay show the stable class, while labels still expose raw
    # values for debugging.
    for cand in candidates:
        vals, depth_ratio = _height_values_in_candidate(
            height_for_rgb,
            valid_for_rgb,
            cand,
            min_depth_pixels=min_depth_pixels,
            min_depth_ratio=min_depth_ratio,
        )
        cand.valid_depth_ratio = max(cand.valid_depth_ratio, depth_ratio)
        vals = vals[(vals >= vals_min) & (vals <= max_piece_height_mm)]

        if vals.size >= min_depth_pixels and depth_ratio >= min_depth_ratio:
            cand.height_median_mm = float(np.median(vals))
            cand.height_p75_mm = float(np.percentile(vals, 75))
            cand.height_p90_mm = float(np.percentile(vals, 90))
            cand.height_max_mm = float(np.max(vals))
            cand.height_top35_mm = _top_band_stat(vals, 0.35)
            cand.height_top25_mm = _top_band_stat(vals, 0.25)
            requested_height = stat_getters[height_stat](vals)
            cand.stack_count, cand.height_used_mm = _estimate_stack_1_or_2(
                vals,
                min_piece_height_mm=min_piece_height_mm,
                max_piece_height_mm=max_piece_height_mm,
                promote_height_mm=stack_promote_height_mm,
                demote_height_mm=stack_demote_height_mm,
                requested_height=requested_height,
            )
            cand.raw_stack_count = cand.stack_count
            cand.confidence = min(
                1.0,
                0.45 * min(1.0, depth_ratio / 0.12)
                + 0.35 * min(1.0, cand.support_pixels / max(float(min_support_pixels * 4), 1.0))
                + 0.20 * min(1.0, cand.support_ratio / max(min_candidate_support_ratio, 1e-6)),
            )
        else:
            cand.raw_stack_count = 0
            cand.stack_count = 0
            cand.confidence = 0.0

        if cand.stack_count <= 0:
            rejected += 1
        else:
            classified.append(cand)

    if stack_smoother is not None:
        classified = stack_smoother.update(classified)

    for cand in classified:
        cx, cy, r = int(round(cand.x)), int(round(cand.y)), max(2, int(round(cand.radius)))
        if cand.stack_count == 1:
            colour = (0, 220, 0)
            label = "1"
        else:
            colour = (0, 220, 255)
            label = "2"

        cv2.circle(class_bgr, (cx, cy), r, colour, -1)
        cv2.circle(overlay, (cx, cy), r, colour, 2)
        cv2.putText(overlay, label, (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2, cv2.LINE_AA)
        raw_note = f"r{cand.raw_stack_count}" if cand.raw_stack_count and cand.raw_stack_count != cand.stack_count else ""
        track_note = f" T{cand.track_id}" if cand.track_id >= 0 else ""
        cv2.putText(
            overlay,
            f"{cand.height_used_mm:.1f}mm {raw_note}{track_note}",
            (cx - 34, cy + r + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            colour,
            1,
            cv2.LINE_AA,
        )

    if draw_rejected_candidates:
        for cand in candidates:
            if cand.stack_count > 0:
                continue
            cx, cy, r = int(round(cand.x)), int(round(cand.y)), max(2, int(round(cand.radius)))
            cv2.circle(overlay, (cx, cy), r, (80, 80, 80), 1)
            cv2.putText(overlay, "?", (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    filtered_circle_mask = np.zeros_like(restored_circle_mask)
    for cand in classified:
        cv2.circle(filtered_circle_mask, (int(round(cand.x)), int(round(cand.y))), max(2, int(round(cand.radius))), 255, -1)

    diagnostics = {
        "candidate_count": float(len(candidates)),
        "proposed_count": float(proposed_count),
        "classified_count": float(len(classified)),
        "rejected_count": float(rejected),
        "valid_overlap_ratio": float(np.count_nonzero(valid_for_rgb)) / float(valid_for_rgb.size),
        "raw_support_ratio": float(np.count_nonzero(raw_support)) / float(raw_support.size),
        "stable_support_ratio": float(np.count_nonzero(stable_support)) / float(stable_support.size),
        "rgb_restored_ratio": float(np.count_nonzero(filtered_circle_mask)) / float(filtered_circle_mask.size),
        "rgb_mask_ratio": float(np.count_nonzero(rgb_mask)) / float(rgb_mask.size),
        "hough_enabled": float(bool(use_hough)),
        "split_touching_enabled": float(bool(split_touching)),
        "min_candidate_support_ratio": float(min_candidate_support_ratio),
        "temporal_history": float(temporal_filter.history_size if temporal_filter is not None else 0),
        "stack_promote_height_mm": float(stack_promote_height_mm),
        "stack_demote_height_mm": float(stack_demote_height_mm),
    }

    return StackClassificationResult(
        candidates=classified,
        height_mm=height_for_rgb,
        valid_overlap_mask=valid_for_rgb.astype(np.uint8),
        stable_depth_support_mask=stable_support,
        rgb_restored_circle_mask=filtered_circle_mask,
        rgb_candidate_mask=rgb_mask,
        piece_presence_mask=stable_support,
        stack_class_bgr=class_bgr,
        overlay_bgr=overlay,
        diagnostics=diagnostics,
    )
