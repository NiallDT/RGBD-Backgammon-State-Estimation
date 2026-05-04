from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

HeightStat = Literal["median", "p75", "p90", "max", "top35", "top25"]
RoiFrac = Tuple[float, float, float, float]


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
    stack_count: int = 0
    confidence: float = 0.0
    raw_stack_count: int = 0
    smoothed_stack_count: int = 0
    track_id: int = -1
    border_distance_px: float = 9999.0
    near_edge: bool = False
    near_corner: bool = False
    held_from_track: bool = False


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


def _roi_mask(shape_hw: Tuple[int, int], roi_frac: Optional[RoiFrac]) -> np.ndarray:
    h, w = shape_hw
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if roi_frac is None:
        return mask
    x1f, y1f, x2f, y2f = roi_frac
    x1 = int(np.clip(x1f, 0.0, 1.0) * w)
    y1 = int(np.clip(y1f, 0.0, 1.0) * h)
    x2 = int(np.clip(x2f, 0.0, 1.0) * w)
    y2 = int(np.clip(y2f, 0.0, 1.0) * h)
    mask[:] = 0
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask




def annotate_candidate_edge_context(
    cand: ChipCandidate,
    shape_hw: Tuple[int, int],
    roi_frac: Optional[RoiFrac],
    *,
    edge_margin_px: float = 42.0,
) -> ChipCandidate:
    """Mark candidates close to image/ROI edges where depth support is weaker."""
    h, w = shape_hw
    if roi_frac is None:
        x1, y1, x2, y2 = 0.0, 0.0, float(w - 1), float(h - 1)
    else:
        x1f, y1f, x2f, y2f = roi_frac
        x1, y1, x2, y2 = x1f * w, y1f * h, x2f * w, y2f * h
    # Distance from centre to the usable ROI/image boundary. Radius is included
    # so a full footprint touching the boundary is treated as edge/corner even
    # when the centre is slightly inside the board.
    left = cand.x - x1
    right = x2 - cand.x
    top = cand.y - y1
    bottom = y2 - cand.y
    border_dist = float(min(left, right, top, bottom))
    threshold = float(edge_margin_px) + 0.35 * float(cand.radius)
    near_x = min(left, right) <= threshold
    near_y = min(top, bottom) <= threshold
    cand.border_distance_px = border_dist
    cand.near_edge = bool(near_x or near_y)
    cand.near_corner = bool(near_x and near_y)
    return cand
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



class TemporalStackSmoother:
    """
    Track chip candidates over a few frames and apply stack-count hysteresis.

    The depth support near board edges/corners can intermittently under-read a
    two-high stack as one-high. This class deliberately keeps detection permissive
    and smooths only the interpreted stack count. Promotion from 1->2 is fairly
    quick, while demotion from 2->1 requires more evidence so edge stacks do not
    flicker.
    """

    def __init__(
        self,
        *,
        window: int = 5,
        promote_votes: int = 2,
        demote_votes: int = 4,
        max_match_distance_px: float = 24.0,
        max_misses: int = 6,
        hold_misses: int = 2,
        edge_margin_px: float = 42.0,
        edge_promote_height_factor: float = 1.30,
        edge_demote_height_factor: float = 1.18,
    ) -> None:
        self.window = max(1, int(window))
        self.promote_votes = max(1, int(promote_votes))
        self.demote_votes = max(1, int(demote_votes))
        self.max_match_distance_px = float(max_match_distance_px)
        self.max_misses = max(1, int(max_misses))
        self.hold_misses = max(0, int(hold_misses))
        self.edge_margin_px = float(edge_margin_px)
        self.edge_promote_height_factor = float(edge_promote_height_factor)
        self.edge_demote_height_factor = float(edge_demote_height_factor)
        self._tracks: List[Dict[str, object]] = []
        self._next_id = 1

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @staticmethod
    def _mode_positive(values: List[int], fallback: int = 1) -> int:
        vals = [int(v) for v in values if int(v) > 0]
        if not vals:
            return fallback
        counts: Dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        # Tie-break toward the larger stack because under-reading is the common
        # failure at edges, while promotion is separately gated by votes.
        return max(counts.keys(), key=lambda k: (counts[k], k))

    def _smooth_count(self, track: Dict[str, object], raw_count: int, height_mm: float) -> int:
        hist: Deque[int] = track["stack_history"]  # type: ignore[assignment]
        heights: Deque[float] = track["height_history"]  # type: ignore[assignment]
        current = int(track.get("stable_stack", raw_count if raw_count > 0 else 1))
        values = list(hist)
        height_values = [float(h) for h in heights if float(h) > 0]
        t = max(float(track.get("chip_thickness_mm", 10.0)), 1.0)

        high_votes = sum(1 for v in values if v >= 2)
        low_votes = sum(1 for v in values if v <= 1)
        near_edge = bool(track.get("near_edge", False))
        near_corner = bool(track.get("near_corner", False))

        recent_heights = height_values[-min(4, len(height_values)):] if height_values else []
        recent_height = float(np.median(recent_heights)) if recent_heights else float(height_mm)
        recent_high = float(max(recent_heights)) if recent_heights else float(height_mm)

        # Edge/corner stacks often have sparse depth support, so raw count may
        # under-read as 1 even when the upper height evidence is still close to
        # two chip thicknesses. Use height votes as additional promotion evidence
        # near borders, but keep the normal raw-vote rule elsewhere.
        edge_promote_floor = self.edge_promote_height_factor * t
        edge_demote_floor = self.edge_demote_height_factor * t
        height_high_votes = sum(1 for h in height_values if h >= edge_promote_floor)

        if current <= 1:
            if high_votes >= self.promote_votes:
                return max(2, self._mode_positive([v for v in values if v >= 2], fallback=2))
            if (near_edge or near_corner) and height_high_votes >= max(1, min(self.promote_votes, 2)):
                return 2
            return 1 if raw_count > 0 else 0

        # Once a stack is believed to be >=2, avoid demoting because of a single
        # weak edge/corner frame. Corners are the worst case: use both extra
        # low-vote evidence and consistently low recent height before demoting.
        if near_edge or near_corner:
            required_low = max(self.demote_votes + (2 if near_corner else 1), self.demote_votes)
            if low_votes >= required_low and recent_high < edge_demote_floor:
                return 1
            # If any recent frame still has plausible two-chip height evidence,
            # keep the existing stack class. This is specifically for bottom-left
            # / corner flicker where the current frame under-reads.
            if recent_high >= edge_demote_floor or high_votes > 0:
                return max(2, current)
            return current

        two_chip_floor = 1.45 * t
        if low_votes >= self.demote_votes and recent_height < two_chip_floor:
            return 1

        high_mode = self._mode_positive([v for v in values if v >= 2], fallback=current)
        return max(current, high_mode)

    def _should_hold_track(self, track: Dict[str, object]) -> bool:
        misses = int(track.get("misses", 0))
        stable = int(track.get("stable_stack", 0))
        near = bool(track.get("near_edge", False)) or bool(track.get("near_corner", False))
        return self.hold_misses > 0 and near and stable >= 2 and 0 < misses <= self.hold_misses

    def _candidate_from_track(self, track: Dict[str, object]) -> ChipCandidate:
        cand = ChipCandidate(
            x=float(track.get("x", 0.0)),
            y=float(track.get("y", 0.0)),
            radius=float(track.get("radius", 16.0)),
            area_px=float(np.pi * float(track.get("radius", 16.0)) ** 2),
            circularity=1.0,
            source="track-hold",
        )
        cand.stack_count = int(track.get("stable_stack", 2))
        cand.smoothed_stack_count = cand.stack_count
        cand.raw_stack_count = 0
        cand.height_used_mm = float(track.get("last_height", 0.0))
        cand.track_id = int(track.get("id", -1))
        cand.near_edge = bool(track.get("near_edge", False))
        cand.near_corner = bool(track.get("near_corner", False))
        cand.border_distance_px = float(track.get("border_distance_px", 9999.0))
        cand.held_from_track = True
        cand.confidence = 0.25
        return cand

    def update(self, candidates: List[ChipCandidate], *, chip_thickness_mm: float = 10.0) -> List[ChipCandidate]:
        if not candidates:
            held: List[ChipCandidate] = []
            for tr in self._tracks:
                tr["misses"] = int(tr.get("misses", 0)) + 1
                if self._should_hold_track(tr):
                    held.append(self._candidate_from_track(tr))
            self._tracks = [tr for tr in self._tracks if int(tr.get("misses", 0)) <= self.max_misses]
            return held

        unmatched = set(range(len(self._tracks)))
        for cand in candidates:
            best_i: Optional[int] = None
            best_d = float("inf")
            for i in list(unmatched):
                tr = self._tracks[i]
                d = float(np.hypot(cand.x - float(tr["x"]), cand.y - float(tr["y"])))
                allowed = max(self.max_match_distance_px, 1.55 * max(cand.radius, float(tr.get("radius", cand.radius))))
                if d <= allowed and d < best_d:
                    best_d = d
                    best_i = i
            if best_i is None:
                tr = {
                    "id": self._next_id,
                    "x": float(cand.x),
                    "y": float(cand.y),
                    "radius": float(cand.radius),
                    "stack_history": deque(maxlen=self.window),
                    "height_history": deque(maxlen=self.window),
                    "stable_stack": int(cand.stack_count),
                    "misses": 0,
                    "chip_thickness_mm": float(chip_thickness_mm),
                    "near_edge": bool(cand.near_edge),
                    "near_corner": bool(cand.near_corner),
                    "border_distance_px": float(cand.border_distance_px),
                    "last_height": float(cand.height_used_mm),
                }
                self._next_id += 1
                self._tracks.append(tr)
            else:
                tr = self._tracks[best_i]
                unmatched.discard(best_i)
                tr["x"] = 0.70 * float(tr["x"]) + 0.30 * float(cand.x)
                tr["y"] = 0.70 * float(tr["y"]) + 0.30 * float(cand.y)
                tr["radius"] = 0.75 * float(tr.get("radius", cand.radius)) + 0.25 * float(cand.radius)
                tr["misses"] = 0
                tr["chip_thickness_mm"] = float(chip_thickness_mm)
                tr["near_edge"] = bool(tr.get("near_edge", False)) or bool(cand.near_edge)
                tr["near_corner"] = bool(tr.get("near_corner", False)) or bool(cand.near_corner)
                tr["border_distance_px"] = min(float(tr.get("border_distance_px", cand.border_distance_px)), float(cand.border_distance_px))
                tr["last_height"] = float(cand.height_used_mm)

            raw = int(cand.stack_count)
            cand.raw_stack_count = raw
            hist: Deque[int] = tr["stack_history"]  # type: ignore[assignment]
            heights: Deque[float] = tr["height_history"]  # type: ignore[assignment]
            hist.append(raw)
            heights.append(float(cand.height_used_mm))
            smooth = self._smooth_count(tr, raw, float(cand.height_used_mm))
            tr["stable_stack"] = int(smooth)
            cand.stack_count = int(smooth)
            cand.smoothed_stack_count = int(smooth)
            cand.track_id = int(tr["id"])

        held: List[ChipCandidate] = []
        for i in unmatched:
            self._tracks[i]["misses"] = int(self._tracks[i].get("misses", 0)) + 1
            if self._should_hold_track(self._tracks[i]):
                held.append(self._candidate_from_track(self._tracks[i]))
        self._tracks = [tr for tr in self._tracks if int(tr.get("misses", 0)) <= self.max_misses]
        if held:
            candidates.extend(held)
        return candidates

def build_height_support_mask(
    height_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_piece_height_mm: float,
    max_piece_height_mm: float,
    roi_frac: Optional[RoiFrac],
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
    raw = cv2.bitwise_and(raw, _roi_mask(raw.shape[:2], roi_frac))
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


def _build_rgb_chip_mask(rgb_bgr: np.ndarray, roi_frac: Optional[RoiFrac]) -> np.ndarray:
    """Bright low-saturation mask for the current pale/white diagnostic chips."""
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # White/off-white chips: bright, relatively unsaturated, roughly neutral in Lab.
    mask = ((v >= 125) & (s <= 95) & (l >= 120) & (np.abs(a.astype(np.int16) - 128) <= 28) & (np.abs(b.astype(np.int16) - 128) <= 38)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, _roi_mask(mask.shape[:2], roi_frac))
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
    expected_radius_px: float,
    min_radius_px: float,
    max_radius_px: float,
    split_touching: bool,
) -> Tuple[List[ChipCandidate], np.ndarray]:
    rgb_mask = _build_rgb_chip_mask(rgb_bgr, roi_frac)
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
) -> Tuple[List[ChipCandidate], np.ndarray, np.ndarray, int]:
    if expected_radius_px is None or expected_radius_px <= 0:
        expected_radius_px = (float(min_radius_px) + float(max_radius_px)) * 0.5

    proposals: List[ChipCandidate] = []
    rgb_props, rgb_mask = _rgb_circle_candidates(
        rgb_bgr,
        stable_support,
        roi_frac=roi_frac,
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
    """Classify stack height from a height distribution with edge-stack tolerance.

    Passive stereo often under-fills the top face of a stack near board edges. A
    simple median can therefore read a two-high stack as one-high. This function
    looks for band evidence near multiples of chip thickness and uses high
    quantiles/top-band statistics as supporting evidence, while avoiding promotion
    from a few isolated spikes.
    """
    t = max(float(chip_thickness_mm), 1.0)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= max(1.0, min_piece_height_mm * 0.65)) & (vals <= max_piece_height_mm)]
    if vals.size == 0:
        return 0, 0.0

    n = int(vals.size)
    p75 = float(np.percentile(vals, 75))
    p85 = float(np.percentile(vals, 85))
    p90 = float(np.percentile(vals, 90))
    top35 = _top_band_stat(vals, 0.35)
    top25 = _top_band_stat(vals, 0.25)

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

    # Strong normal case: enough samples in a thickness band.
    best_idx = int(np.argmax(band_counts))
    best_count = band_counts[best_idx]
    best_stack = best_idx + 1
    if best_count >= max(4, int(0.22 * n)):
        # If the one-chip band barely wins but the upper quantiles/top-band are
        # clearly around two chips, keep the two-stack hypothesis alive.
        if best_stack == 1 and max_stack >= 2:
            two_count = band_counts[1]
            two_like_height = max(float(requested_height), p85, top35)
            if two_like_height >= 1.48 * t and two_count >= max(2, int(0.045 * n)):
                return 2, max(two_like_height, band_medians[1])
        return best_stack, band_medians[best_idx]

    # Sparse/edge case: accept a two-stack when several height statistics agree,
    # rather than requiring a dense top face.
    if max_stack >= 2:
        two_band = vals[(vals >= 1.35 * t) & (vals <= 2.65 * t)]
        two_count = int(two_band.size)
        two_like_height = max(float(requested_height), p85, p90, top35, top25)
        enough_two_pixels = two_count >= max(2, int(0.04 * n))
        not_just_one_spike = p75 >= 1.18 * t or two_count >= max(3, int(0.08 * n))
        if two_like_height >= 1.48 * t and enough_two_pixels and not_just_one_spike:
            return 2, float(np.median(two_band)) if two_band.size else two_like_height

    # Conservative fallback from requested robust statistic.
    est = max(0, classify_stack_count(requested_height, t, min_piece_height_mm))
    if est > 1:
        idx = min(est, len(band_counts)) - 1
        # Do not promote to a taller stack purely from one or two isolated spikes.
        if band_counts[idx] < max(2, int(0.04 * n)):
            est = 1
            requested_height = band_medians[0] if band_counts[0] else float(np.median(vals))
    return est, float(requested_height)


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
    temporal_filter: Optional[TemporalMaskFilter] = None,
    stack_smoother: Optional[TemporalStackSmoother] = None,
    noise_margin_mm: float = 1.5,
    edge_margin_px: float = 42.0,
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

    raw_support = build_height_support_mask(
        height_for_rgb,
        valid_for_rgb,
        min_piece_height_mm=min_piece_height_mm,
        max_piece_height_mm=max_piece_height_mm,
        roi_frac=roi_frac,
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
    )

    for cand in candidates:
        annotate_candidate_edge_context(cand, height_for_rgb.shape[:2], roi_frac, edge_margin_px=edge_margin_px)

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
    prelim_classified: List[ChipCandidate] = []
    rejected = 0
    vals_min = max(1.0, min_piece_height_mm * 0.60)

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
            cand.stack_count, cand.height_used_mm = _estimate_stack_from_distribution(vals, chip_thickness_mm, min_piece_height_mm, max_piece_height_mm, requested_height)
            cand.raw_stack_count = int(cand.stack_count)
            cand.confidence = min(
                1.0,
                0.45 * min(1.0, depth_ratio / 0.12)
                + 0.35 * min(1.0, cand.support_pixels / max(float(min_support_pixels * 4), 1.0))
                + 0.20 * min(1.0, cand.support_ratio / max(min_candidate_support_ratio, 1e-6)),
            )
        else:
            cand.stack_count = 0
            cand.raw_stack_count = 0
            cand.confidence = 0.0

        if cand.stack_count <= 0:
            rejected += 1
            if draw_rejected_candidates:
                cx, cy, r = int(round(cand.x)), int(round(cand.y)), max(2, int(round(cand.radius)))
                cv2.circle(overlay, (cx, cy), r, (80, 80, 80), 1)
                cv2.putText(overlay, "?", (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
            continue

        prelim_classified.append(cand)

    classified = stack_smoother.update(prelim_classified, chip_thickness_mm=chip_thickness_mm) if stack_smoother is not None else prelim_classified

    for cand in classified:
        cx, cy, r = int(round(cand.x)), int(round(cand.y)), max(2, int(round(cand.radius)))
        if cand.stack_count == 1:
            colour = (0, 220, 0)
            label = "1"
        elif cand.stack_count == 2:
            colour = (0, 220, 255)
            label = "2"
        else:
            colour = (0, 0, 255)
            label = str(cand.stack_count)

        cv2.circle(class_bgr, (cx, cy), r, colour, -1)
        cv2.circle(overlay, (cx, cy), r, colour, 2)
        cv2.putText(overlay, label, (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2, cv2.LINE_AA)
        raw_note = f" r{cand.raw_stack_count}" if cand.raw_stack_count and cand.raw_stack_count != cand.stack_count else ""
        edge_note = " C" if cand.near_corner else (" E" if cand.near_edge else "")
        hold_note = " H" if cand.held_from_track else ""
        cv2.putText(overlay, f"{cand.height_used_mm:.1f}mm{raw_note} T{cand.track_id}{edge_note}{hold_note}", (cx - 38, cy + r + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1, cv2.LINE_AA)

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
        "stack_temporal_tracks": float(stack_smoother.track_count if stack_smoother is not None else 0),
        "edge_candidate_count": float(sum(1 for c in classified if c.near_edge)),
        "corner_candidate_count": float(sum(1 for c in classified if c.near_corner)),
        "held_candidate_count": float(sum(1 for c in classified if c.held_from_track)),
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
