from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


ColourName = str
RegionName = str


@dataclass
class BoardLock:
    valid: bool
    corners_xy: np.ndarray
    confidence: float
    method: str
    frame_size_hw: Tuple[int, int]
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RectifiedBoard:
    rgb_bgr: np.ndarray
    depth_mm: np.ndarray
    valid_mask: np.ndarray
    height_map_mm: np.ndarray
    homography: np.ndarray
    inverse_homography: np.ndarray
    board_mask: np.ndarray
    plane_coeffs: Tuple[float, float, float]


@dataclass
class NormalisedBoard:
    rgb_bgr: np.ndarray
    rgb_gray: np.ndarray
    depth_mm: np.ndarray
    depth_gray: np.ndarray
    valid_mask: np.ndarray
    height_map_mm: np.ndarray
    height_uint8: np.ndarray


@dataclass
class RegionMasks:
    masks: Dict[RegionName, np.ndarray]
    overlay_bgr: np.ndarray
    point_names: List[RegionName]
    auxiliary_names: List[RegionName]


@dataclass
class PieceInstance:
    region_name: RegionName
    colour_name: ColourName
    centroid_xy: Tuple[float, float]
    area_px: float
    radius_px: float
    height_mm: float
    stack_count: int
    confidence: float
    contour: Optional[np.ndarray] = None


@dataclass
class PieceDetectionResult:
    pieces: List[PieceInstance]
    region_counts: Dict[RegionName, Dict[ColourName, int]]
    overlay_bgr: np.ndarray
    confidence: float
    # Diagnostic masks used while tuning the detector. The main pipeline can
    # ignore these once the counts and overlay have been produced.
    stable_depth_support_mask: Optional[np.ndarray] = None
    slot_candidate_mask: Optional[np.ndarray] = None
    checker_detection_area_mask: Optional[np.ndarray] = None
    stack_class_bgr: Optional[np.ndarray] = None
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiceObservation:
    bbox_xyxy: Tuple[int, int, int, int]
    value: Optional[int]
    confidence: float


@dataclass
class CubeObservation:
    bbox_xyxy: Tuple[int, int, int, int]
    value: Optional[int]
    confidence: float
    text: Optional[str] = None


@dataclass
class DiceCubeObservation:
    dice: List[DiceObservation]
    cube: Optional[CubeObservation]
    overlay_bgr: np.ndarray
    confidence: float


@dataclass
class BoardState:
    region_counts: Dict[RegionName, Dict[ColourName, int]]
    dice: Tuple[Optional[int], Optional[int]] = (None, None)
    cube_value: Optional[int] = None
    confidence: float = 0.0
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str]
    warnings: List[str]
    confidence: float


@dataclass
class TemporalEstimate:
    candidate_state: BoardState
    fused_state: BoardState
    changed_vs_last_commit: bool
    history_size: int


@dataclass
class StateEvent:
    event_type: str
    description: str
    timestamp: Optional[float] = None
    confidence: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    board_lock: BoardLock
    rectified: Optional[RectifiedBoard]
    normalised: Optional[NormalisedBoard]
    regions: Optional[RegionMasks]
    pieces: Optional[PieceDetectionResult]
    dice_cube: Optional[DiceCubeObservation]
    temporal: Optional[TemporalEstimate]
    validation: Optional[ValidationReport]
    committed: bool
    state_changed: bool
    debug: Dict[str, Any] = field(default_factory=dict)
    events: List[StateEvent] = field(default_factory=list)


def make_empty_region_counts(region_names: List[RegionName]) -> Dict[RegionName, Dict[ColourName, int]]:
    return {name: {"light": 0, "dark": 0} for name in region_names}


__all__ = [
    "BoardLock",
    "RectifiedBoard",
    "NormalisedBoard",
    "RegionMasks",
    "PieceInstance",
    "PieceDetectionResult",
    "DiceObservation",
    "CubeObservation",
    "DiceCubeObservation",
    "BoardState",
    "ValidationReport",
    "TemporalEstimate",
    "StateEvent",
    "PipelineResult",
    "make_empty_region_counts",
]
