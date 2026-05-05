from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

from backgammon_types import BoardState, PipelineResult
from board_registration import BoardRegistrar
from perspective_rectification import PerspectiveRectifier
from image_normalisation import BoardNormaliser
from point_tray_segmentation import PointTraySegmenter
from piece_detection import RGBDPieceDetector
from dice_cube_reader import DiceCubeReader
from temporal_state_estimator import TemporalStateEstimator
from rule_validation import BoardStateValidator
from event_inference import BoardStateEventInferer


@dataclass
class PipelineConfig:
    commit_threshold: float = 0.55


class BoardStatePipeline:
    """
    Runs the board-state pipeline from frame/keyframe input through to a pipeline result.

    The caller should pass either a raw RGB-D frame or an accepted KeyframePacket.
    This class returns a PipelineResult containing both intermediate artifacts and
    the final temporal/validation decision for that frame.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.registrar = BoardRegistrar()
        self.rectifier = PerspectiveRectifier()
        self.normaliser = BoardNormaliser()
        self.segmenter = PointTraySegmenter()
        self.piece_detector = RGBDPieceDetector()
        self.dice_cube_reader = DiceCubeReader()
        self.temporal = TemporalStateEstimator(window_size=5)
        self.validator = BoardStateValidator(require_exact_totals=False)
        self.event_inferer = BoardStateEventInferer()

    @staticmethod
    def _candidate_state_from_results(piece_result, dice_cube_result) -> BoardState:
        dice_values = [obs.value for obs in dice_cube_result.dice][:2]
        while len(dice_values) < 2:
            dice_values.append(None)
        cube_value = dice_cube_result.cube.value if dice_cube_result.cube is not None else None
        confidence_parts = [piece_result.confidence, dice_cube_result.confidence]
        confidence = sum(confidence_parts) / max(len(confidence_parts), 1)
        return BoardState(
            region_counts=piece_result.region_counts,
            dice=(dice_values[0], dice_values[1]),
            cube_value=cube_value,
            confidence=confidence,
            timestamp=time.time(),
        )

    def process_frame(self, rgb_bgr, depth_mm) -> PipelineResult:
        board_lock = self.registrar.update(rgb_bgr)
        if not board_lock.valid:
            return PipelineResult(
                board_lock=board_lock,
                rectified=None,
                normalised=None,
                regions=None,
                pieces=None,
                dice_cube=None,
                temporal=None,
                validation=None,
                committed=False,
                state_changed=False,
                debug={"stage": "board_registration"},
            )

        rectified = self.rectifier.rectify(rgb_bgr, depth_mm, board_lock)
        normalised = self.normaliser.normalize(rectified)
        regions = self.segmenter.segment(normalised.rgb_bgr.shape[:2])

        # Geometry-constrained interpretation. Piece detection uses RegionMasks
        # to avoid dice/cube/bear-off areas. Dice/cube reading also uses manual
        # dice_area/cube_area masks when they are present.
        pieces = self.piece_detector.detect(normalised, regions)
        dice_cube = self.dice_cube_reader.read(normalised, regions)

        candidate_state = self._candidate_state_from_results(pieces, dice_cube)
        temporal_estimate = self.temporal.update(candidate_state)
        validation = self.validator.validate(temporal_estimate.fused_state)

        committed = bool(validation.valid and temporal_estimate.fused_state.confidence >= self.config.commit_threshold)
        events = []
        if committed:
            events = self.event_inferer.update(temporal_estimate.fused_state)
            self.temporal.commit(temporal_estimate.fused_state)

        return PipelineResult(
            board_lock=board_lock,
            rectified=rectified,
            normalised=normalised,
            regions=regions,
            pieces=pieces,
            dice_cube=dice_cube,
            temporal=temporal_estimate,
            validation=validation,
            committed=committed,
            state_changed=temporal_estimate.changed_vs_last_commit,
            debug={"stage": "complete"},
            events=events,
        )

    def process_keyframe_packet(self, packet) -> PipelineResult:
        return self.process_frame(packet.left_raw, packet.depth_raw)


__all__ = ["PipelineConfig", "BoardStatePipeline"]
