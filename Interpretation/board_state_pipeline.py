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


@dataclass
class PipelineConfig:
    commit_threshold: float = 0.55


class BoardStatePipeline:
    """
    End-to-end orchestration for the flowchart in your dissertation notebook.

    Input:
      RGB-D frame (typically a keyframe packet's left_norm/left_raw + depth_raw)

    Output:
      PipelineResult containing the board lock, rectified/normalised images,
      segmentation masks, checker detections, dice/cube observation, temporal
      fusion result, validation report, and commit flag.
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

    @staticmethod
    def _candidate_state_from_results(piece_result, dice_cube_result) -> BoardState:
        dice_values = [obs.value for obs in dice_cube_result.dice][:2]
        while len(dice_values) < 2:
            dice_values.append(None)

        cube_value = dice_cube_result.cube.value if dice_cube_result.cube is not None else None

        confidence_parts = [
            piece_result.confidence,
            dice_cube_result.confidence,
        ]
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
        pieces = self.piece_detector.detect(normalised, regions)
        dice_cube = self.dice_cube_reader.read(normalised)

        candidate_state = self._candidate_state_from_results(pieces, dice_cube)
        temporal_estimate = self.temporal.update(candidate_state)
        validation = self.validator.validate(temporal_estimate.fused_state)

        committed = bool(validation.valid and temporal_estimate.fused_state.confidence >= self.config.commit_threshold)
        if committed:
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
        )

    def process_keyframe_packet(self, packet) -> PipelineResult:
        return self.process_frame(packet.left_raw, packet.depth_raw)
