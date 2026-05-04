from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

from backgammon_types import BoardState


@dataclass
class BoardStateEvaluation:
    total_regions: int
    exact_region_matches: int
    total_colour_slots: int
    correct_colour_counts: int
    absolute_checker_error: int
    dice_match: Optional[bool]
    cube_match: Optional[bool]
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def region_accuracy(self) -> float:
        return self.exact_region_matches / max(self.total_regions, 1)

    @property
    def colour_count_accuracy(self) -> float:
        return self.correct_colour_counts / max(self.total_colour_slots, 1)


class BoardStateEvaluator:
    """
    Simple evaluation helper for known-position tests.

    Expected JSON format:
    {
      "region_counts": {"point_01": {"light": 2, "dark": 0}, ...},
      "dice": [1, 2],
      "cube_value": 2
    }
    """

    @staticmethod
    def load_expected(path: str | Path) -> Dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _counts(obj: Dict[str, Any], region: str, colour: str) -> int:
        return int(obj.get(region, {}).get(colour, 0))

    def evaluate(self, expected: Dict[str, Any], observed: BoardState) -> BoardStateEvaluation:
        expected_counts = expected.get("region_counts", {})
        observed_counts = observed.region_counts
        regions = sorted(set(expected_counts) | set(observed_counts))
        exact_region_matches = 0
        correct_colour_counts = 0
        total_colour_slots = 0
        absolute_checker_error = 0
        region_details = {}

        for region in regions:
            exp_l = self._counts(expected_counts, region, "light")
            exp_d = self._counts(expected_counts, region, "dark")
            obs_l = self._counts(observed_counts, region, "light")
            obs_d = self._counts(observed_counts, region, "dark")
            exact = (exp_l == obs_l and exp_d == obs_d)
            exact_region_matches += int(exact)
            correct_colour_counts += int(exp_l == obs_l) + int(exp_d == obs_d)
            total_colour_slots += 2
            absolute_checker_error += abs(exp_l - obs_l) + abs(exp_d - obs_d)
            if not exact:
                region_details[region] = {
                    "expected": {"light": exp_l, "dark": exp_d},
                    "observed": {"light": obs_l, "dark": obs_d},
                }

        expected_dice = expected.get("dice")
        dice_match = None if expected_dice is None else tuple(expected_dice) == tuple(observed.dice)
        expected_cube = expected.get("cube_value")
        cube_match = None if expected_cube is None else expected_cube == observed.cube_value

        return BoardStateEvaluation(
            total_regions=len(regions),
            exact_region_matches=exact_region_matches,
            total_colour_slots=total_colour_slots,
            correct_colour_counts=correct_colour_counts,
            absolute_checker_error=absolute_checker_error,
            dice_match=dice_match,
            cube_match=cube_match,
            details={"region_mismatches": region_details},
        )


__all__ = ["BoardStateEvaluation", "BoardStateEvaluator"]
