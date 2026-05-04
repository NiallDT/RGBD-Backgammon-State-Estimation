from __future__ import annotations

from typing import Optional

from backgammon_types import BoardState, ValidationReport


class BoardStateValidator:
    """
    Rule-based validation pass for the fused board state.
    """

    def __init__(self, expected_checkers_per_colour: int = 15, require_exact_totals: bool = False) -> None:
        self.expected_checkers_per_colour = expected_checkers_per_colour
        self.require_exact_totals = require_exact_totals

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def validate(self, state: BoardState) -> ValidationReport:
        errors = []
        warnings = []

        total_light = 0
        total_dark = 0

        for region_name, counts in state.region_counts.items():
            light = int(counts.get("light", 0))
            dark = int(counts.get("dark", 0))

            if light < 0 or dark < 0:
                errors.append(f"Negative checker count in {region_name}.")

            if light > 0 and dark > 0:
                errors.append(f"Both colours occupy {region_name}, which is not a legal settled point state.")

            total_light += light
            total_dark += dark

        if self.require_exact_totals:
            if total_light != self.expected_checkers_per_colour:
                errors.append(f"Light total is {total_light}, expected {self.expected_checkers_per_colour}.")
            if total_dark != self.expected_checkers_per_colour:
                errors.append(f"Dark total is {total_dark}, expected {self.expected_checkers_per_colour}.")
        else:
            if total_light > self.expected_checkers_per_colour:
                errors.append(f"Light total exceeds {self.expected_checkers_per_colour}.")
            if total_dark > self.expected_checkers_per_colour:
                errors.append(f"Dark total exceeds {self.expected_checkers_per_colour}.")
            if total_light < self.expected_checkers_per_colour:
                warnings.append(f"Light total below {self.expected_checkers_per_colour}; detection may be incomplete.")
            if total_dark < self.expected_checkers_per_colour:
                warnings.append(f"Dark total below {self.expected_checkers_per_colour}; detection may be incomplete.")

        for die_value in state.dice:
            if die_value is not None and not (1 <= die_value <= 6):
                errors.append(f"Invalid die value {die_value}.")

        if state.cube_value is not None and not self._is_power_of_two(state.cube_value):
            errors.append(f"Cube value {state.cube_value} is not a power of two.")

        confidence = max(0.0, 1.0 - 0.15 * len(errors) - 0.05 * len(warnings))
        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence,
        )
