"""Configurable success criteria for G1 box placement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoxPlacementSuccessCriteria:
    """Episode success thresholds for box-on-platform placement."""

    horizontal_tolerance_m: float = 0.1
    height_tolerance_m: float = 0.05
    max_tilt_deg: float = 15.0
    stable_steps: int = 10

    def frame_satisfied(
        self,
        *,
        horizontal_error: np.ndarray,
        height_error: np.ndarray,
        tilt_deg: np.ndarray,
    ) -> np.ndarray:
        """Return per-env booleans for a single simulation frame."""
        return np.logical_and.reduce(
            [
                horizontal_error <= self.horizontal_tolerance_m,
                height_error <= self.height_tolerance_m,
                tilt_deg <= self.max_tilt_deg,
            ]
        )

    def update_stable_counter(
        self,
        stable_counter: np.ndarray,
        frame_ok: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance stable-step counters and return newly succeeded env ids."""
        stable_counter = np.where(frame_ok, stable_counter + 1, 0).astype(np.int32)
        succeeded = stable_counter >= int(self.stable_steps)
        return stable_counter, succeeded
