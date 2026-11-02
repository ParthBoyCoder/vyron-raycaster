"""Autonomous control interface for Vyron.

The brain owns perception and decision-making. The game loop owns simulation,
collision, rendering, and execution of validated actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Mapping

import numpy as np


class Movement(StrEnum):
    """Supported movement directions."""

    NONE = "none"
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class BrainAction:
    """A bounded action emitted by an autonomous brain."""

    movement: Movement = Movement.NONE
    rotation: float = 0.0
    duration: float = 0.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.rotation <= 1.0:
            raise ValueError("rotation must be between -1.0 and 1.0")
        if not 0.0 <= self.duration <= 0.5:
            raise ValueError("duration must be between 0.0 and 0.5 seconds")

    @classmethod
    def stop(cls) -> BrainAction:
        """Return the safe action used when no decision is available."""
        return cls()

    def as_dict(self) -> dict[str, str | float]:
        """Serialize the action for a model/tool boundary."""
        return {
            "movement": self.movement.value,
            "rotation": self.rotation,
            "duration": self.duration,
        }


MAX_FRAME_BYTES: Final = 8_000_000


class VyronBrain:
    """Decision boundary for Gemini Live or another multimodal provider.

    The provider-specific streaming client will be added behind this class.
    Until it is connected, every decision is an explicit safe stop.
    """

    def __init__(self) -> None:
        self._last_action = BrainAction.stop()

    @property
    def last_action(self) -> BrainAction:
        return self._last_action

    def observe(self, frame: np.ndarray) -> BrainAction:
        """Accept one BGR OpenCV frame and return the next bounded action."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3)")
        if frame.nbytes > MAX_FRAME_BYTES:
            raise ValueError("frame exceeds the maximum supported size")

        # Decision logic is intentionally unavailable until the Gemini Live
        # transport is connected; stopping is safer than guessing movement.
        self._last_action = BrainAction.stop()
        return self._last_action

    def action_from_response(self, response: Mapping[str, object]) -> BrainAction:
        """Validate a model/tool response before it reaches the game loop."""
        movement = Movement(str(response.get("movement", Movement.NONE.value)))
        rotation = float(response.get("rotation", 0.0))
        duration = float(response.get("duration", 0.0))
        action = BrainAction(movement, rotation, duration)
        self._last_action = action
        return action