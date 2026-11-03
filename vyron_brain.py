"""Autonomous control interface for Vyron.

The brain owns perception and decision-making. The game loop owns simulation,
collision, rendering, and execution of validated actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Mapping

import cv2
import numpy as np


logger = logging.getLogger(__name__)


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
GEMINI_API_KEY: Final[str | None] = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL: Final[str] = "gemini-2.0-flash-live-001"
FRAME_INTERVAL_SECONDS: Final[float] = 1.0
ACTION_PROMPT: Final[str] = (
    "Look at the latest game frame and choose the safest useful next action. "
    'Reply with JSON only: {"movement":"none|forward|backward|left|right",'
    '"rotation":number,"duration":number}. rotation is -1 to 1 and duration '
    "is 0 to 0.5 seconds."
)


class VyronBrain:
    """Decision boundary for a Gemini Live multimodal control session."""

    def __init__(self) -> None:
        self._last_action = BrainAction.stop()
        self._frames: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_frame_at = 0.0

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

        now = time.monotonic()
        if now - self._last_frame_at >= FRAME_INTERVAL_SECONDS:
            self._ensure_worker()
            encoded = self._encode_frame(frame)
            self._last_frame_at = now
            try:
                self._frames.put_nowait(encoded)
            except queue.Full:
                self._frames.get_nowait()
                self._frames.put_nowait(encoded)
        return self._last_action

    def action_from_response(self, response: Mapping[str, object]) -> BrainAction:
        """Validate a model/tool response before it reaches the game loop."""
        movement = Movement(str(response.get("movement", Movement.NONE.value)).lower())
        rotation = float(response.get("rotation", 0.0))
        duration = float(response.get("duration", 0.0))
        action = BrainAction(movement, rotation, duration)
        self._last_action = action
        return action

    def close(self) -> None:
        """Stop the Live worker before the application exits."""
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._run_live_session,
                name="vyron-gemini-live",
                daemon=True,
            )
            self._worker.start()

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            raise ValueError("failed to encode frame as JPEG")
        return encoded.tobytes()

    def _run_live_session(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.error("Gemini Live requires the google-genai package")
            return

        if not GEMINI_API_KEY:
            logger.error("Set the GEMINI_API_KEY environment variable before autonomous mode")
            return

        asyncio.run(self._live_session(genai, types))

    async def _live_session(self, genai: Any, types: Any) -> None:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            config = {
                "response_modalities": ["TEXT"],
                "system_instruction": (
                    "You control a first-person ray-casting game. Avoid walls and "
                    "make small, safe movements. Respond only with the requested JSON."
                ),
            }
            async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
                sender = asyncio.create_task(self._send_frames(session, types))
                receiver = asyncio.create_task(self._receive_actions(session))
                done, pending = await asyncio.wait(
                    (sender, receiver),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Gemini Live session stopped: %s", exc)

    async def _send_frames(self, session: Any, types: Any) -> None:
        while not self._stop_event.is_set():
            try:
                frame = await asyncio.to_thread(self._frames.get, True, 0.1)
            except queue.Empty:
                continue
            await session.send_realtime_input(
                video=types.Blob(data=frame, mime_type="image/jpeg")
            )
            await session.send_realtime_input(text=ACTION_PROMPT)

    async def _receive_actions(self, session: Any) -> None:
        response_text = ""
        async for response in session.receive():
            content = response.server_content
            if content and content.model_turn:
                for part in content.model_turn.parts:
                    if part.text:
                        response_text += part.text
            if content and content.turn_complete and response_text:
                self._apply_response_text(response_text)
                response_text = ""

    def _apply_response_text(self, response_text: str) -> None:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start < 0 or end <= start:
            logger.warning("Gemini returned no action JSON: %s", response_text)
            self._last_action = BrainAction.stop()
            return
        try:
            payload = json.loads(response_text[start : end + 1])
            if not isinstance(payload, dict):
                raise ValueError("action response must be a JSON object")
            self.action_from_response(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid Gemini action: %s", exc)
            self._last_action = BrainAction.stop()