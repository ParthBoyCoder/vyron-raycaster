"""Autonomous control interface for Vyron.

The brain owns perception and decision-making. The game loop owns simulation,
collision, rendering, and execution of validated actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
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
        if not math.isfinite(self.rotation) or not -1.0 <= self.rotation <= 1.0:
            raise ValueError("rotation must be between -1.0 and 1.0")
        if not math.isfinite(self.duration) or not 0.0 <= self.duration <= 0.5:
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
GEMINI_MODEL: Final[str] = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live")
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
        self._action_expires_at = 0.0
        self._lock = threading.Lock()
        self._status = "not started"
        self._provider_ready = self._can_use_live_provider()

    @staticmethod
    def _load_env_file() -> None:
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    @classmethod
    def _can_use_live_provider(cls) -> bool:
        cls._load_env_file()
        return bool(os.environ.get("GEMINI_API_KEY")) and cls._has_google_genai()

    @staticmethod
    def _has_google_genai() -> bool:
        try:
            import google.genai  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def last_action(self) -> BrainAction:
        with self._lock:
            if time.monotonic() >= self._action_expires_at:
                return BrainAction.stop()
            return self._last_action

    @property
    def status(self) -> str:
        """Return the latest provider/session status for the UI or logs."""
        with self._lock:
            return self._status

    @property
    def provider_ready(self) -> bool:
        return self._provider_ready

    def observe(self, frame: np.ndarray) -> BrainAction:
        """Accept one BGR OpenCV frame and return the next bounded action."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3)")
        if frame.nbytes > MAX_FRAME_BYTES:
            raise ValueError("frame exceeds the maximum supported size")

        self._provider_ready = self._can_use_live_provider()
        if self._provider_ready:
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
            return self.last_action

        action = self._decide_local_action(frame)
        with self._lock:
            self._last_action = action
            self._action_expires_at = time.monotonic() + action.duration
        self._set_status("local autonomy")
        return action

    def action_from_response(self, response: Mapping[str, object]) -> BrainAction:
        """Validate a model/tool response before it reaches the game loop."""
        movement = Movement(str(response.get("movement", Movement.NONE.value)).lower())
        rotation = float(response.get("rotation", 0.0))
        duration = float(response.get("duration", 0.0))
        action = BrainAction(movement, rotation, duration)
        with self._lock:
            self._last_action = action
            self._action_expires_at = time.monotonic() + action.duration
        return action

    def close(self) -> None:
        """Stop the Live worker before the application exits."""
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._set_status("stopped")

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def _set_stop_action(self) -> None:
        with self._lock:
            self._last_action = BrainAction.stop()
            self._action_expires_at = 0.0

    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        if not os.environ.get("GEMINI_API_KEY"):
            self._set_status("missing GEMINI_API_KEY")
            logger.error("Set GEMINI_API_KEY before starting autonomous mode")
            return
        self._worker = threading.Thread(
            target=self._run_live_session,
            name="vyron-gemini-live",
            daemon=True,
        )
        self._worker.start()
        self._set_status("connecting")

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            raise ValueError("failed to encode frame as JPEG")
        return encoded.tobytes()

    def _decide_local_action(self, frame: np.ndarray) -> BrainAction:
        """Small safe local heuristic when no live provider is available."""
        _, width, _ = frame.shape
        left_strip = frame[:, : width // 3, :]
        center_strip = frame[:, width // 3 : 2 * width // 3, :]
        right_strip = frame[:, 2 * width // 3 :, :]

        def brightness(strip: np.ndarray) -> float:
            if strip.size == 0:
                return 0.0
            return float(strip.mean())

        left = brightness(left_strip)
        center = brightness(center_strip)
        right = brightness(right_strip)

        if center > max(left, right) * 0.95 and center > 80:
            return BrainAction(Movement.FORWARD, 0.0, 0.12)
        if left > right:
            return BrainAction(Movement.RIGHT, 0.35, 0.12)
        if right > left:
            return BrainAction(Movement.LEFT, -0.35, 0.12)
        return BrainAction(Movement.NONE, 0.15, 0.08)

    def _run_live_session(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            self._set_status("missing google-genai")
            logger.error("Gemini Live requires the google-genai package")
            return

        try:
            asyncio.run(self._live_session(genai, types))
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._set_status("connection failed")
            logger.error("Gemini Live worker stopped: %s", exc)

    async def _live_session(self, genai: Any, types: Any) -> None:
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=api_key)
            config = {
                "response_modalities": ["AUDIO"],
                "output_audio_transcription": {},
                "system_instruction": (
                    "You control a first-person ray-casting game. Avoid walls and "
                    "speak only the requested JSON object, with no markdown or explanation."
                ),
            }
            async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
                self._set_status(f"connected ({GEMINI_MODEL})")
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
            self._set_status("connection failed")
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
            content = getattr(response, "server_content", None)
            if content:
                transcription = getattr(content, "output_transcription", None)
                if transcription and transcription.text:
                    response_text += transcription.text
            if content and content.turn_complete and response_text:
                self._apply_response_text(response_text)
                response_text = ""

    def _apply_response_text(self, response_text: str) -> None:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start < 0 or end <= start:
            logger.warning("Gemini returned no action JSON: %s", response_text)
            self._set_stop_action()
            return
        try:
            payload = json.loads(response_text[start : end + 1])
            if not isinstance(payload, dict):
                raise ValueError("action response must be a JSON object")
            self.action_from_response(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid Gemini action: %s", exc)
            self._set_stop_action()
