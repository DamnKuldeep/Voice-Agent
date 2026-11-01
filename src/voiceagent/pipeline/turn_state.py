"""State and per-turn data shared by the session and the reply flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..core.metrics import TurnTrace
from .responder import Responder


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    ENDPOINTING = "endpointing"
    THINKING = "thinking"
    SPEAKING = "speaking"

    @property
    def public(self) -> str:
        """The UI-facing name: listening and endpointing look the same from outside."""
        return "listening" if self is State.ENDPOINTING else self.value


@dataclass
class Turn:
    trace: TurnTrace
    frames: list[np.ndarray]
    last_speech_at: float
    carry: str = ""                 # words already recognised before this turn's audio started
    partial: str = ""
    silence_ms: int = 0
    eot_prob: float | None = None
    eot_task: asyncio.Future[float] | None = None
    user_text: str = ""
    reply_text: str = ""
    responder: Responder | None = None
    task: asyncio.Task[None] | None = None
    quiet_ms: int = 0               # audio below the release threshold since the last speech frame
    tail_left_ms: int = 0           # post-commit audio still to feed the recogniser before finishing
    tail_done: asyncio.Event = field(default_factory=asyncio.Event)
    late: list[np.ndarray] = field(default_factory=list)  # frames after the tail, while the reply is prepared
