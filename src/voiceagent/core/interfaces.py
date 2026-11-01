"""Contracts every swappable component implements. The pipeline depends only on these."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import asyncio

    import numpy as np


class VoiceActivityDetector(Protocol):
    def reset(self) -> None: ...

    def process(self, frame: np.ndarray) -> float:
        """Speech probability in [0, 1] for one 16 kHz mono float32 frame."""


class TurnDetector(Protocol):
    def end_of_turn_probability(self, audio: np.ndarray) -> float:
        """Probability the speaker has finished, given the whole current turn (16 kHz mono)."""


class StreamingASR(Protocol):
    def reset(self) -> None: ...

    def push(self, audio: np.ndarray) -> str:
        """Feed audio, return the current partial transcript."""

    def finish(self) -> str:
        """Flush and return the final transcript for the turn."""


class ChatModel(Protocol):
    def stream(
        self, messages: list[dict[str, str]], *, max_tokens: int, cancel: asyncio.Event
    ) -> AsyncIterator[str]:
        """Yield text deltas. Must stop promptly once ``cancel`` is set."""


class SpeechSynthesizer(Protocol):
    sample_rate: int

    def warmup(self) -> None: ...

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 mono PCM as it is produced, so playback can begin before the end."""


class LiveSession(Protocol):
    """What a transport needs from the running conversation (pipeline.session.ConversationSession)."""

    def feed_mic(self, samples: np.ndarray, rate: int) -> None: ...

    def next_playout_frame(self) -> np.ndarray | None: ...


class ConversationHost(Protocol):
    """Owns the one active conversation; every transport connection talks to the same host."""

    async def open_session(self, owner: object) -> LiveSession: ...

    async def close_session(self, owner: object) -> None: ...

    def session_for(self, owner: object) -> LiveSession | None: ...
