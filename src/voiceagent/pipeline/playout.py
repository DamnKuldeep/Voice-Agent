"""Bounded output queue of 20 ms frames. Thread-safe: the transport pulls, the event loop pushes.

Volume changes (the barge-in duck) are applied as frames are handed out, not when they are queued, so
they take effect on the very next frame, and they ramp linearly so a step in gain never clicks.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

FRAME_MS = 20
MAX_AHEAD_MS = 200  # small enough that clear() leaves little stale audio, large enough to ride out jitter
GAIN_RAMP_MS = 30   # long enough not to click, short enough to sound immediate


@dataclass
class _Chunk:
    text: str
    start: int
    end: int | None = None


class Playout:
    def __init__(self, rate: int) -> None:
        self.frame_samples = rate * FRAME_MS // 1000
        self._rate = rate
        self._gain = 1.0
        self._target = 1.0
        self._gain_step = 0.0   # gain change per sample while ramping
        self._max_frames = MAX_AHEAD_MS // FRAME_MS
        self._lock = threading.Lock()
        self._frames: deque[np.ndarray] = deque()
        self._tail = np.zeros(0, dtype=np.float32)
        self._written = 0
        self._played = 0
        self._chunks: list[_Chunk] = []

    def begin_reply(self) -> None:
        with self._lock:
            self._written = self._played = 0
            self._chunks = []
            self._gain = self._target = 1.0

    def set_gain(self, target: float) -> None:
        """Ramp the output volume to ``target`` over GAIN_RAMP_MS."""
        with self._lock:
            self._target = target
            self._gain_step = abs(target - self._gain) / max(1, self._rate * GAIN_RAMP_MS // 1000)

    @property
    def gain(self) -> float:
        with self._lock:
            return self._gain

    def room(self) -> int:
        """Samples that can be written now without exceeding the bound."""
        with self._lock:
            return self._room()

    def write(self, samples: np.ndarray) -> int:
        """Queue as much of ``samples`` as fits; returns how many samples were taken."""
        with self._lock:
            take = min(len(samples), self._room())
            data = np.concatenate([self._tail, samples[:take].astype(np.float32, copy=False)])
            whole = len(data) // self.frame_samples * self.frame_samples
            for start in range(0, whole, self.frame_samples):
                self._frames.append(data[start:start + self.frame_samples])
            self._tail = data[whole:]
            self._written += take
            return take

    def begin_chunk(self, text: str) -> None:
        with self._lock:
            self._chunks.append(_Chunk(text, self._written))

    def end_chunk(self) -> None:
        with self._lock:
            if self._chunks and self._chunks[-1].end is None:
                self._chunks[-1].end = self._written

    def finish(self) -> None:
        """Pad the last partial frame with silence so the end of the reply is not held back."""
        with self._lock:
            if len(self._tail):
                pad = self.frame_samples - len(self._tail)
                self._frames.append(np.concatenate([self._tail, np.zeros(pad, dtype=np.float32)]))
                self._tail = np.zeros(0, dtype=np.float32)

    def next_frame(self) -> np.ndarray | None:
        with self._lock:
            if not self._frames:
                return None
            self._played += self.frame_samples
            return self._apply_gain(self._frames.popleft())

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._tail = np.zeros(0, dtype=np.float32)

    @property
    def idle(self) -> bool:
        with self._lock:
            return not self._frames and len(self._tail) == 0

    def spoken_text(self) -> str:
        """The words of the reply that were handed to the transport, by sample position."""
        with self._lock:
            played, parts = self._played, []
            for chunk in self._chunks:
                if played <= chunk.start:
                    break
                end = chunk.end if chunk.end is not None else self._written
                if played >= end:
                    parts.append(chunk.text)
                    continue
                words = chunk.text.split()
                share = (played - chunk.start) / (end - chunk.start)
                parts.append(" ".join(words[: int(len(words) * share)]))
            return " ".join(p for p in parts if p).strip()

    def recent_text(self, window_ms: int) -> str:
        """Words handed to the transport in the last ``window_ms``, by sample position."""
        with self._lock:
            played, since, words = self._played, self._played - self._rate * window_ms // 1000, []
            for chunk in self._chunks:
                end = chunk.end if chunk.end is not None else self._written
                parts = chunk.text.split()
                if not parts or end <= chunk.start:
                    continue
                per_word = (end - chunk.start) / len(parts)
                for index, word in enumerate(parts):
                    start = chunk.start + index * per_word
                    if start < played and start + per_word > since:
                        words.append(word)
            return " ".join(words)

    def _apply_gain(self, frame: np.ndarray) -> np.ndarray:
        if self._gain == self._target:
            return frame if self._gain == 1.0 else frame * np.float32(self._gain)
        direction = 1.0 if self._target > self._gain else -1.0
        ramp = self._gain + direction * self._gain_step * np.arange(1, len(frame) + 1, dtype=np.float32)
        ramp = np.minimum(ramp, self._target) if direction > 0 else np.maximum(ramp, self._target)
        self._gain = float(ramp[-1])
        return (frame * ramp).astype(np.float32, copy=False)

    def _room(self) -> int:
        queued = len(self._frames) * self.frame_samples + len(self._tail)
        return max(0, self._max_frames * self.frame_samples - queued)
