"""Pure audio-buffer helpers: resampling to the 16 kHz analysis rate, framing, pre-roll."""

from __future__ import annotations

from collections import deque
from math import gcd

import numpy as np
import torch
import torchaudio.functional as taf

ANALYSIS_RATE = 16000


def _to_mono_float(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32, copy=False)
    if x.ndim == 2:
        # Channels are the short axis whichever way the transport lays the array out.
        x = x.mean(axis=0 if x.shape[0] <= x.shape[1] else 1)
    return np.ascontiguousarray(x.reshape(-1), dtype=np.float32)


def resample_to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    """Return float32 mono 16 kHz in [-1, 1] from int16/float32, mono or multichannel input."""
    x = _to_mono_float(samples)
    if rate != ANALYSIS_RATE and x.size:
        g = gcd(rate, ANALYSIS_RATE)
        with torch.inference_mode():
            y = taf.resample(torch.from_numpy(x), rate // g, ANALYSIS_RATE // g)
        x = y.numpy()
    return np.clip(x, -1.0, 1.0)


class FrameCutter:
    """Re-blocks an arbitrary-length sample stream into exact fixed-size frames."""

    def __init__(self, frame_samples: int) -> None:
        self._n = frame_samples
        self._rest = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        buf = np.concatenate([self._rest, samples.astype(np.float32, copy=False)])
        count = len(buf) // self._n
        self._rest = buf[count * self._n :].copy()
        return [buf[i * self._n : (i + 1) * self._n].copy() for i in range(count)]


class PreRollBuffer:
    """Keeps the most recent ``ms`` of audio so speech onset is not clipped."""

    def __init__(self, ms: int, rate: int = ANALYSIS_RATE) -> None:
        self._max = ms * rate // 1000
        self._frames: deque[np.ndarray] = deque()
        self._size = 0

    def push(self, frame: np.ndarray) -> None:
        self._frames.append(frame)
        self._size += len(frame)
        while self._frames and self._size - len(self._frames[0]) >= self._max:
            self._size -= len(self._frames.popleft())

    def drain(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(list(self._frames))
        self._frames.clear()
        self._size = 0
        return out[-self._max :] if self._max else out[:0]
