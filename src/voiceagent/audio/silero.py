"""Silero VAD adapter (stateful, one 512-sample frame at 16 kHz per call)."""

from __future__ import annotations

import numpy as np
import torch
from silero_vad import load_silero_vad

FRAME_SAMPLES = 512


class SileroVAD:
    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model = load_silero_vad().to(device).eval()

    def reset(self) -> None:
        self._model.reset_states()

    def process(self, frame: np.ndarray) -> float:
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(f"Silero VAD needs {FRAME_SAMPLES} samples, got {len(frame)}")
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32)).to(self._device)
            return float(self._model(x, 16000).item())
