"""Kokoro-82M speech synthesis, resident on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from kokoro import KModel, KPipeline

from ..config import Tts

REPO_ID = "hexgrad/Kokoro-82M"
WARMUP_TEXT = "Hello there, this is a short warm up phrase."


class KokoroSynthesizer:
    sample_rate = 24000

    def __init__(self, settings: Tts, device: str) -> None:
        self._settings = settings
        self._device = device
        self._model = KModel(repo_id=REPO_ID).to(device).eval()
        # lang_code "a" is American English; the model is passed in so the pipeline cannot reload it on the CPU.
        self._pipeline = KPipeline(lang_code="a", repo_id=REPO_ID, model=self._model)

    def torch_modules(self) -> list[torch.nn.Module]:
        return [self._model]

    def warmup(self) -> None:
        for _ in self.synthesize(WARMUP_TEXT):
            pass
        torch.cuda.synchronize(self._device)

    @torch.inference_mode()
    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        for result in self._pipeline(text, voice=self._settings.voice, speed=self._settings.speed):
            if result.audio is not None:
                yield result.audio.detach().float().cpu().numpy()
