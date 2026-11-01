"""Builds the real model adapters once, warms them up and checks they sit on the GPU."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv

from ..config import Settings
from ..core.interfaces import (
    ChatModel,
    SpeechSynthesizer,
    StreamingASR,
    TurnDetector,
    VoiceActivityDetector,
)

log = logging.getLogger(__name__)

# asr.backend -> (module, class). A new recogniser is one adapter module, one entry here and one
# Literal choice in config.Asr.
ASR_BACKENDS = {"nemotron": ("voiceagent.asr.nemotron", "NemotronStreamingASR")}
FILLER_PHRASES = ("Hmm, let me think.", "One moment.", "Let me see.")


def _pool(name: str) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)


@dataclass
class Components:
    vad: VoiceActivityDetector
    turn_detector: TurnDetector | None
    asr: StreamingASR
    chat: ChatModel
    tts: SpeechSynthesizer
    fillers: list[np.ndarray] = field(default_factory=list)
    vram_peak_mib: Callable[[], float] | None = None
    # One thread each, shared by every session: models keep their per-thread GPU state warm.
    asr_pool: ThreadPoolExecutor = field(default_factory=lambda: _pool("asr"))
    tts_pool: ThreadPoolExecutor = field(default_factory=lambda: _pool("tts"))
    turn_pool: ThreadPoolExecutor = field(default_factory=lambda: _pool("turn"))

    def close(self) -> None:
        """Release the models and threads. The LLM client closes on the running loop if there is one."""
        closer = getattr(self.asr, "close", None)
        if closer is not None:
            closer()
        for pool in (self.asr_pool, self.tts_pool, self.turn_pool):
            pool.shutdown(wait=False, cancel_futures=True)
        aclose = getattr(self.chat, "aclose", None)
        if aclose is None:
            return
        try:
            asyncio.get_running_loop().create_task(aclose())
        except RuntimeError:
            asyncio.run(aclose())


def _build_asr(settings: Settings, say: Callable[[str], None]) -> StreamingASR:
    """Import the configured adapter only now, so its model library loads only when it is used."""
    backend = settings.asr.backend
    say(f"Loading speech recognition ({backend})")
    module, name = ASR_BACKENDS[backend]
    return getattr(importlib.import_module(module), name)(settings.asr, settings.devices.asr)


def _build_turn_detector(settings: Settings, say: Callable[[str], None]) -> TurnDetector | None:
    if settings.turn.detector != "smart_turn":
        return None
    say("Loading turn-end model")
    try:
        from ..audio.smart_turn import SmartTurnDetector
        return SmartTurnDetector(device=settings.devices.turn)
    except Exception as exc:  # noqa: BLE001  documented fallback: fixed-silence endpointing
        log.warning("turn model unavailable, using fixed silence: %s", exc)
        return None


def _render_fillers(tts: SpeechSynthesizer, pool: ThreadPoolExecutor) -> list[np.ndarray]:
    def render() -> list[np.ndarray]:
        return [np.concatenate(list(tts.synthesize(phrase))) for phrase in FILLER_PHRASES]

    return pool.submit(render).result()


def build_components(settings: Settings, progress: Callable[[str], None] | None = None) -> Components:
    from .. import gpu
    from ..audio.silero import SileroVAD
    from ..llm.openrouter import OpenRouterChat
    from ..tts.kokoro import KokoroSynthesizer

    say = progress or (lambda _msg: None)
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; put it in .env")
    gpu.apply_cpu_threads(settings.devices.cpu_threads)
    devices = settings.devices

    say("Loading voice activity detector")
    vad = SileroVAD(device=devices.vad)
    turn_detector = _build_turn_detector(settings, say)
    asr = _build_asr(settings, say)
    say("Loading speech synthesis")
    tts = KokoroSynthesizer(settings.tts, devices.tts)
    components = Components(
        vad=vad, turn_detector=turn_detector, asr=asr, tts=tts,
        chat=OpenRouterChat(settings.llm, api_key),
        vram_peak_mib=lambda: gpu.process_memory_mib(devices.tts)["peak_allocated"],
    )
    say("Warming up")
    components.asr_pool.submit(asr.warmup).result()
    if settings.tts.warmup:
        components.tts_pool.submit(tts.warmup).result()
    if devices.require_full_gpu_residency:
        gpu.assert_resident("asr", asr.torch_modules(), devices.asr)
        gpu.assert_resident("tts", tts.torch_modules(), devices.tts)
    if settings.tts.fillers:
        components.fillers = _render_fillers(tts, components.tts_pool)
    return components
