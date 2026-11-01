"""While the agent speaks: watch the mic, duck on a candidate interruption, and stop or resume.

Duck-then-confirm: after ``min_speech_ms`` of speech by VAD alone the playback volume drops to
``duck_gain`` (what the user perceives as the agent yielding); the words then decide. A non-backchannel
word stops playback; noise, a backchannel or the agent's own echo restores the volume. While a candidate
is open no new TTS synthesis starts, so the GPU is free for the recogniser.

Every frame of a candidate, pre-roll first, ends up in the new turn exactly once and in order: frames
are fed to the recogniser as they arrive, except while a backchannel check is finishing the recogniser,
when they are held in ``_unfed`` and fed to the new turn's recogniser after its reset.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..audio.frames import PreRollBuffer
from ..config import BargeIn
from ..core.events import Mark
from .asr_stage import AsrStage
from .barge_in import BargeEvent, BargeInDetector
from .echo_guard import ECHO_WINDOW_MS, EchoVerdict, echo_verdict
from .playout import Playout
from .turn_state import Turn

log = logging.getLogger(__name__)


@dataclass
class Interruption:
    audio: np.ndarray               # the whole candidate, pre-roll first: the new turn's audio
    onset_at: float
    carry: str                      # words recognised already and no longer held by the recogniser
    asr_audio: np.ndarray | None    # reset the recogniser and feed this; None: it already has it all


class BargeController:
    def __init__(
        self, cfg: BargeIn, end_silence_ms: int, frame_ms: int, asr: AsrStage, preroll: PreRollBuffer,
        playout: Playout, synth_gate: asyncio.Event, emit: Callable[[dict], None],
        on_interrupt: Callable[[Turn, Interruption], None], still_speaking: Callable[[Turn], bool],
    ) -> None:
        self._cfg, self._frame_ms = cfg, frame_ms
        self._detector = BargeInDetector(cfg, end_silence_ms)
        self._asr, self._preroll, self._playout = asr, preroll, playout
        self._gate, self._emit = synth_gate, emit
        self._on_interrupt, self._still_speaking = on_interrupt, still_speaking
        self._task: asyncio.Task[None] | None = None
        self._frames: list[np.ndarray] = []
        self._unfed: list[np.ndarray] = []
        self._onset_at = 0.0
        self._ducked = False
        self._heard = ""

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def handle(self, turn: Turn, frame: np.ndarray, prob: float) -> None:
        if not self._cfg.enabled:
            return
        self._preroll.push(frame)
        if self._detector.active:
            self._frames.append(frame)
            if self._task is None:
                self._asr.feed(frame)
            else:
                self._unfed.append(frame)
        for event in self._detector.process(prob, self._frame_ms):
            if event is BargeEvent.ONSET:
                self._onset(turn)
            elif event is BargeEvent.DUCK:
                self._duck(turn)
            elif event is BargeEvent.CONFIRMED:
                self._confirm(turn, carry="", asr_audio=None)
            elif event is BargeEvent.CHECK:
                self._unfed = []
                self._task = asyncio.ensure_future(self._check(turn))
            elif event is BargeEvent.DISMISSED:
                self._dismiss()

    def on_partial(self, turn: Turn, text: str) -> None:
        if self._task is not None or not self._detector.active:
            return
        if text and text != self._heard:
            self._heard = text
            self._emit({"type": "partial", "text": text, "barge": True})
        if self._cfg.echo_guard:
            verdict = echo_verdict(text, self._playout.recent_text(ECHO_WINDOW_MS))
            if verdict is EchoVerdict.ECHO:
                self._dismiss()
                return
            if verdict is EchoVerdict.UNDECIDED:
                return
        if self._detector.on_partial(text) is BargeEvent.CONFIRMED:
            self._confirm(turn, carry="", asr_audio=None)

    def take(self) -> Interruption | None:
        """The reply ended on its own during a candidate: hand over the candidate if it was real speech."""
        if not self._ducked:
            self.cancel()
            return None
        checking = self._task is not None
        audio = self._audio()
        cut = Interruption(audio, self._onset_at, "", audio if checking else None)
        self._end("stop", report=True)
        return cut

    def cancel(self) -> None:
        self._end("resume", report=self._ducked)

    async def _check(self, turn: Turn) -> None:
        """A short phrase ended: let the recogniser say whether it was more than "uh-huh"."""
        try:
            text = await self._asr.finish()
        except Exception as exc:  # noqa: BLE001  a failed check is treated as noise
            log.warning("backchannel check failed: %s", exc)
            text = ""
        self._task = None
        if not self._still_speaking(turn):
            return
        if text:
            self._emit({"type": "partial", "text": text, "barge": True})
        echo = (self._cfg.echo_guard and echo_verdict(
            text, self._playout.recent_text(ECHO_WINDOW_MS), final=True) is EchoVerdict.ECHO)
        if self._detector.resolve(text) is BargeEvent.CONFIRMED and not echo:
            unfed = np.concatenate(self._unfed) if self._unfed else np.zeros(0, dtype=np.float32)
            self._confirm(turn, carry=text, asr_audio=unfed)
        else:
            self._dismiss()

    def _onset(self, turn: Turn) -> None:
        self._onset_at = time.perf_counter()
        turn.trace.mark(Mark.BARGE_IN_ONSET, self._onset_at)
        self._frames = [self._preroll.drain()]
        self._unfed = []
        self._heard = ""
        self._gate.clear()
        self._asr.begin_turn()
        self._asr.feed(self._frames[0])

    def _duck(self, turn: Turn) -> None:
        self._ducked = True
        self._playout.set_gain(self._cfg.duck_gain)
        turn.trace.mark(Mark.BARGE_IN_DUCKED)
        self._emit({"type": "barge", "phase": "duck"})

    def _confirm(self, turn: Turn, *, carry: str, asr_audio: np.ndarray | None) -> None:
        cut = Interruption(self._audio(), self._onset_at, carry, asr_audio)
        self._end("stop", report=True)
        self._on_interrupt(turn, cut)

    def _dismiss(self) -> None:
        self._end("resume", report=self._ducked)
        self._asr.begin_turn()

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)

    def _end(self, phase: str, *, report: bool) -> None:
        """Close the candidate: release the TTS and the volume, and report ``phase`` if asked to."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._detector.reset()
        self._frames, self._unfed, self._heard = [], [], ""
        self._gate.set()
        if self._ducked and phase == "resume":
            self._playout.set_gain(1.0)
        self._ducked = False
        if report:
            self._emit({"type": "barge", "phase": phase})
