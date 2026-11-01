"""The live conversation: mic frames in, state machine, speech frames out.

Silence is counted in audio time (frames received), so decisions do not depend on scheduling jitter and a
replay faster than real time behaves like a live one. Wall-clock time is used only for the trace marks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..audio.frames import PreRollBuffer
from ..config import Settings
from ..core.events import Flag, Mark
from ..core.metrics import TurnTrace, summarize
from ..llm.persona import load_persona
from .asr_stage import AsrStage
from .barge_control import BargeController, Interruption
from .endpointing import Endpointer
from .history import History
from .listening import Listening, join_partial
from .mic_input import MicInput
from .playout import Playout
from .reply_flow import ReplyFlow
from .trace_log import TRACE_DIR, TraceWriter
from .turn_state import State, Turn

if TYPE_CHECKING:
    from .factory import Components

log = logging.getLogger(__name__)


class ConversationSession:
    def __init__(self, settings: Settings, components: Components, *, trace_dir: Path | None = TRACE_DIR):
        self._cfg = settings
        self._c = components
        audio = settings.audio
        turn_cfg = settings.turn
        if components.turn_detector is None:
            turn_cfg = turn_cfg.model_copy(update={"detector": "silence"})  # no turn model: fixed silence
        self._mic = MicInput(audio.analysis_rate * audio.frame_ms // 1000, self._on_frame, self._emit)
        self._preroll = PreRollBuffer(settings.vad.pre_roll_ms + settings.vad.min_speech_ms,
                                      audio.analysis_rate)
        self._playout = Playout(audio.output_rate)
        self._history = History(settings.llm.history_turns)
        self._asr = AsrStage(components.asr, components.asr_pool, settings.asr.chunk_ms,
                             self._on_partial, self._on_asr_error,
                             lambda ms: self._emit({"type": "asr_lag", "ms": ms}))
        self._flow = ReplyFlow(
            settings, components.chat, components.tts, components.tts_pool, self._playout,
            self._history, self._asr, load_persona(settings.llm.persona_file), components.fillers,
            self._emit, self._set_state)
        self._endpointer = Endpointer(turn_cfg, components.turn_detector, components.turn_pool,
                                      audio.frame_ms)
        self._listen = Listening(
            settings, self._endpointer, self._asr, self._flow, self._preroll, self._set_state,
            lambda audio_, at, carry: self._begin_turn(audio_, at, carry=carry), self._start_reply)
        self._barge = BargeController(
            settings.barge_in, settings.turn.min_silence_ms, audio.frame_ms, self._asr, self._preroll,
            self._playout, self._flow.synth_gate, self._emit, self._interrupt,
            lambda turn: self._state is State.SPEAKING and self._turn is turn)
        self._writer = (TraceWriter(trace_dir, settings, components.vram_peak_mib)
                        if trace_dir is not None else None)
        self._listener: Callable[[dict], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state = State.IDLE
        self._turn: Turn | None = None
        self._turns_started = 0
        self.traces: list[TurnTrace] = []

    @property
    def state(self) -> str:
        return self._state.public

    @property
    def dropped_frames(self) -> int:
        return self._mic.dropped_frames

    def set_listener(self, fn: Callable[[dict], None]) -> None:
        self._listener = fn

    def summary(self) -> dict[str, dict[str, float]]:
        return summarize(self.traces)

    def reset_conversation(self) -> None:
        self._history.clear()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._asr.start()
        self._mic.start(self._loop)
        prewarm = getattr(self._c.chat, "prewarm", None)
        if prewarm is not None:
            try:
                await prewarm()
            except Exception as exc:  # noqa: BLE001  the first turn just pays the connection cost
                log.warning("LLM prewarm failed: %s", exc)

    async def stop(self) -> None:
        """Stop intake, cancel the turn in progress and wait for every task this session started."""
        await self._mic.stop()
        turn, tasks = self._turn, set()
        if turn is not None:
            self._flow.abandon(turn)
            self._endpointer.cancel(turn)
            if turn.task is not None:
                tasks.add(turn.task)
        if self._barge.task is not None:
            tasks.add(self._barge.task)
        self._barge.cancel()
        self._turn = None
        self._set_state(State.IDLE)
        if tasks:
            await asyncio.wait(tasks)
        await self._flow.aclose()
        await self._asr.stop()
        self._loop = None

    def feed_mic(self, samples: np.ndarray, rate: int) -> None:
        self._mic.feed(samples, rate)

    def next_playout_frame(self) -> np.ndarray | None:
        frame = self._playout.next_frame()
        turn = self._turn
        if frame is not None and turn is not None and Mark.COMMIT in turn.trace.marks:
            turn.trace.mark(Mark.AUDIO_OUT_START)
        return frame

    def _emit(self, event: dict) -> None:
        if self._listener is not None:
            try:
                self._listener(event)
            except Exception:
                log.exception("listener failed")

    def _set_state(self, state: State) -> None:
        if state is self._state:
            return
        changed = state.public != self._state.public
        self._state = state
        if changed:
            self._emit({"type": "state", "state": state.public})

    def _on_partial(self, text: str) -> None:
        turn = self._turn
        if turn is not None and self._state is State.SPEAKING:
            self._barge.on_partial(turn, text)
            return
        if turn is None or self._state not in (State.LISTENING, State.ENDPOINTING, State.THINKING):
            return
        if self._state is not State.THINKING:  # after the commit only the tail is still being decoded
            self._flow.spec.observe(text)  # unchanged partials count: stability is measured in chunks
        if text != turn.partial:
            turn.partial = text
            self._emit({"type": "partial", "text": join_partial(turn)})  # carry included: the whole utterance

    def _on_asr_error(self, exc: Exception) -> None:
        log.warning("ASR error during a turn: %s", exc)

    # Per-frame state machine ------------------------------------------------------------------------

    def _on_frame(self, frame: np.ndarray) -> None:
        prob = self._c.vad.process(frame)
        turn = self._turn
        if self._state is State.IDLE or turn is None:
            self._listen.idle(frame, prob)
            return
        handler = {
            State.LISTENING: self._listen.listening, State.ENDPOINTING: self._listen.endpointing,
            State.THINKING: self._listen.thinking, State.SPEAKING: self._barge.handle,
        }[self._state]
        handler(turn, frame, prob)

    def _begin_turn(self, audio: np.ndarray, started_at: float, *, carry: str = "",
                    asr_audio: np.ndarray | None = None, continue_asr: bool = False) -> None:
        """Open a turn on ``audio``. The recogniser is reset and fed ``asr_audio`` (default: ``audio``),
        or with ``continue_asr`` keeps what it already holds; earlier words travel in ``carry``."""
        self._turns_started += 1
        trace = TurnTrace(self._turns_started)
        trace.mark(Mark.SPEECH_START, started_at)
        self._turn = Turn(trace, [audio], started_at, carry=carry)
        self._listen.reset_run()
        self._preroll.drain()  # anything left is either in ``audio`` already or stale
        self._flow.spec.reset()
        if not continue_asr:
            self._asr.take_max_lag_ms()
            self._asr.begin_turn()
            self._asr.feed(audio if asr_audio is None else asr_audio)
        self._set_state(State.LISTENING)

    def _begin_from(self, cut: Interruption) -> None:
        self._begin_turn(cut.audio, cut.onset_at, carry=cut.carry, asr_audio=cut.asr_audio,
                         continue_asr=cut.asr_audio is None)

    def _start_reply(self, turn: Turn) -> None:
        turn.task = asyncio.ensure_future(self._turn_main(turn))

    async def _turn_main(self, turn: Turn) -> None:
        try:
            completed = await self._flow.run(turn)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("turn failed")
            completed = False
        cut = self._barge.take() if self._turn is turn else None
        self._finish_turn(turn, completed, to_idle=cut is None)
        if cut is not None:
            self._begin_from(cut)  # the user started talking as the reply ended: keep every word

    def _finish_turn(self, turn: Turn, completed: bool, *, to_idle: bool = True) -> None:
        if completed:
            turn.trace.flag(Flag.ASR_MAX_LAG_MS, round(self._asr.take_max_lag_ms(), 1))
            self.traces.append(turn.trace)
            if self._writer is not None and self._loop is not None:
                self._loop.run_in_executor(None, self._writer.write, turn.trace, turn.user_text,
                                           turn.reply_text)
            self._emit({
                "type": "turn", "metrics": turn.trace.derived(),
                "flags": {f.value: v for f, v in turn.trace.flags.items()},
                "transcript": turn.user_text, "reply": turn.reply_text,
            })
        if self._turn is turn:
            self._turn = None
        self._barge.cancel()
        self._listen.reset_run()
        if to_idle:
            self._set_state(State.IDLE)

    def _interrupt(self, turn: Turn, cut: Interruption) -> None:
        self._flow.interrupt(turn)
        self._finish_turn(turn, completed=True, to_idle=False)
        self._begin_from(cut)
