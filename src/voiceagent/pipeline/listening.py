"""The user's side of the state machine: when speech starts, when it ends, and when the turn is committed.

Handles frames in IDLE, LISTENING, ENDPOINTING and THINKING; the session owns the state, the turn and
everything that happens once the agent answers.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..audio.frames import PreRollBuffer
from ..config import Settings
from ..core.events import Mark
from .asr_stage import AsrStage
from .endpointing import Endpointer
from .reply_flow import ReplyFlow
from .turn_state import State, Turn

VAD_HYSTERESIS = 0.15    # Silero's own default gap between the speech and the non-speech threshold
LATE_MAX_FRAMES = 300    # about 10 s of post-commit audio kept in case the user goes on talking

# begin_turn(audio, onset time, carry): open a new turn whose recogniser is reset and fed ``audio``
BeginTurn = Callable[[np.ndarray, float, str], None]


def join_partial(turn: Turn) -> str:
    return " ".join(p for p in (turn.carry, turn.partial) if p)


class Listening:
    def __init__(
        self, settings: Settings, endpointer: Endpointer, asr: AsrStage, flow: ReplyFlow,
        preroll: PreRollBuffer, set_state: Callable[[State], None], begin_turn: BeginTurn,
        on_commit: Callable[[Turn], None],
    ) -> None:
        self._vad, self._tail_ms, self._frame_ms = settings.vad, settings.asr.tail_ms, settings.audio.frame_ms
        self._neg_threshold = max(settings.vad.threshold - VAD_HYSTERESIS, 0.0)
        self._endpointer, self._asr, self._flow, self._preroll = endpointer, asr, flow, preroll
        self._set_state, self._begin_turn, self._on_commit = set_state, begin_turn, on_commit
        self._run_ms = 0
        self._run_at = 0.0

    def reset_run(self) -> None:
        self._run_ms = 0

    def idle(self, frame: np.ndarray, prob: float) -> None:
        if self._speech_run(frame, prob):
            self._begin_turn(self._preroll.drain(), self._run_at, "")

    def listening(self, turn: Turn, frame: np.ndarray, prob: float) -> None:
        self._append(turn, frame)
        if prob >= self._neg_threshold:
            turn.last_speech_at = time.perf_counter()
            turn.quiet_ms = 0
            return
        turn.quiet_ms += self._frame_ms
        if turn.quiet_ms < self._vad.end_hangover_ms:
            return  # one soft frame inside a word is not the end of speech
        turn.trace.mark(Mark.SPEECH_END, turn.last_speech_at)
        turn.silence_ms = turn.quiet_ms - self._frame_ms  # the tick below counts this frame
        self._set_state(State.ENDPOINTING)
        self._tick(turn)

    def endpointing(self, turn: Turn, frame: np.ndarray, prob: float) -> None:
        if prob >= self._vad.threshold:
            self._resume(turn, frame)
            return
        self._append(turn, frame, to_asr=not self._gated_out(turn.silence_ms))
        self._tick(turn)

    def thinking(self, turn: Turn, frame: np.ndarray, prob: float) -> None:
        in_tail = turn.tail_left_ms > 0
        if in_tail:
            self._append(turn, frame)  # a soft last word that runs past the commit
            turn.tail_left_ms -= self._frame_ms
        else:
            turn.late.append(frame)
            del turn.late[:-LATE_MAX_FRAMES]
        if not self._speech_run(frame, prob):
            if in_tail and turn.tail_left_ms <= 0:
                turn.tail_done.set()
            return
        # The user kept talking after the commit: drop the reply and recognise the whole turn again.
        # Restarting the recogniser on the new words alone (and carrying the old text) lost or garbled the
        # words at the pause ("faculty and distinguished" became "faculty distinguished") because it
        # started with no context; decoding everything together costs a little catch-up time instead.
        self._flow.abandon(turn)
        audio = np.concatenate(turn.frames + turn.late)
        self._begin_turn(audio, turn.trace.marks.get(Mark.SPEECH_START, self._run_at), "")

    def _speech_run(self, frame: np.ndarray, prob: float) -> bool:
        """Count consecutive speech frames while no turn is open; True once min_speech_ms is reached."""
        self._preroll.push(frame)
        if prob < self._vad.threshold:
            self._run_ms = 0
            return False
        if self._run_ms == 0:
            self._run_at = time.perf_counter()
        self._run_ms += self._frame_ms
        return self._run_ms >= self._vad.min_speech_ms

    def _append(self, turn: Turn, frame: np.ndarray, *, to_asr: bool = True) -> None:
        turn.frames.append(frame)
        if to_asr:
            self._asr.feed(frame)

    def _gated_out(self, silence_ms: int) -> bool:
        """Silence this long no longer reaches the recogniser (vad.gate_asr)."""
        return self._vad.gate_asr and silence_ms >= self._vad.pre_roll_ms

    def _resume(self, turn: Turn, frame: np.ndarray) -> None:
        self._flow.cancel_speculation(turn)
        self._endpointer.cancel(turn)
        turn.trace.marks.pop(Mark.SPEECH_END, None)  # the real end of speech comes later
        turn.last_speech_at = time.perf_counter()
        turn.quiet_ms = 0
        self._append(turn, frame)
        self._set_state(State.LISTENING)

    def _tick(self, turn: Turn) -> None:
        commit = self._endpointer.tick(turn)
        self._flow.maybe_speculate(turn)  # even on the commit frame: the request overlaps the final decode
        if commit:
            self._commit(turn)

    def _commit(self, turn: Turn) -> None:
        turn.trace.mark(Mark.COMMIT)
        self._endpointer.cancel(turn)
        self._run_ms = 0
        # The recogniser must hear tail_ms of real audio after the end of speech before it finishes. The
        # endpointing pause already supplied part of it (all of it when the gate is off and the pause
        # was at least tail_ms); waiting the full tail again cost 160 ms per turn for nothing
        # (asr_final p50 81 -> 322 ms, Measured).
        heard = min(turn.silence_ms, self._vad.pre_roll_ms) if self._vad.gate_asr else turn.silence_ms
        fed = not self._gated_out(turn.silence_ms - self._frame_ms)
        turn.tail_left_ms = max(0, self._tail_ms - heard) if fed else 0
        if turn.tail_left_ms <= 0:
            turn.tail_done.set()
        self._set_state(State.THINKING)
        self._on_commit(turn)
