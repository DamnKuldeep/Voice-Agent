"""The user-side frame logic with stand-ins: end-of-speech hangover and the post-commit ASR tail."""

import numpy as np

from pipeline_fakes import make_settings
from voiceagent.audio.frames import PreRollBuffer
from voiceagent.core.events import Mark
from voiceagent.core.metrics import TurnTrace
from voiceagent.pipeline.listening import Listening
from voiceagent.pipeline.turn_state import State, Turn

FRAME = 512
SPEECH, QUIET = 0.9, 0.05


class Recorder:
    """Stands in for the recogniser, the endpointer and the reply flow; records what reaches each."""

    def __init__(self, commit_at_ms: int = 10_000) -> None:
        self.fed: list[float] = []
        self.commit_at_ms = commit_at_ms
        self.speculation_cancels = 0

    def feed(self, audio: np.ndarray) -> None:
        self.fed.append(float(audio[0]))

    def tick(self, turn: Turn) -> bool:
        turn.silence_ms += 32
        return turn.silence_ms >= self.commit_at_ms

    def cancel(self, turn: Turn) -> None:
        pass

    def cancel_speculation(self, turn: Turn) -> None:
        self.speculation_cancels += 1

    def maybe_speculate(self, turn: Turn) -> None:
        pass

    def abandon(self, turn: Turn) -> None:
        pass


class Rig:
    def __init__(self, commit_at_ms: int = 10_000, **vad_or_asr) -> None:
        sections = {k: v for k, v in vad_or_asr.items() if isinstance(v, dict)}
        settings = make_settings(**sections)
        self.rec = Recorder(commit_at_ms)
        self.states: list[State] = []
        self.commits: list[Turn] = []
        self.turn = Turn(TurnTrace(1), [], 0.0)
        self.listen = Listening(settings, self.rec, self.rec, self.rec, PreRollBuffer(500),  # type: ignore[arg-type]
                                self.states.append, lambda *_: None, self.commits.append)
        self.state = State.LISTENING
        self.stamp = 0

    def frames(self, prob: float, count: int) -> None:
        for _ in range(count):
            self.stamp += 1
            frame = np.full(FRAME, float(self.stamp), dtype=np.float32)
            handler = {State.LISTENING: self.listen.listening, State.ENDPOINTING: self.listen.endpointing,
                       State.THINKING: self.listen.thinking}[self.state]
            handler(self.turn, frame, prob)
            if self.states:
                self.state = self.states[-1]


def test_a_dip_shorter_than_the_hangover_does_not_end_speech():
    rig = Rig()
    rig.frames(SPEECH, 5)
    rig.frames(QUIET, 2)                       # 64 ms < 96 ms hangover
    rig.frames(SPEECH, 3)
    assert rig.states == [] and Mark.SPEECH_END not in rig.turn.trace.marks
    rig.frames(QUIET, 3)                       # 96 ms: now it is the end of speech
    assert rig.states == [State.ENDPOINTING] and rig.turn.silence_ms == 96
    assert len(rig.rec.fed) == 13              # every frame still reached the recogniser


def test_without_hangover_one_quiet_frame_starts_endpointing():
    rig = Rig(vad={"end_hangover_ms": 0})
    rig.frames(SPEECH, 3)
    rig.frames(QUIET, 1)
    assert rig.states == [State.ENDPOINTING] and rig.turn.silence_ms == 32


def test_after_an_early_commit_the_rest_of_the_tail_still_reaches_the_recogniser():
    rig = Rig(commit_at_ms=96)
    rig.frames(SPEECH, 5)
    rig.frames(QUIET, 3)                       # the hangover ends and the turn commits at 96 ms of silence
    assert rig.commits and rig.state is State.THINKING
    assert rig.turn.tail_left_ms == 64 and not rig.turn.tail_done.is_set()  # 160 ms tail less 96 heard
    committed_at = rig.stamp
    rig.frames(QUIET, 8)
    tail = [s for s in rig.rec.fed if s > committed_at]
    assert tail == [committed_at + 1, committed_at + 2]       # 2 frames of 32 ms, then no more
    assert rig.turn.tail_done.is_set()
    assert rig.turn.frames[-1][0] == committed_at + 2


def test_no_tail_wait_when_the_pause_already_supplied_it():
    rig = Rig(commit_at_ms=300)
    rig.frames(SPEECH, 5)
    rig.frames(QUIET, 10)                      # 320 ms of pause, all of it fed (gate off)
    assert rig.commits and rig.turn.tail_left_ms == 0 and rig.turn.tail_done.is_set()


def test_no_tail_once_the_vad_gate_had_stopped_feeding():
    rig = Rig(commit_at_ms=900, vad={"gate_asr": True})
    rig.frames(SPEECH, 5)
    rig.frames(QUIET, 30)                      # 960 ms: gated off at 300 ms, committed at 900 ms
    assert rig.commits and rig.turn.tail_left_ms == 0 and rig.turn.tail_done.is_set()


def test_tail_can_be_switched_off():
    rig = Rig(commit_at_ms=160, asr={"tail_ms": 0})
    rig.frames(SPEECH, 5)
    rig.frames(QUIET, 5)
    assert rig.turn.tail_done.is_set()
    fed = len(rig.rec.fed)
    rig.frames(QUIET, 3)
    assert len(rig.rec.fed) == fed
