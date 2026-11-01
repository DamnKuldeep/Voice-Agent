"""BargeController with a scripted recogniser: every candidate frame reaches the new turn once, in order."""

import asyncio

import numpy as np
import pytest

from voiceagent.audio.frames import PreRollBuffer
from voiceagent.config import load_settings
from voiceagent.core.events import Mark
from voiceagent.core.metrics import TurnTrace
from voiceagent.pipeline.barge_control import BargeController, Interruption
from voiceagent.pipeline.playout import Playout
from voiceagent.pipeline.turn_state import Turn

FRAME_MS = 32
FRAME = 512
SPEECH, QUIET = 0.9, 0.1
CFG = load_settings().barge_in
AGENT_SAYS = "the weather today is sunny and warm with a light breeze"


class ScriptedAsr:
    """Records what the controller hands the recogniser; ``finish`` waits until the test answers."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, np.ndarray | None]] = []
        self.answer: asyncio.Future[str] | None = None

    def begin_turn(self) -> None:
        self.calls.append(("reset", None))

    def feed(self, audio: np.ndarray) -> None:
        self.calls.append(("feed", audio.copy()))

    async def finish(self) -> str:
        self.calls.append(("finish", None))
        self.answer = asyncio.get_running_loop().create_future()
        return await self.answer

    def fed_since_last_reset(self) -> np.ndarray:
        last = max(i for i, (kind, _) in enumerate(self.calls) if kind == "reset")
        parts = [a for kind, a in self.calls[last:] if kind == "feed" and a is not None]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


class Rig:
    def __init__(self, **changes) -> None:
        self.asr = ScriptedAsr()
        self.playout = Playout(24000)
        self.playout.begin_reply()
        self.playout.begin_chunk(AGENT_SAYS)
        self.playout.write(np.full(24000 * 200 // 1000, 0.5, dtype=np.float32))
        self.playout.end_chunk()
        for _ in range(10):
            self.playout.next_frame()
        self.gate = asyncio.Event()
        self.gate.set()
        self.events: list[dict] = []
        self.cuts: list[Interruption] = []
        self.turn = Turn(TurnTrace(1), [], 0.0)
        self.stamp = 0
        self.ctl = BargeController(
            CFG.model_copy(update=changes), 160, FRAME_MS, self.asr,  # type: ignore[arg-type]
            PreRollBuffer(500), self.playout, self.gate, self.events.append,
            lambda _turn, cut: self.cuts.append(cut), lambda _turn: True)

    def frames(self, prob: float, count: int) -> list[np.ndarray]:
        """Feed ``count`` frames, each filled with its own serial number so order can be checked."""
        made = []
        for _ in range(count):
            self.stamp += 1
            frame = np.full(FRAME, float(self.stamp), dtype=np.float32)
            made.append(frame)
            self.ctl.handle(self.turn, frame, prob)
        return made

    def phases(self) -> list[str]:
        return [e["phase"] for e in self.events if e["type"] == "barge"]


def stamps(audio: np.ndarray) -> list[int]:
    return [int(v) for v in audio[::FRAME]]


async def test_frames_spoken_during_the_backchannel_check_reach_the_new_turn():
    rig = Rig()
    rig.frames(QUIET, 5)                                   # pre-roll
    rig.frames(SPEECH, 13)                                 # 416 ms: ducked, then a short phrase
    rig.frames(QUIET, 5)                                   # 160 ms of silence starts the check
    assert rig.ctl.task is not None and not rig.gate.is_set()
    await asyncio.sleep(0)
    during = rig.frames(SPEECH, 4)                         # the user goes on while the check runs
    rig.asr.answer.set_result("wait a second")
    await asyncio.sleep(0)
    cut = rig.cuts[0]
    assert cut.carry == "wait a second"
    assert stamps(cut.asr_audio) == stamps(np.concatenate(during))
    assert stamps(cut.audio) == list(range(1, rig.stamp + 1))  # pre-roll and every frame once, in order
    heard_before_check = [a for kind, a in rig.asr.calls[:rig.asr.calls.index(("finish", None))]
                          if kind == "feed"]
    assert stamps(np.concatenate(heard_before_check + [cut.asr_audio])) == list(range(1, rig.stamp + 1))
    assert rig.phases() == ["duck", "stop"] and rig.gate.is_set()


async def test_a_long_candidate_is_confirmed_with_the_recogniser_holding_everything():
    rig = Rig()
    rig.frames(QUIET, 3)
    rig.frames(SPEECH, 22)                                 # 704 ms: just past backchannel_max_ms
    cut = rig.cuts[0]
    assert cut.asr_audio is None and cut.carry == ""
    assert stamps(rig.asr.fed_since_last_reset()) == stamps(cut.audio) == list(range(1, rig.stamp + 1))


async def test_duck_then_backchannel_resumes_full_volume():
    rig = Rig()
    rig.frames(SPEECH, 9)
    assert rig.phases() == ["duck"] and rig.playout._target == CFG.duck_gain
    assert Mark.BARGE_IN_DUCKED in rig.turn.trace.marks
    rig.frames(QUIET, 5)
    await asyncio.sleep(0)
    rig.asr.answer.set_result("yeah")
    await asyncio.sleep(0)
    assert rig.phases() == ["duck", "resume"] and rig.playout._target == 1.0
    assert rig.cuts == [] and rig.gate.is_set()


async def test_a_blip_never_ducks_but_pauses_synthesis_until_dismissed():
    rig = Rig()
    rig.frames(SPEECH, 2)
    assert not rig.gate.is_set()                           # no new TTS while the candidate is open
    rig.frames(QUIET, 6)
    assert rig.phases() == [] and rig.gate.is_set() and rig.playout._target == 1.0


async def test_first_content_word_stops_playback_and_partials_are_shown():
    rig = Rig()
    rig.frames(SPEECH, 9)
    rig.ctl.on_partial(rig.turn, "hold on")
    assert len(rig.cuts) == 1 and rig.phases() == ["duck", "stop"]
    assert {"type": "partial", "text": "hold on", "barge": True} in rig.events


async def test_the_agents_own_words_coming_back_are_dismissed_as_echo():
    rig = Rig(echo_guard=True)
    rig.frames(SPEECH, 9)
    rig.ctl.on_partial(rig.turn, "the weather")
    assert rig.cuts == [] and rig.phases() == ["duck", "resume"]
    assert rig.asr.calls[-1] == ("reset", None)


async def test_echo_guard_can_be_switched_off():
    rig = Rig(echo_guard=False)
    rig.frames(SPEECH, 9)
    rig.ctl.on_partial(rig.turn, "the weather")
    assert len(rig.cuts) == 1


async def test_one_echo_like_word_waits_for_the_next():
    rig = Rig(echo_guard=True)
    rig.frames(SPEECH, 9)
    rig.ctl.on_partial(rig.turn, "weather")
    assert rig.cuts == [] and rig.phases() == ["duck"]
    rig.ctl.on_partial(rig.turn, "weather in london")
    assert len(rig.cuts) == 1


@pytest.mark.parametrize("checking", [False, True])
async def test_a_reply_ending_during_a_ducked_candidate_hands_it_over(checking):
    rig = Rig()
    rig.frames(QUIET, 2)
    rig.frames(SPEECH, 12)
    if checking:
        rig.frames(QUIET, 5)
        assert rig.ctl.task is not None
    cut = rig.ctl.take()
    assert cut is not None and stamps(cut.audio) == list(range(1, rig.stamp + 1))
    assert (cut.asr_audio is not None) == checking         # a finished recogniser must hear it all again
    assert rig.ctl.task is None and rig.phases() == ["duck", "stop"] and rig.gate.is_set()


async def test_a_reply_ending_during_a_blip_hands_nothing_over():
    rig = Rig()
    rig.frames(SPEECH, 3)
    assert rig.ctl.take() is None and rig.gate.is_set()
