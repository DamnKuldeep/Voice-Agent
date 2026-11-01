import asyncio

import pytest

from pipeline_fakes import FakeChat, FakeTTS, make_settings
from voiceagent.core.events import Mark
from voiceagent.pipeline.playout import Playout
from voiceagent.pipeline.responder import LLM_FAILURE_REPLY, Responder

MESSAGES = [{"role": "user", "content": "hi"}]
FOUR_SENTENCES = "One two three four five. Six seven eight nine ten. Eleven twelve thirteen fourteen."


class Rig:
    def __init__(self, chat=None, fake_tts=None, fillers=(), **sections) -> None:
        self.settings = make_settings(**sections)
        self.chat = chat or FakeChat()
        self.tts = fake_tts or FakeTTS()
        self.playout = Playout(24000)
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.responder = Responder(self.settings, self.chat, self.tts, self.playout, self.pool, 1, fillers)

    def start(self, *, hold: bool = False) -> Responder:
        self.responder.start(MESSAGES, 90, hold=hold)
        return self.responder

    async def pump(self, responder: Responder) -> int:
        """Pull frames like a transport would until the responder finishes and the queue drains."""
        frames = 0
        while not (responder.task.done() and self.playout.idle):
            frames += self.playout.next_frame() is not None
            await asyncio.sleep(0.002)
        return frames


@pytest.fixture
def rig():
    made: list[Rig] = []

    def build(**kwargs) -> Rig:
        made.append(Rig(**kwargs))
        return made[-1]

    yield build
    for r in made:
        r.pool.shutdown(wait=False, cancel_futures=True)


async def test_streams_text_into_audio_and_marks_every_stage(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES))
    responder = r.start()
    frames = await r.pump(responder)
    assert responder.text.startswith("One two three four five.")
    assert frames >= 14 * 5 - 5            # 14 words at 100 ms, 20 ms frames
    for mark in (Mark.LLM_REQUEST, Mark.LLM_FIRST_TOKEN, Mark.FIRST_CHUNK_READY, Mark.TTS_FIRST_AUDIO):
        assert mark in responder.trace.marks
    assert responder.model == "fake/model" and responder.usage["completion_tokens"] == 7
    assert responder.tokens_used == 7


async def test_first_chunk_is_spoken_before_the_llm_has_finished(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES, per_delta=0.03))
    responder = r.start()
    await responder.first_audio.wait()
    assert not responder.task.done()
    assert len(r.tts.texts) == 1
    responder.cancel()


async def test_held_responder_buffers_text_but_stays_silent_until_released(rig):
    r = rig()
    responder = r.start(hold=True)
    await asyncio.sleep(0.15)
    assert responder.text and not responder.first_audio.is_set() and r.tts.texts == []
    assert Mark.FIRST_CHUNK_READY in responder.trace.marks
    responder.release()
    await asyncio.wait_for(responder.first_audio.wait(), 2)
    await r.pump(responder)
    assert r.tts.texts


async def test_adopt_mirrors_past_and_future_marks_and_text(rig):
    from voiceagent.core.metrics import TurnTrace
    r = rig()
    responder = r.start(hold=True)
    await asyncio.sleep(0.15)
    sink, seen = TurnTrace(1), []
    responder.adopt(sink, seen.append)
    assert Mark.LLM_FIRST_TOKEN in sink.marks and seen
    responder.release()
    await r.pump(responder)
    assert Mark.TTS_FIRST_AUDIO in sink.marks


async def test_non_streaming_llm_chunks_only_after_the_whole_reply(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES, per_delta=0.02), llm={"stream": False})
    responder = r.start()
    await asyncio.sleep(0.1)
    assert responder.text == "" and Mark.LLM_FIRST_TOKEN in responder.trace.marks
    await r.pump(responder)
    marks = responder.trace.marks
    assert marks[Mark.FIRST_CHUNK_READY] - marks[Mark.LLM_FIRST_TOKEN] > 0.2


async def test_non_streaming_tts_synthesises_the_whole_reply_in_one_call(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES), tts={"stream": False})
    responder = r.start()
    await r.pump(responder)
    assert len(r.tts.texts) == 1 and r.tts.texts[0].startswith("One two three four five.")


async def test_next_chunk_is_synthesised_while_the_first_is_still_queued(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES), tts={"prefetch_chunks": 1})
    responder = r.start()
    await asyncio.sleep(0.6)   # nothing pulls the playout, so chunk 1 stays unplayed
    assert len(r.tts.texts) == 2
    responder.cancel()


async def test_playout_stays_within_its_bound_while_nobody_plays(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES))
    responder = r.start()
    await asyncio.sleep(0.4)
    assert r.playout.room() == 0
    responder.cancel()


async def test_length_finish_drops_the_cut_off_sentence(rig):
    r = rig(chat=FakeChat(reply="First is done. Second is also done. Third gets cut off here",
                          finish_reason="length"))
    responder = r.start()
    await r.pump(responder)
    assert responder.text.endswith("also done.") and "cut off" not in responder.text


async def test_cancel_stops_everything_promptly(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES, per_delta=0.05))
    responder = r.start()
    await asyncio.wait_for(responder.first_audio.wait(), 2)
    responder.cancel()
    r.playout.clear()
    await asyncio.wait_for(responder.wait(), 0.5)
    assert r.chat.cancels[0].is_set() and r.playout.idle
    await asyncio.sleep(0.2)
    assert r.playout.idle  # nothing leaks in after the cancel


async def test_llm_failure_is_answered_with_a_spoken_apology(rig):
    r = rig(chat=FakeChat(fail=True))
    responder = r.start()
    await r.pump(responder)
    assert responder.text == LLM_FAILURE_REPLY and responder.error is not None
    assert r.tts.texts == [LLM_FAILURE_REPLY]


async def test_tts_failure_is_recorded_and_does_not_hang(rig):
    r = rig(fake_tts=FakeTTS(fail=True))
    responder = r.start()
    await asyncio.wait_for(responder.wait(), 2)
    assert responder.error is not None and not responder.first_audio.is_set()


async def test_filler_plays_when_no_audio_arrives_in_time():
    import numpy as np

    filler = np.full(2400, 0.1, dtype=np.float32)
    r = Rig(chat=FakeChat(delay=0.3), fillers=[filler], tts={"filler_after_ms": 50})
    responder = r.start()
    await asyncio.sleep(0.15)
    assert not r.playout.idle and not responder.first_audio.is_set()
    responder.cancel()
    r.pool.shutdown(wait=False, cancel_futures=True)


async def test_canned_text_is_spoken_without_the_llm(rig):
    r = rig()
    r.responder.start_canned("Sorry about that.")
    await r.pump(r.responder)
    assert r.chat.calls == [] and r.tts.texts == ["Sorry about that."]


async def test_a_closed_synth_gate_holds_new_syntheses_but_queued_audio_still_plays(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES))
    gate = asyncio.Event()
    responder = Responder(r.settings, r.chat, r.tts, r.playout, r.pool, 1, synth_gate=gate)
    responder.start(MESSAGES, 90, hold=False)
    await asyncio.sleep(0.15)
    assert responder.text and r.tts.texts == [] and not responder.first_audio.is_set()
    gate.set()
    await asyncio.wait_for(responder.first_audio.wait(), 2)
    gate.clear()
    started = len(r.tts.texts)
    played = 0
    for _ in range(40):                                  # 40 x 5 ms of transport pulls
        played += r.playout.next_frame() is not None
        await asyncio.sleep(0.005)
    assert played > 0                                    # the audio already made kept playing
    assert len(r.tts.texts) <= started + 1               # at most one that had passed the gate already
    gate.set()
    await r.pump(responder)
    assert len(r.tts.texts) == 3


async def test_aclose_waits_for_every_stage(rig):
    r = rig(chat=FakeChat(reply=FOUR_SENTENCES, per_delta=0.03))
    responder = r.start()
    await asyncio.sleep(0.05)
    await responder.aclose()
    assert responder.task.done() and all(t.done() for t in responder._tasks)
