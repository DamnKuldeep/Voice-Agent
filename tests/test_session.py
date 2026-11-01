import asyncio
import threading

import numpy as np
import pytest

from pipeline_fakes import (
    LONG_REPLY,
    Driver,
    FakeChat,
    FakeTTS,
    make_components,
    make_settings,
    silence,
    speech,
)
from voiceagent.core.events import Flag, Mark
from voiceagent.pipeline.history import INTERRUPTED_MARK
from voiceagent.pipeline.reply_flow import ASR_FAILURE_REPLY
from voiceagent.pipeline.session import ConversationSession

QUESTION = "what is the capital of france"


@pytest.fixture
async def make():
    opened: list[tuple[ConversationSession, object]] = []

    async def build(settings=None, **component_args):
        components = make_components(**component_args)
        session = ConversationSession(settings or make_settings(), components, trace_dir=None)
        await session.start()
        opened.append((session, components))
        return session, components, Driver(session)

    yield build
    for session, components in opened:
        await session.stop()
        components.close()


def reply_done(session: ConversationSession, count: int = 1):
    return lambda: len(session.traces) >= count and session.state == "idle"


async def test_full_turn_walks_the_state_machine_and_reports_events(make):
    session, comps, d = await make()
    assert session.state == "idle"
    await d.feed(speech(1.2))
    await d.until(reply_done(session))

    assert d.states() == ["listening", "thinking", "speaking", "idle"]
    assert d.of_type("partial") and d.of_type("final")[0]["text"] == QUESTION
    replies = d.of_type("reply")
    assert replies[-1]["final"] is True and replies[-1]["text"] == "Paris is the capital of France."
    turn = d.of_type("turn")[0]
    assert turn["transcript"] == QUESTION and turn["flags"][Flag.LLM_MODEL.value] == "fake/model"
    assert {"endpoint_ms", "asr_final_ms", "llm_ttft_ms", "tts_ttfa_ms", "v2v_ms"} <= set(turn["metrics"])
    assert d.of_type("level")
    assert sum(len(f) for f in d.output) >= 24000 * 0.5
    assert session.summary()["v2v_ms"]["n"] == 1.0
    assert comps.chat.calls[0][-1] == {"role": "user", "content": QUESTION}


async def test_history_reaches_the_next_request(make):
    session, comps, d = await make(transcripts=[QUESTION, "and germany"])
    await d.feed(speech(1.2))
    await d.until(reply_done(session))
    await d.feed(speech(0.6))
    await d.until(reply_done(session, 2))
    roles = [m["role"] for m in comps.chat.calls[-1]]
    assert roles[-3:] == ["user", "assistant", "user"]
    assert comps.chat.calls[-1][-3]["content"] == QUESTION


async def test_noise_shorter_than_min_speech_starts_nothing(make):
    session, comps, d = await make()
    await d.feed(speech(0.1))
    await d.feed(silence(1.0))
    assert {e["type"] for _, e in d.events} <= {"level"}
    assert session.state == "idle" and comps.chat.calls == [] and not session.traces


async def test_empty_transcript_is_dropped_without_a_reply(make):
    session, comps, d = await make(transcripts=[""], finals=[""])
    await d.feed(speech(0.6))
    await d.until(lambda: "thinking" in d.states() and session.state == "idle")
    assert comps.chat.calls == [] and not session.traces and d.output == []


@pytest.mark.parametrize(("prob", "text", "low", "high"), [
    (0.9, QUESTION, 100, 450),
    (0.1, QUESTION, 850, 1300),
    (0.9, "i want to order a pizza and", 850, 1300),
])
async def test_smart_turn_probability_decides_how_long_to_wait(make, prob, text, low, high):
    session, _comps, d = await make(transcripts=[text], turn_prob=prob, chat=FakeChat(delay=1.0))
    await d.feed(speech(1.3))
    spoke_until = d.fed_ms
    await d.until(lambda: session.state == "thinking")
    waited = d.at("thinking") - spoke_until
    assert low <= waited <= high, waited


async def test_without_a_turn_model_a_fixed_silence_is_used(make):
    session, _comps, d = await make(chat=FakeChat(delay=1.0))
    await d.feed(speech(1.2))
    spoke_until = d.fed_ms
    await d.until(lambda: session.state == "thinking")
    assert 450 <= d.at("thinking") - spoke_until <= 750


async def test_pause_shorter_than_the_commit_delay_continues_the_same_turn(make):
    session, _comps, d = await make(transcripts=["i would like a pizza"])
    await d.feed(speech(0.8))
    await d.feed(silence(0.25))
    await d.feed(speech(0.4))
    await d.until(reply_done(session))
    assert d.states().count("thinking") == 1 and len(session.traces) == 1


async def test_speculation_hit_reuses_the_early_request(make):
    session, comps, d = await make()
    await d.feed(speech(1.2))
    await d.until(reply_done(session))
    trace = session.traces[0]
    assert len(comps.chat.calls) == 1
    assert trace.flags[Flag.SPECULATION_USED] is True and Flag.SPECULATION_WASTED not in trace.flags
    assert trace.marks[Mark.LLM_REQUEST] < trace.marks[Mark.COMMIT]


async def test_speculation_miss_cancels_and_asks_again(make):
    session, comps, d = await make(finals=["what is the capital of germany"])
    await d.feed(speech(1.2))
    await d.until(reply_done(session))
    assert len(comps.chat.calls) == 2 and comps.chat.cancels[0].is_set()
    assert session.traces[0].flags[Flag.SPECULATION_WASTED] is True
    assert comps.chat.calls[1][-1]["content"] == "what is the capital of germany"


async def test_speculation_is_disabled_by_config(make):
    session, _comps, d = await make(settings=make_settings(speculation={"enabled": False}))
    await d.feed(speech(1.2))
    await d.until(reply_done(session))
    trace = session.traces[0]
    assert trace.marks[Mark.LLM_REQUEST] > trace.marks[Mark.COMMIT]


async def test_user_resuming_cancels_the_speculative_request(make):
    session, comps, d = await make()
    await d.feed(speech(1.0))
    await d.feed(silence(0.3))
    assert comps.chat.calls, "speculation should have started"
    await d.feed(speech(0.8))
    assert comps.chat.cancels[0].is_set()
    await d.until(reply_done(session))
    assert session.traces[0].flags[Flag.SPECULATION_WASTED] is True


async def test_resuming_after_the_commit_drops_the_reply_and_redecodes_the_whole_turn(make):
    chat = FakeChat(delay=1.0)
    session, comps, d = await make(
        transcripts=["what is the capital", "what is the capital of france please"], chat=chat,
        settings=make_settings(speculation={"enabled": False}))
    await d.feed(speech(1.0))
    await d.until(lambda: session.state == "thinking")
    resets_before = comps.asr.resets
    await d.feed(speech(0.8))
    assert session.state == "listening" and chat.cancels[0].is_set()
    await d.until(reply_done(session), 20)
    # One recogniser pass over both parts: no text carried across the pause.
    assert chat.calls[-1][-1]["content"] == "what is the capital of france please"
    assert comps.asr.resets > resets_before
    assert sum(comps.asr.pushes) >= 16000 * 2.6  # the first second of speech was decoded twice
    assert len(session.traces) == 1


async def test_long_speech_over_the_reply_interrupts_it(make):
    chat = FakeChat(reply=LONG_REPLY)
    session, _comps, d = await make(transcripts=[QUESTION, "no tell me about spain"], chat=chat)
    await d.feed(speech(1.2))
    await d.until(lambda: session.state == "speaking")
    await d.feed(silence(0.6))
    played_before = len(d.output)
    await d.feed(speech(1.0))
    assert session.state == "listening"
    assert session.next_playout_frame() is None
    stopped_at = len(d.output)
    await d.feed(speech(0.2))
    assert len(d.output) == stopped_at and stopped_at > played_before

    await d.until(reply_done(session, 2), 20)
    first = session.traces[0]
    assert first.flags[Flag.INTERRUPTED] is True and "barge_in_stop_ms" in first.derived()
    heard = d.of_type("turn")[0]["reply"]
    assert 0 < len(heard) < len(LONG_REPLY)
    history = chat.calls[1]
    assistant = next(m for m in history if m["role"] == "assistant")
    assert assistant["content"] == f"{heard} {INTERRUPTED_MARK}"
    assert history[-1]["content"] == "no tell me about spain"


async def test_backchannel_does_not_interrupt(make):
    chat = FakeChat(reply=LONG_REPLY)
    session, comps, d = await make(transcripts=[QUESTION, "yeah"], chat=chat)
    await d.feed(speech(1.2))
    await d.until(lambda: session.state == "speaking")
    await d.feed(silence(0.3))
    await d.feed(speech(0.45))
    await d.feed(silence(0.5))
    assert session.state == "speaking"
    await d.until(reply_done(session), 20)
    assert len(session.traces) == 1 and Flag.INTERRUPTED not in session.traces[0].flags
    assert len(d.output) >= 140   # the whole 30-word reply was still played
    assert comps.asr.transcripts == []


async def test_barge_in_can_be_switched_off(make):
    chat = FakeChat(reply=LONG_REPLY)
    session, _comps, d = await make(settings=make_settings(barge_in={"enabled": False}), chat=chat)
    await d.feed(speech(1.2))
    await d.until(lambda: session.state == "speaking")
    await d.feed(speech(1.0))
    assert session.state == "speaking"
    await d.until(reply_done(session), 20)
    assert Flag.INTERRUPTED not in session.traces[0].flags


async def test_asr_failure_is_apologised_for_aloud(make):
    session, comps, d = await make()
    comps.asr.fail_finish = True
    await d.feed(speech(1.0))
    await d.until(lambda: d.states().count("speaking") == 1 and session.state == "idle")
    assert comps.tts.texts == [ASR_FAILURE_REPLY]
    assert all(c.is_set() for c in comps.chat.cancels)  # any speculative request was cancelled
    assert not session.traces and d.output


async def test_llm_failure_falls_back_to_a_spoken_line_and_the_session_survives(make):
    chat = FakeChat(fail=True)
    session, _comps, d = await make(transcripts=[QUESTION, QUESTION], chat=chat)
    await d.feed(speech(1.2))
    await d.until(reply_done(session))
    assert "trouble" in d.of_type("turn")[0]["reply"]
    chat.fail = False
    await d.feed(speech(1.2))
    await d.until(reply_done(session, 2))
    assert d.of_type("turn")[1]["reply"] == "Paris is the capital of France."


async def test_tts_failure_does_not_crash_the_session(make):
    session, _comps, d = await make(tts=FakeTTS(fail=True))
    await d.feed(speech(1.2))
    await d.until(lambda: "thinking" in d.states() and session.state == "idle")
    assert session.state == "idle"


async def test_feed_mic_accepts_other_rates_and_dtypes_from_another_thread(make):
    session, _comps, d = await make()
    pcm48 = (np.full(960 * 60, 0.5) * 32767).astype(np.int16)

    def push() -> None:
        for start in range(0, len(pcm48), 960):
            session.feed_mic(pcm48[start:start + 960], 48000)

    worker = threading.Thread(target=push)
    worker.start()
    worker.join()
    await d.feed(silence(0.0))
    await asyncio.sleep(0.3)
    assert "listening" in [e["state"] for e in d.of_type("state")]


async def test_next_playout_frame_is_none_when_nothing_is_queued(make):
    session, _, _ = await make()
    assert session.next_playout_frame() is None


async def test_reset_conversation_forgets_history(make):
    session, comps, d = await make(transcripts=[QUESTION, QUESTION])
    await d.feed(speech(1.2))
    await d.until(reply_done(session))
    session.reset_conversation()
    await d.feed(speech(1.2))
    await d.until(reply_done(session, 2))
    assert [m["role"] for m in comps.chat.calls[-1]][-1:] == ["user"]
    assert all(m["content"] != QUESTION for m in comps.chat.calls[-1][:-1])

