import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pipeline_fakes import FakeASR
from voiceagent.pipeline.asr_stage import AsrStage

CHUNK = 2560  # 160 ms at 16 kHz


@pytest.fixture
async def stage():
    pool = ThreadPoolExecutor(max_workers=1)
    asr = FakeASR(["one two three four five six"])
    partials: list[str] = []
    errors: list[Exception] = []
    s = AsrStage(asr, pool, 160, partials.append, errors.append)
    s.start()
    yield s, asr, partials, errors
    await s.stop()
    pool.shutdown()


def audio(samples: int) -> np.ndarray:
    return np.full(samples, 0.5, dtype=np.float32)


async def test_audio_is_pushed_in_whole_chunks_and_the_remainder_on_finish(stage):
    s, asr, partials, _ = stage
    s.begin_turn()
    s.feed(audio(1000))
    s.feed(audio(CHUNK + 500))
    s.feed(audio(CHUNK))
    final = await s.finish()
    assert asr.pushes == [CHUNK, CHUNK, 1000 + 500]
    assert final == "one two three four five six" and partials


async def test_partials_from_before_a_new_turn_are_discarded(stage):
    s, _asr, partials, _ = stage
    s.begin_turn()
    s.feed(audio(CHUNK * 2))
    s.begin_turn()                      # queued behind the pushes: their partials are stale
    await s.finish()
    assert partials == []


async def test_a_failing_finish_raises_to_the_caller_and_the_worker_survives(stage):
    s, asr, _, _ = stage
    s.begin_turn()
    s.feed(audio(CHUNK))
    asr.fail_finish = True
    with pytest.raises(RuntimeError):
        await s.finish()
    asr.fail_finish = False
    s.begin_turn()
    s.feed(audio(CHUNK))
    assert await asyncio.wait_for(s.finish(), 2) != ""


async def test_a_full_queue_drops_chunks_instead_of_growing():
    pool = ThreadPoolExecutor(max_workers=1)
    s = AsrStage(FakeASR(["x"]), pool, 160, lambda _t: None, lambda _e: None)  # worker not started
    for _ in range(80):
        s.feed(audio(CHUNK))
    assert s.dropped_chunks == 80 - 64
    pool.shutdown()


async def test_reset_purges_queued_audio_so_the_model_never_decodes_it(stage):
    s, asr, partials, _ = stage
    asr.push_delay_s = 0.05
    s.begin_turn()
    s.feed(audio(CHUNK * 6))
    await asyncio.sleep(0.02)            # the first chunk is in the recogniser, five are queued
    s.begin_turn()
    assert s.backlog_ms == 0
    s.feed(audio(CHUNK))
    await s.finish()
    assert len(asr.pushes) == 2 and s.skipped_chunks == 5
    assert len(partials) == 1            # the in-flight chunk's partial was stale and dropped


async def test_a_finish_made_stale_by_a_reset_resolves_empty(stage):
    s, asr, _, _ = stage
    asr.push_delay_s = 0.05
    s.begin_turn()
    s.feed(audio(CHUNK * 3))
    pending = asyncio.ensure_future(s.finish())
    await asyncio.sleep(0.01)
    s.begin_turn()
    assert await asyncio.wait_for(pending, 1) == ""


async def test_backlog_is_reported_while_behind_and_once_more_on_recovery():
    pool = ThreadPoolExecutor(max_workers=1)
    asr = FakeASR(["one two three"])
    asr.push_delay_s = 0.03
    lags: list[float] = []
    s = AsrStage(asr, pool, 160, lambda _t: None, lambda _e: None, lags.append)
    s.start()
    s.begin_turn()
    s.feed(audio(CHUNK * 4))             # 640 ms of audio handed over at once
    assert lags == [320.0]               # reported on crossing 300 ms, then at most twice a second
    await s.finish()
    assert lags[-1] == 0.0 and len(lags) == 2
    assert s.take_max_lag_ms() == 640.0 and s.take_max_lag_ms() == 0.0
    await s.stop()
    pool.shutdown()


async def test_a_small_backlog_is_not_reported():
    pool = ThreadPoolExecutor(max_workers=1)
    lags: list[float] = []
    s = AsrStage(FakeASR(["x"]), pool, 160, lambda _t: None, lambda _e: None, lags.append)
    s.start()
    s.begin_turn()
    s.feed(audio(CHUNK))
    await s.finish()
    assert lags == []
    await s.stop()
    pool.shutdown()


@pytest.mark.parametrize("chunk_ms", [80, 160, 560])
async def test_chunk_size_follows_the_setting(chunk_ms):
    pool = ThreadPoolExecutor(max_workers=1)
    asr = FakeASR(["x"])
    s = AsrStage(asr, pool, chunk_ms, lambda _t: None, lambda _e: None)
    s.start()
    s.begin_turn()
    samples = chunk_ms * 16
    s.feed(audio(samples * 2 + 100))
    await s.finish()
    assert asr.pushes == [samples, samples, 100]
    await s.stop()
    pool.shutdown()
