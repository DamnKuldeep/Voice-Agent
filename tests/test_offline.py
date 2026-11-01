import time

import numpy as np

from voiceagent.transport.offline import Cue, Echo, replay


class RecordingSession:
    def __init__(self) -> None:
        self.fed: list[np.ndarray] = []
        self.polls = 0

    def feed_mic(self, samples: np.ndarray, rate: int) -> None:
        self.fed.append(samples)

    def next_playout_frame(self):
        self.polls += 1
        return np.full(480, 0.1, dtype=np.float32) if self.polls % 2 else None


class SpeakingSession(RecordingSession):
    """Plays a constant 0.5 tone (20 ms frames at 24 kHz) from poll ``start`` to poll ``stop``."""

    def __init__(self, start: int = 10, stop: int = 30) -> None:
        super().__init__()
        self.start, self.stop = start, stop

    def next_playout_frame(self):
        self.polls += 1
        return np.full(480, 0.5, dtype=np.float32) if self.start <= self.polls < self.stop else None


async def test_feeds_in_20ms_steps_polls_each_step_and_collects_output():
    session = RecordingSession()
    out = await replay(session, np.zeros(16000, dtype=np.float32), 16000, tail_seconds=1.0, realtime=False)
    assert len(session.fed) == 100 and all(len(c) == 320 for c in session.fed)
    assert session.polls == 100 and len(out) == 50


async def test_injected_clip_is_mixed_in_at_its_offset():
    session = RecordingSession()
    clip = np.full(1600, 0.25, dtype=np.float32)
    base = np.full(32000, 0.5, dtype=np.float32)
    await replay(session, base, 16000, tail_seconds=0.0, realtime=False, inject=[(0.5, clip)])
    mixed = np.concatenate(session.fed)
    assert mixed[7999] == 0.5 and np.isclose(mixed[8000], 0.75) and np.isclose(mixed[9599], 0.75)
    assert mixed[9600] == 0.5


async def test_injection_past_the_end_extends_the_stream():
    session = RecordingSession()
    clip = np.full(1600, 0.25, dtype=np.float32)
    await replay(session, np.zeros(1600, dtype=np.float32), 16000, tail_seconds=0.0, realtime=False,
                 inject=[(1.0, clip)])
    assert sum(len(c) for c in session.fed) == 16000 + 1600


async def test_cue_is_mixed_in_a_delay_after_its_condition_first_holds():
    session = SpeakingSession(start=10)
    cue = Cue(lambda: session.polls >= 10, np.full(640, 0.25, dtype=np.float32), delay_s=0.1)
    await replay(session, np.zeros(320, dtype=np.float32), 16000, tail_seconds=1.0, realtime=False,
                 cues=[cue])
    mixed = np.concatenate(session.fed)
    # condition seen before step 10 is fed (polls == 10 after it), so due at step 10 + 5
    assert np.isclose(cue.fired_audio_s, 0.3) and cue.fired_at is not None
    assert mixed[15 * 320 - 1] == 0.0 and np.isclose(mixed[15 * 320], 0.25)
    assert np.isclose(mixed[17 * 320 - 1], 0.25) and mixed[17 * 320] == 0.0


async def test_a_cue_that_never_fires_leaves_only_the_tail():
    session = RecordingSession()
    cue = Cue(lambda: False, np.ones(320, dtype=np.float32))
    await replay(session, np.zeros(320, dtype=np.float32), 16000, tail_seconds=0.2, realtime=False,
                 cues=[cue])
    assert cue.fired_at is None and len(session.fed) == 11


async def test_echo_feeds_the_output_back_attenuated_and_delayed():
    session = SpeakingSession(start=3, stop=6)  # three output frames, pulled at steps 2, 3, 4
    await replay(session, np.zeros(320, dtype=np.float32), 16000, tail_seconds=0.4, realtime=False,
                 echo=Echo(gain_db=-20.0, delay_ms=60.0))
    mixed = np.concatenate(session.fed)
    first = 2 * 320 + 3 * 320  # played from step 2's start, heard 60 ms (3 steps) later
    assert np.all(mixed[:first] == 0.0)
    assert np.allclose(mixed[first:first + 3 * 320], 0.05, atol=1e-6)
    assert np.all(mixed[first + 3 * 320:] == 0.0)


async def test_until_ends_the_tail_early_and_is_only_asked_during_silence():
    session = RecordingSession()
    asked: list[int] = []

    def until() -> bool:
        asked.append(len(session.fed))
        return len(asked) == 3

    await replay(session, np.zeros(1600, dtype=np.float32), 16000, tail_seconds=5.0, realtime=False,
                 until=until)
    assert asked == [5, 6, 7] and len(session.fed) == 7


async def test_speed_scales_the_realtime_pacing():
    session = RecordingSession()
    started = time.perf_counter()
    await replay(session, np.zeros(16000, dtype=np.float32), 16000, tail_seconds=0.0, speed=10.0)
    assert 0.08 <= time.perf_counter() - started < 0.5 and len(session.fed) == 50
