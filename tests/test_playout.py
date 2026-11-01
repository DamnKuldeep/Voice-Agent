import numpy as np
import pytest

from voiceagent.pipeline.playout import GAIN_RAMP_MS, MAX_AHEAD_MS, Playout

RATE = 24000
FRAME = RATE * 20 // 1000


def pcm(ms: int, value: float = 0.3) -> np.ndarray:
    return np.full(RATE * ms // 1000, value, dtype=np.float32)


def test_audio_is_cut_into_20ms_frames():
    playout = Playout(RATE)
    playout.write(pcm(100))
    frames = []
    while (frame := playout.next_frame()) is not None:
        frames.append(frame)
    assert len(frames) == 5 and all(len(f) == FRAME for f in frames)


def test_queue_is_bounded_to_about_200ms():
    playout = Playout(RATE)
    taken = playout.write(pcm(1000))
    assert taken == RATE * MAX_AHEAD_MS // 1000
    assert playout.room() == 0
    assert playout.write(pcm(100)) == 0
    playout.next_frame()
    assert playout.room() == FRAME


def test_clear_is_instant_and_leaves_the_queue_empty():
    playout = Playout(RATE)
    playout.write(pcm(190))
    playout.clear()
    assert playout.idle and playout.next_frame() is None and playout.room() > 0


def test_partial_frame_is_held_until_finish_pads_it():
    playout = Playout(RATE)
    playout.write(pcm(30))
    assert playout.next_frame() is not None and playout.next_frame() is None
    assert not playout.idle
    playout.finish()
    last = playout.next_frame()
    assert last is not None and last[-1] == 0.0 and playout.idle


def test_spoken_text_follows_the_samples_handed_out():
    playout = Playout(RATE)
    playout.begin_reply()
    playout.begin_chunk("one two three four")
    playout.write(pcm(100))
    playout.end_chunk()
    playout.begin_chunk("five six")
    for _ in range(3):  # 60 ms of the first chunk
        playout.next_frame()
    assert playout.spoken_text() == "one two"
    for _ in range(2):
        playout.next_frame()
    assert playout.spoken_text() == "one two three four"


def test_begin_reply_resets_the_accounting():
    playout = Playout(RATE)
    playout.begin_chunk("hello there")
    playout.write(pcm(40))
    playout.end_chunk()
    playout.next_frame()
    playout.next_frame()
    assert playout.spoken_text() == "hello there"
    playout.begin_reply()
    assert playout.spoken_text() == ""


def drain(playout: Playout) -> np.ndarray:
    frames = []
    while (frame := playout.next_frame()) is not None:
        frames.append(frame)
    return np.concatenate(frames)


def test_gain_ramps_smoothly_to_the_duck_level_and_back():
    playout = Playout(RATE)
    playout.write(pcm(200, value=1.0))
    playout.set_gain(0.25)
    out = drain(playout)
    ramp = RATE * GAIN_RAMP_MS // 1000
    assert np.all(np.diff(out) <= 1e-6)                   # never steps up while ducking
    assert np.max(np.abs(np.diff(out))) < 0.01            # no click: tiny per-sample change
    assert out[ramp:] == pytest.approx(0.25, abs=1e-4)    # reached within the ramp
    assert out[0] > 0.99
    playout.write(pcm(100, value=1.0))
    playout.set_gain(1.0)
    back = drain(playout)
    assert np.all(np.diff(back) >= -1e-6) and back[-1] == pytest.approx(1.0)


def test_a_new_reply_starts_at_full_volume():
    playout = Playout(RATE)
    playout.set_gain(0.25)
    playout.write(pcm(100, value=1.0))
    drain(playout)
    playout.begin_reply()
    playout.write(pcm(40, value=1.0))
    assert drain(playout) == pytest.approx(1.0)


def test_recent_text_covers_only_the_window_already_played():
    playout = Playout(RATE)
    playout.begin_reply()
    playout.begin_chunk("one two three four")
    playout.write(pcm(200))
    playout.end_chunk()
    for _ in range(5):                                    # 100 ms of 200: words one and two
        playout.next_frame()
    assert playout.recent_text(3000) == "one two"
    assert playout.recent_text(40) == "two"
