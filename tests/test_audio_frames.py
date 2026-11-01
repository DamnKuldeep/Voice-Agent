import numpy as np
import pytest

from voiceagent.audio.frames import FrameCutter, PreRollBuffer, resample_to_16k


def tone(rate: int, seconds: float = 0.5, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.mark.parametrize("rate", [48000, 24000, 16000, 44100])
def test_resample_length_and_dtype(rate):
    out = resample_to_16k(tone(rate), rate)
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - 8000) <= 2
    assert np.abs(out).max() <= 1.0


def test_resample_preserves_tone_frequency():
    out = resample_to_16k(tone(48000, 1.0), 48000)
    peak = np.argmax(np.abs(np.fft.rfft(out))) * 16000 / len(out)
    assert abs(peak - 440) < 2


def test_int16_scaled_to_unit_range():
    out = resample_to_16k(np.array([16384, -32768, 0], dtype=np.int16), 16000)
    assert np.allclose(out, [0.5, -1.0, 0.0])


@pytest.mark.parametrize("shape", [(1, 480), (480, 1), (2, 480), (480, 2)])
def test_channel_layouts_become_mono(shape):
    out = resample_to_16k(np.full(shape, 0.25, dtype=np.float32), 16000)
    assert out.shape == (480,) and np.allclose(out, 0.25)


def test_empty_input():
    assert len(resample_to_16k(np.zeros(0, dtype=np.int16), 48000)) == 0


def test_frame_cutter_keeps_remainder():
    cutter = FrameCutter(512)
    first = cutter.push(np.arange(700, dtype=np.float32))
    assert len(first) == 1 and len(first[0]) == 512
    second = cutter.push(np.arange(700, 1100, dtype=np.float32))
    assert len(second) == 1
    assert np.array_equal(second[0], np.arange(512, 1024, dtype=np.float32))
    assert cutter.push(np.zeros(0, dtype=np.float32)) == []


def test_pre_roll_keeps_only_recent_audio_oldest_first():
    buf = PreRollBuffer(ms=100, rate=16000)  # 1600 samples
    for i in range(10):
        buf.push(np.full(512, i, dtype=np.float32))
    out = buf.drain()
    assert len(out) == 1600
    assert out[-1] == 9 and out[0] == 6  # oldest first, trimmed mid-frame to the window
    assert len(buf.drain()) == 0


def test_pre_roll_drain_when_underfilled():
    buf = PreRollBuffer(ms=300)
    buf.push(np.ones(512, dtype=np.float32))
    assert len(buf.drain()) == 512
