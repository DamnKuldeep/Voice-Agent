from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voiceagent.audio.frames import resample_to_16k

pytestmark = pytest.mark.integration
AUDIO = Path(__file__).resolve().parents[1] / "evals" / "audio"


def load(name: str) -> np.ndarray:
    data, rate = sf.read(AUDIO / name, dtype="float32")
    return resample_to_16k(data, rate)


def test_silero_separates_speech_from_silence():
    from voiceagent.audio.silero import SileroVAD

    vad = SileroVAD()
    speech = load("libri_000.wav")

    def probs(audio):
        vad.reset()
        return [vad.process(audio[i : i + 512]) for i in range(0, len(audio) - 511, 512)]

    assert max(probs(speech)) > 0.9
    noise = (np.random.default_rng(0).normal(0, 0.003, 32000)).astype(np.float32)
    assert max(probs(noise)) < 0.3
    with pytest.raises(ValueError):
        vad.process(np.zeros(100, dtype=np.float32))


def test_smart_turn_complete_vs_cut_off():
    from voiceagent.audio.smart_turn import SmartTurnDetector

    det = SmartTurnDetector()
    audio = load("libri_000.wav")
    complete = det.end_of_turn_probability(audio)
    cut = det.end_of_turn_probability(audio[: int(len(audio) * 0.6)])
    assert 0.0 <= cut <= 1.0 and complete > cut
