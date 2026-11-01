"""Co-residency test: ASR and TTS in one process, residency check, then a worst-case turn.

Worst case = about 30 s of speech streamed through the ASR, three 25-word syntheses back to back, then a
second long ASR turn. Reports process peak and whole-GPU memory (which includes the CUDA context and
whatever Windows is drawing).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torchaudio

from voiceagent.asr.nemotron import RATE, NemotronStreamingASR
from voiceagent.config import ROOT, load_settings
from voiceagent.gpu import (
    ClockSampler,
    apply_cpu_threads,
    assert_resident,
    device_memory_mib,
    process_memory_mib,
)
from voiceagent.tts.kokoro import KokoroSynthesizer

LONG_TEXTS = [
    ("The moon is about three hundred eighty four thousand kilometres from Earth, and light needs just over a "
     "second to cross that distance, which is why astronauts hear a short delay."),
    ("If you only have rice, lentils and onions, a simple khichdi with fried onions is quick, filling and "
     "surprisingly tasty, and you can add a little ghee if you like."),
    ("Resetting a password without email access usually needs a recovery phone number, a backup code or help "
     "from the support team, so keep those details safe in advance."),
]


def stream_turn(asr: NemotronStreamingASR, audio: np.ndarray) -> float:
    asr.reset()
    step = RATE * 160 // 1000
    start = time.perf_counter()
    for i in range(0, len(audio), step):
        asr.push(audio[i:i + step])
    asr.finish()
    return time.perf_counter() - start


def main() -> None:
    settings = load_settings()
    apply_cpu_threads(settings.devices.cpu_threads)
    base = device_memory_mib()
    print(f"before loading: whole-GPU used {base['used']:.0f} of {base['total']:.0f} MiB")
    tts = KokoroSynthesizer(settings.tts, settings.devices.tts)
    asr = NemotronStreamingASR(settings.asr, settings.devices.asr)
    n_tts = assert_resident("tts", tts.torch_modules(), settings.devices.tts)
    n_asr = assert_resident("asr", asr.torch_modules(), settings.devices.asr)
    print(f"residency ok: tts {n_tts} tensors, asr {n_asr} tensors")
    tts.warmup()
    asr.warmup()
    loaded = device_memory_mib()
    print(f"after load + warm-up: whole-GPU used {loaded['used']:.0f} MiB, process {process_memory_mib()}")

    sentences = [ln for ln in (ROOT / "evals" / "sets" / "read_aloud.txt").read_text().splitlines() if ln.strip()]
    wav = np.concatenate([np.concatenate(list(tts.synthesize(s))) for s in sentences[:9]])
    long_audio = torchaudio.functional.resample(torch.from_numpy(wav), tts.sample_rate, RATE).numpy()
    print(f"worst-case ASR input: {len(long_audio) / RATE:.0f} s of speech")

    torch.cuda.reset_peak_memory_stats()
    peak_used = loaded["used"]
    with ClockSampler() as clocks:
        t1 = stream_turn(asr, long_audio)
        peak_used = max(peak_used, device_memory_mib()["used"])
        for text in LONG_TEXTS:
            audio_s = sum(len(c) for c in tts.synthesize(text)) / tts.sample_rate
            peak_used = max(peak_used, device_memory_mib()["used"])
            print(f"  25-word synthesis -> {audio_s:.1f} s of audio")
        t2 = stream_turn(asr, long_audio)
        peak_used = max(peak_used, device_memory_mib()["used"])
    assert_resident("tts", tts.torch_modules(), settings.devices.tts)
    assert_resident("asr", asr.torch_modules(), settings.devices.asr)
    mem = process_memory_mib()
    print(f"\nlong ASR turn: {t1:.1f} s and {t2:.1f} s (RTF {t1 / (len(long_audio) / RATE):.2f}, "
          f"{t2 / (len(long_audio) / RATE):.2f}); residency still ok afterwards")
    print(f"process peak allocated {mem['peak_allocated']:.0f} MiB, peak reserved {mem['peak_reserved']:.0f} MiB")
    print(f"whole-GPU peak used {peak_used:.0f} MiB ({peak_used / 1024:.2f} GiB) of {base['total']:.0f} MiB; "
          f"budget is 3.6 GB = {3.6 * 1024:.0f} MiB")
    print(clocks.summary())
    asr.close()


if __name__ == "__main__":
    main()
