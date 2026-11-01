"""Kokoro time to first audio, real-time factor and VRAM for chunks of 4, 8 and 25 words."""

from __future__ import annotations

import argparse
import time

import torch

from voiceagent.config import load_settings
from voiceagent.core.metrics import percentile
from voiceagent.gpu import apply_cpu_threads, assert_resident, device_memory_mib, process_memory_mib
from voiceagent.tts.kokoro import KokoroSynthesizer

BASE_SENTENCES = [
    "The weather in Roorkee is pleasant this evening, so you might enjoy a short walk after dinner with your family.",
    "To book a train ticket you can open the railway app, choose your stations, pick a date and pay online in minutes.",
    "A process has its own memory space while threads inside it share memory, which makes threads lighter but riskier.",
    "The moon is about three hundred eighty four thousand kilometres from Earth, and light needs just over a second to cross.",
    "If you only have rice, lentils and onions, a simple khichdi with fried onions is quick, filling and surprisingly tasty.",
    "Staying focused while studying is easier when you silence your phone, work in short blocks and take real breaks often.",
    "Supply and demand means prices rise when many people want something scarce and fall when sellers have too much of it.",
    "Canberra is the capital of Australia because it was chosen as a compromise between the rival cities of Sydney and Melbourne.",
    "Resetting a password without email access usually needs a recovery phone number, a backup code or help from the support team.",
    "The sky looks blue because air scatters short blue wavelengths more strongly, and red at sunset because light travels farther.",
]


def make_texts(words: int, n: int) -> list[str]:
    texts = []
    for i in range(n):
        parts = BASE_SENTENCES[i % len(BASE_SENTENCES)].split()
        start = (i // len(BASE_SENTENCES)) % 3
        texts.append(" ".join(parts[start:start + words]).rstrip(",.") + ".")
    return texts


def time_one(tts: KokoroSynthesizer, text: str) -> tuple[float, float, float]:
    """Return (seconds to first audio, total synthesis seconds, audio seconds)."""
    start = time.perf_counter()
    first = None
    samples = 0
    for chunk in tts.synthesize(text):
        if first is None:
            torch.cuda.synchronize()
            first = time.perf_counter() - start
        samples += len(chunk)
    return first or 0.0, time.perf_counter() - start, samples / tts.sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config) if args.config else load_settings()
    apply_cpu_threads(settings.devices.cpu_threads)
    device = settings.devices.tts
    before = device_memory_mib()
    tts = KokoroSynthesizer(settings.tts, device)
    checked = assert_resident("tts", tts.torch_modules(), device)
    print(f"residency ok ({checked} tensors on {device}); smi used before load {before['used']:.0f} MiB")
    tts.warmup()
    for text in make_texts(8, 3):  # extra warm-up per length bucket so kernel selection is settled
        time_one(tts, text)
    torch.cuda.reset_peak_memory_stats(device)
    print(f"n = {args.n} per size, warm-up excluded. Times in ms. (Measured, this machine)\n")
    print("| words | TTFA p50 | TTFA p90 | TTFA max | RTF (synth/audio) p50 | audio s p50 |")
    print("|---|---|---|---|---|---|")
    for words in (4, 8, 25):
        runs = [time_one(tts, t) for t in make_texts(words, args.n)]
        ttfa = [r[0] * 1000 for r in runs]
        rtf = [r[1] / r[2] for r in runs]
        dur = [r[2] for r in runs]
        print(f"| {words} | {percentile(ttfa, 50):.0f} | {percentile(ttfa, 90):.0f} | {max(ttfa):.0f} | "
              f"{percentile(rtf, 50):.3f} | {percentile(dur, 50):.1f} |")
    mem, smi = process_memory_mib(device), device_memory_mib()
    print(f"\nprocess peak allocated {mem['peak_allocated']:.0f} MiB, peak reserved {mem['peak_reserved']:.0f} MiB; "
          f"whole-GPU used now {smi['used']:.0f} of {smi['total']:.0f} MiB (incl. CUDA context)")


if __name__ == "__main__":
    main()
