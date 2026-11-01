"""Streaming ASR benchmark: real-time factor, final-transcript delay, WER and VRAM per chunk size.

Input is a manifest of recordings with reference text: your own (evals/sets/utterances.yaml) or public
ones (evals/sets/public_utterances.yaml, see scripts/fetch_eval_audio.py). With --synthetic it speaks the
read-aloud sentences with Kokoro instead; that audio is too clean for WER, so WER is skipped.
Audio is fed as fast as the model accepts it: RTF = compute time / audio duration.
"""

from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from voiceagent.asr.nemotron import RATE, NemotronStreamingASR
from voiceagent.config import ROOT, load_settings
from voiceagent.core.metrics import percentile
from voiceagent.core.text_metrics import wer
from voiceagent.gpu import (
    ClockSampler,
    apply_cpu_threads,
    assert_resident,
    device_memory_mib,
    process_memory_mib,
)

SENTENCES = ROOT / "evals" / "sets" / "read_aloud.txt"

Clip = tuple[str, np.ndarray, str]  # reference text, 16 kHz audio, source label


def load_manifest(name: str) -> list[Clip]:
    items = yaml.safe_load((ROOT / "evals" / "sets" / name).read_text(encoding="utf-8"))["utterances"]
    out = []
    for item in items:
        audio, rate = sf.read(ROOT / item["file"], dtype="float32")
        assert rate == RATE, f"{item['file']} is {rate} Hz"
        out.append((item["text"], audio, item.get("source", "own_voice")))
    return out


def synthesize(settings) -> list[Clip]:
    from voiceagent.tts.kokoro import KokoroSynthesizer

    tts = KokoroSynthesizer(settings.tts, settings.devices.tts)
    out = []
    for text in SENTENCES.read_text(encoding="utf-8").splitlines():
        if text.strip():
            wav = torch.from_numpy(np.concatenate(list(tts.synthesize(text))))
            out.append((text, torchaudio.functional.resample(wav, tts.sample_rate, RATE).numpy(), "synthetic"))
    del tts
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_utterance(asr: NemotronStreamingASR, audio: np.ndarray, step: int) -> tuple[str, float, float]:
    """Return (final text, seconds spent in push, seconds spent in finish)."""
    asr.reset()
    compute = 0.0
    for i in range(0, len(audio), step):
        t = time.perf_counter()
        asr.push(audio[i:i + step])
        compute += time.perf_counter() - t
    t = time.perf_counter()
    text = asr.finish()
    flush = time.perf_counter() - t
    return text, compute, flush


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, nargs="+", default=[80, 160, 560])
    parser.add_argument("--manifest", default="utterances.yaml")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    apply_cpu_threads(settings.devices.cpu_threads)
    data = synthesize(settings) if args.synthetic else load_manifest(args.manifest)
    sources = sorted({s for _, _, s in data})
    print(f"{len(data)} utterances from {sources}, {sum(len(a) for _, a, _ in data) / RATE:.0f} s of audio. "
          "(Measured, this machine)\n")
    print("| chunk ms | RTF pushes only p50 | RTF incl. flush p50 | RTF incl. flush p90 | flush ms p50 | flush ms p90 "
          "| process peak MiB |")
    print("|---|---|---|---|---|---|---|")
    wer_lines = []
    for chunk_ms in args.chunks:
        gc.collect()
        torch.cuda.empty_cache()
        asr = NemotronStreamingASR(settings.asr.model_copy(update={"chunk_ms": chunk_ms}), settings.devices.asr)
        assert_resident("asr", asr.torch_modules(), settings.devices.asr)
        step = RATE * chunk_ms // 1000
        for _, audio, _ in data[:3]:  # warm-up, excluded
            run_utterance(asr, audio, step)
        torch.cuda.reset_peak_memory_stats()
        with ClockSampler() as clocks:
            rows = [(text, source, len(audio) / RATE, *run_utterance(asr, audio, step))
                    for text, audio, source in data]
        pushes = [push / seconds for _, _, seconds, _, push, _ in rows]
        total = [(push + fin) / seconds for _, _, seconds, _, push, fin in rows]
        flush = [r[5] * 1000 for r in rows]
        print(f"| {chunk_ms} | {percentile(pushes, 50):.2f} | {percentile(total, 50):.2f} | {percentile(total, 90):.2f} | "
              f"{percentile(flush, 50):.0f} | {percentile(flush, 90):.0f} | "
              f"{process_memory_mib()['peak_allocated']:.0f} |", flush=True)
        print(f"  {clocks.summary()}", flush=True)
        if not args.synthetic:
            by_source = defaultdict(list)
            for ref, source, _, hyp, _, _ in rows:
                by_source[source].append((ref, hyp))
            wer_lines.append(f"| {chunk_ms} | " + " | ".join(
                f"{wer(by_source[s]):.1%} (n={len(by_source[s])})" for s in sources) + " |")
        asr.close()  # the decode thread holds a reference, so del alone would keep the weights on the GPU
        del asr
    if wer_lines:
        print("\nWER by source (corpus WER, case and punctuation ignored)\n")
        print("| chunk ms | " + " | ".join(sources) + " |")
        print("|---|" + "---|" * len(sources))
        print("\n".join(wer_lines))
    print(f"\nwhole-GPU used {device_memory_mib()['used']:.0f} MiB at end (incl. CUDA context and Windows)")


if __name__ == "__main__":
    main()
