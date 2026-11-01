"""Record yourself reading evals/sets/read_aloud.txt, one 16 kHz mono WAV per sentence.

Writes evals/audio/read_NN.wav (git-ignored) and evals/sets/utterances.yaml (reference text).
Press Enter to start each sentence and Enter again to stop; type r to redo, s to skip, q to quit.
Use the headset you will use in the live demo.
"""

from __future__ import annotations

import queue

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml

from voiceagent.config import ROOT

RATE = 16000
SENTENCES = ROOT / "evals" / "sets" / "read_aloud.txt"
AUDIO_DIR = ROOT / "evals" / "audio"
MANIFEST = ROOT / "evals" / "sets" / "utterances.yaml"


def record_until_enter() -> np.ndarray:
    blocks: queue.Queue[np.ndarray] = queue.Queue()
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                        callback=lambda data, *_: blocks.put(data.copy())):
        input("  recording... press Enter to stop ")
    chunks = []
    while not blocks.empty():
        chunks.append(blocks.get())
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype=np.float32)


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in SENTENCES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    items = []
    print(f"{len(lines)} sentences. Default input device: {sd.query_devices(kind='input')['name']}")
    for i, text in enumerate(lines, 1):
        while True:
            print(f"\n[{i}/{len(lines)}] {text}")
            cmd = input("  Enter = record, s = skip, q = quit: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                break
            audio = record_until_enter()
            peak = float(np.abs(audio).max()) if audio.size else 0.0
            print(f"  {audio.size / RATE:.1f} s, peak {peak:.2f}" + ("  (too quiet?)" if peak < 0.05 else ""))
            if input("  keep? (Enter = yes, r = redo): ").strip().lower() == "r":
                continue
            name = f"read_{i:02d}.wav"
            sf.write(AUDIO_DIR / name, audio, RATE)
            items.append({"id": f"read_{i:02d}", "file": f"evals/audio/{name}", "text": text,
                          "source": "own_voice"})
            break
        if cmd == "q":
            break
    MANIFEST.write_text(yaml.safe_dump({"utterances": items}, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    print(f"\nSaved {len(items)} recordings; manifest at {MANIFEST}")


if __name__ == "__main__":
    main()
