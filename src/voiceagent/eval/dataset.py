"""Spoken test inputs: manifest loading and deterministic, source-balanced subsets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..config import ROOT

SETS_DIR = ROOT / "evals" / "sets"
RATE = 16000


@dataclass(frozen=True)
class Utterance:
    id: str
    file: str
    text: str
    source: str


def load_manifest(name: str | Path) -> list[Utterance]:
    path = Path(name) if Path(name).is_absolute() else SETS_DIR / name
    items = yaml.safe_load(path.read_text(encoding="utf-8"))["utterances"]
    return [Utterance(i["id"], i["file"], i["text"], i.get("source", "own_voice")) for i in items]


def balanced_subset(items: list[Utterance], n: int) -> list[Utterance]:
    """Round-robin over sources (sorted by name), each source in manifest order, until ``n`` are taken.

    The same manifest and ``n`` always give the same clips, so two runs measure the same inputs.
    """
    by_source: dict[str, list[Utterance]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)
    queues = [by_source[s] for s in sorted(by_source)]
    out: list[Utterance] = []
    depth = 0
    while len(out) < n and any(depth < len(q) for q in queues):
        out.extend(q[depth] for q in queues if depth < len(q))
        depth += 1
    return out[:n]


def load_audio(item: Utterance) -> np.ndarray:
    """16 kHz mono float32."""
    import soundfile as sf

    from ..audio.frames import resample_to_16k

    audio, rate = sf.read(ROOT / item.file, dtype="float32")
    return resample_to_16k(audio, rate) if rate != RATE or audio.ndim > 1 else audio
