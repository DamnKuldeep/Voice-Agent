"""Opt-in saving of one spoken utterance into the evaluation set. Nothing is written unless asked."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from ..config import ROOT

RATE = 16000
_LEAD_IN_S = 0.3
_SPEECH_PEAK = 0.02


def trim_leading_silence(audio: np.ndarray) -> np.ndarray:
    """Drop quiet audio before the first speech, keeping a short lead-in."""
    loud = np.flatnonzero(np.abs(audio) > _SPEECH_PEAK)
    if loud.size == 0:
        return audio
    return audio[max(0, int(loud[0]) - int(_LEAD_IN_S * RATE)):]


def save_utterance(
    audio: np.ndarray,
    text: str,
    audio_dir: Path = ROOT / "evals" / "audio",
    manifest: Path = ROOT / "evals" / "sets" / "utterances.yaml",
) -> str:
    """Write the wav and append a manifest entry. Returns the new utterance id."""
    if audio.size == 0:
        raise ValueError("no utterance recorded yet")
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
    items = (data or {}).get("utterances", [])
    utterance_id = "ui_" + time.strftime("%Y%m%d_%H%M%S")
    if any(item["id"] == utterance_id for item in items):
        utterance_id += f"_{len(items)}"
    sf.write(audio_dir / f"{utterance_id}.wav", trim_leading_silence(audio), RATE)
    items.append({"id": utterance_id, "file": f"evals/audio/{utterance_id}.wav",
                  "text": text.strip(), "source": "own_voice_ui"})
    manifest.write_text(yaml.safe_dump({"utterances": items}, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    return utterance_id
