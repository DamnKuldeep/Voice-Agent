"""Fetch public speech with reference text, for ASR and listening tests without your own recordings.

Three sources, all read speech, written to evals/audio/ (git-ignored) with a text manifest:
  cv_in_*   Common Voice clips by Indian-accent speakers, one clip per speaker (accent stress test)
  svarah_*  AI4Bharat Svarah (CC BY 4.0, gated: accept the terms on the dataset page), many mother tongues
  libri_*   LibriSpeech dummy validation set (clean US read speech, control)
Results on them are labelled by source; they measure these speakers, not any one user's voice.
HF_TOKEN in .env is needed for the gated Svarah set.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
import yaml
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from voiceagent.asr.nemotron import RATE
from voiceagent.config import ROOT

AUDIO_DIR = ROOT / "evals" / "audio"
MANIFEST = ROOT / "evals" / "sets" / "public_utterances.yaml"
SVARAH_REPO, SVARAH_FILE = "ai4bharat/Svarah", "data/test-00000-of-00003.parquet"
CV_REPO, CV_FILE ="ishands/commonvoice-indian_accent", "data/train-00000-of-00009.parquet"
LIBRI_REPO, LIBRI_FILE = "hf-internal-testing/librispeech_asr_dummy", "clean/validation-00000-of-00001.parquet"


def to_16k(raw: bytes) -> torch.Tensor:
    audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return torchaudio.functional.resample(torch.from_numpy(audio.mean(axis=1)), rate, RATE)


def save(name: str, raw: bytes) -> str:
    path = AUDIO_DIR / f"{name}.wav"
    sf.write(path, to_16k(raw).numpy(), RATE)
    return f"evals/audio/{name}.wav"


def common_voice(token: str, n: int) -> list[dict]:
    table = pq.read_table(hf_hub_download(CV_REPO, CV_FILE, repo_type="dataset", token=token))
    rows = table.to_pylist()
    keep = [r for r in rows if "India" in (r["accents"] or "") and r["up_votes"] >= 2
            and r["down_votes"] == 0 and 3000 <= r["duration_ms"] <= 10000]
    best: dict[str, dict] = {}
    for r in sorted(keep, key=lambda r: hashlib.sha1(r["sentence_id"].encode()).hexdigest()):
        best.setdefault(r["client_id"], r)  # one clip per speaker, deterministic order
    picked = list(best.values())[:n]
    return [{"id": f"cv_in_{i:03d}", "file": save(f"cv_in_{i:03d}", r["audio"]["bytes"]),
             "text": r["sentence"], "source": "common_voice_indian_accent"}
            for i, r in enumerate(picked)]


def svarah(token: str, n: int) -> list[dict]:
    table = pq.read_table(hf_hub_download(SVARAH_REPO, SVARAH_FILE, repo_type="dataset", token=token))
    rows = [r for r in table.to_pylist() if 2.0 <= r["duration"] <= 10.0 and 3 <= len(r["text"].split()) <= 25]
    by_language: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: hashlib.sha1(r["text"].encode()).hexdigest()):
        by_language.setdefault(r["primary_language"], []).append(r)
    picked = []
    while len(picked) < n and any(by_language.values()):  # round robin so no mother tongue dominates
        for clips in by_language.values():
            if clips and len(picked) < n:
                picked.append(clips.pop(0))
    return [{"id": f"svarah_{i:03d}", "file": save(f"svarah_{i:03d}", r["audio_filepath"]["bytes"]),
             "text": " ".join(r["text"].split()), "source": "svarah_ai4bharat_cc_by_4",
             "mother_tongue": r["primary_language"]}
            for i, r in enumerate(picked)]


def librispeech(token: str, n: int) -> list[dict]:
    rows = pq.read_table(hf_hub_download(LIBRI_REPO, LIBRI_FILE, repo_type="dataset", token=token)).to_pylist()[:n]
    return [{"id": f"libri_{i:03d}", "file": save(f"libri_{i:03d}", r["audio"]["bytes"]),
             "text": r["text"].lower(), "source": "librispeech_dummy"}
            for i, r in enumerate(rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indian", type=int, default=60)
    parser.add_argument("--svarah", type=int, default=60)
    parser.add_argument("--libri", type=int, default=30)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    token = os.environ["HF_TOKEN"]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    items = common_voice(token, args.indian) + svarah(token, args.svarah) + librispeech(token, args.libri)
    MANIFEST.write_text(yaml.safe_dump({"utterances": items}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {len(items)} clips and {MANIFEST}")


if __name__ == "__main__":
    main()
