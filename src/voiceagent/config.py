"""Typed settings loaded from YAML. Unknown keys are rejected so config and code cannot drift apart."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Devices(_Strict):
    asr: str
    tts: str
    vad: str
    turn: str
    require_full_gpu_residency: bool
    cpu_threads: int


class Audio(_Strict):
    analysis_rate: int
    output_rate: int
    frame_ms: int


class Vad(_Strict):
    threshold: float
    min_speech_ms: int
    pre_roll_ms: int
    gate_asr: bool
    end_hangover_ms: int


class Turn(_Strict):
    detector: Literal["smart_turn", "silence"]
    min_silence_ms: int
    max_silence_ms: int
    silence_only_ms: int
    complete_prob: float
    incomplete_prob: float
    hold_trailing_words: list[str]


class Speculation(_Strict):
    enabled: bool
    start_silence_ms: int
    stable_chunks: int


class Asr(_Strict):
    backend: Literal["nemotron"]
    model: str
    chunk_ms: Literal[80, 160, 560, 1120]
    dtype: Literal["float16", "float32"]
    flush_chunks: int
    tail_ms: int


class Llm(_Strict):
    model: str
    fallback_models: list[str]
    routing_sort: Literal["latency", "throughput", "price", "default"]
    provider_order: list[str]
    stream: bool
    ttft_timeout_s: float
    hedge_after_s: float
    temperature: float
    history_turns: int
    prewarm_connection: bool
    persona_file: str


class TokenBudget(_Strict):
    max_tokens: int


class LengthPolicy(_Strict):
    enabled: bool
    brief: TokenBudget
    detail: TokenBudget
    detail_cues: list[str]


class Chunker(_Strict):
    first_chunk_min_words: int
    first_chunk_max_words: int
    chunk_max_words: int


class Tts(_Strict):
    backend: Literal["kokoro"]
    voice: str
    speed: float
    stream: bool
    prefetch_chunks: int
    warmup: bool
    fillers: bool
    filler_after_ms: int


class BargeIn(_Strict):
    enabled: bool
    min_speech_ms: int
    vad_threshold: float
    ignore_backchannels: bool
    backchannel_max_ms: int
    duck_gain: float = Field(ge=0.0, le=1.0)
    echo_guard: bool


class Ui(_Strict):
    host: str
    port: int
    share: bool
    transport: Literal["auto", "webrtc", "websocket"]


class Settings(_Strict):
    devices: Devices
    audio: Audio
    vad: Vad
    turn: Turn
    speculation: Speculation
    asr: Asr
    llm: Llm
    length_policy: LengthPolicy
    chunker: Chunker
    tts: Tts
    barge_in: BargeIn
    ui: Ui


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``extra`` merged in, recursing into nested dicts."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_dotted(pairs: list[str]) -> dict[str, Any]:
    """Turn ``["tts.stream=false", "llm.model=x/y"]`` into a nested dict. Values are parsed as YAML."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def load_settings(
    path: str | Path = DEFAULT_CONFIG, overrides: dict[str, Any] | None = None
) -> Settings:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if overrides:
        data = deep_merge(data, overrides)
    return Settings(**data)
