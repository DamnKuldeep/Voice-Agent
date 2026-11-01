"""Per-turn timing trace and summary statistics. Pure Python, no model dependencies."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from .events import Flag, Mark

# derived metric name -> (start mark, end mark). All results are in milliseconds.
SPANS: dict[str, tuple[Mark, Mark]] = {
    "endpoint_ms": (Mark.SPEECH_END, Mark.COMMIT),
    "asr_final_ms": (Mark.COMMIT, Mark.ASR_FINAL),
    "asr_flush_ms": (Mark.ASR_TAIL_END, Mark.ASR_FINAL),  # asr_final_ms without the post-commit tail
    "llm_ttft_ms": (Mark.LLM_REQUEST, Mark.LLM_FIRST_TOKEN),
    "first_chunk_ms": (Mark.LLM_FIRST_TOKEN, Mark.FIRST_CHUNK_READY),
    "tts_ttfa_ms": (Mark.FIRST_CHUNK_READY, Mark.TTS_FIRST_AUDIO),
    "v2v_ms": (Mark.SPEECH_END, Mark.AUDIO_OUT_START),  # what the user perceives
    "reply_audio_ms": (Mark.AUDIO_OUT_START, Mark.AUDIO_OUT_END),
    "barge_in_duck_ms": (Mark.BARGE_IN_ONSET, Mark.BARGE_IN_DUCKED),  # the yield the user perceives
    "barge_in_stop_ms": (Mark.BARGE_IN_ONSET, Mark.PLAYBACK_STOPPED),
}


@dataclass
class TurnTrace:
    turn_id: int
    marks: dict[Mark, float] = field(default_factory=dict)
    flags: dict[Flag, object] = field(default_factory=dict)

    def mark(self, name: Mark, at: float | None = None) -> None:
        """Record a timestamp (seconds, perf_counter clock). The first write wins."""
        self.marks.setdefault(name, time.perf_counter() if at is None else at)

    def flag(self, name: Flag, value: object = True) -> None:
        self.flags[name] = value

    def span_ms(self, start: Mark, end: Mark) -> float | None:
        if start not in self.marks or end not in self.marks:
            return None
        return (self.marks[end] - self.marks[start]) * 1000.0

    def derived(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, (start, end) in SPANS.items():
            value = self.span_ms(start, end)
            if value is not None:
                out[name] = value
        return out


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(traces: Iterable[TurnTrace]) -> dict[str, dict[str, float]]:
    """For each derived metric: count, mean, p50, p90 and max across turns."""
    columns: dict[str, list[float]] = {}
    for trace in traces:
        for name, value in trace.derived().items():
            columns.setdefault(name, []).append(value)
    return {
        name: {
            "n": float(len(vals)),
            "mean": sum(vals) / len(vals),
            "p50": percentile(vals, 50),
            "p90": percentile(vals, 90),
            "max": max(vals),
        }
        for name, vals in columns.items()
    }
