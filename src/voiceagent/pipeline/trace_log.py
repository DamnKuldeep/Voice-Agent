"""One JSON line per completed turn, appended to evals/results/traces/<date>.jsonl."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path

from ..config import ROOT, Settings
from ..core.metrics import TurnTrace

log = logging.getLogger(__name__)

TRACE_DIR = ROOT / "evals" / "results" / "traces"


def config_hash(settings: Settings) -> str:
    return hashlib.sha1(settings.model_dump_json().encode("utf-8")).hexdigest()[:12]


class TraceWriter:
    def __init__(
        self, directory: Path, settings: Settings, vram_peak_mib: Callable[[], float] | None = None
    ) -> None:
        self._dir = directory
        self._hash = config_hash(settings)
        self._vram = vram_peak_mib
        self._lock = threading.Lock()

    def write(self, trace: TurnTrace, transcript: str, reply: str) -> None:
        origin = min(trace.marks.values(), default=0.0)
        record = {
            "turn_id": trace.turn_id,
            "marks_ms": {m.value: round((t - origin) * 1000, 1) for m, t in trace.marks.items()},
            "metrics": {k: round(v, 1) for k, v in trace.derived().items()},
            "flags": {f.value: v for f, v in trace.flags.items()},
            "transcript": transcript,
            "reply": reply,
            "config_hash": self._hash,
            "vram_peak_mib": round(self._vram(), 1) if self._vram else None,
        }
        try:
            with self._lock:
                self._dir.mkdir(parents=True, exist_ok=True)
                with (self._dir / f"{date.today().isoformat()}.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.warning("could not write turn trace: %s", exc)
