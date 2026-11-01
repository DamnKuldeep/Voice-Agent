"""Listens to a session's events and keeps what the scorers need, time-stamped on the trace clock."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable


class Observer:
    """Pass to ``ConversationSession.set_listener``. Only the public event stream is used."""

    def __init__(self) -> None:
        self.turns: list[dict] = []          # "turn" events, in order
        self.commits = 0                     # entries into "thinking": the session decided a turn ended
        self.counts: Counter[str] = Counter()
        self.asr_lag_max_ms: float | None = None
        self.barge_partials: list[str] = []  # words the session heard while the agent talked
        self.state = "idle"
        self.spoke_at: float | None = None   # first entry into "speaking"
        self._first: dict[tuple[str, str], float] = {}

    def __call__(self, event: dict) -> None:
        kind = event.get("type", "")
        if kind == "level":
            return
        now = time.perf_counter()
        self.counts[kind] += 1
        if kind == "state":
            self.state = event["state"]
            self.commits += self.state == "thinking"
            if self.state == "speaking" and self.spoke_at is None:
                self.spoke_at = now
        elif kind == "turn":
            self.turns.append(event)
        elif kind == "asr_lag":
            self.asr_lag_max_ms = max(self.asr_lag_max_ms or 0.0, float(event["ms"]))
        elif kind == "partial" and event.get("barge"):
            self.barge_partials.append(event["text"])
        self._first.setdefault((kind, str(event.get("phase", ""))), now)

    def first(self, kind: str, phase: str = "") -> float | None:
        """perf_counter time of the first event of ``kind`` (and ``phase``, for barge events)."""
        return self._first.get((kind, phase))

    @property
    def interrupted(self) -> bool:
        return any(t.get("flags", {}).get("interrupted") for t in self.turns)


def settled(condition: Callable[[], bool], steps: int) -> Callable[[], bool]:
    """True once ``condition`` has held on ``steps`` consecutive calls (replay calls once per 20 ms step)."""
    run = 0

    def check() -> bool:
        nonlocal run
        run = run + 1 if condition() else 0
        return run >= steps

    return check
