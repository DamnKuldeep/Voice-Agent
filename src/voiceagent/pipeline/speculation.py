"""Speculative LLM start: detect a stable partial transcript, reuse the result if the final matches."""

from __future__ import annotations

import re
from typing import Protocol

from ..config import Speculation
from .turn_manager import ends_with_hold_word


class Speculative(Protocol):
    tokens_used: int

    def cancel(self) -> None: ...


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


class Speculator:
    def __init__(self, cfg: Speculation, hold_words: list[str]) -> None:
        self._cfg = cfg
        self._hold_words = hold_words
        self.hits = 0
        self.wasted = 0
        self.wasted_tokens = 0
        self.reset()

    def reset(self) -> None:
        """Start of a turn. Anything still running must have been resolved or cancelled already."""
        self._last = ""
        self._stable = 0
        self._handle: Speculative | None = None
        self._text = ""

    @property
    def running(self) -> bool:
        return self._handle is not None

    def observe(self, partial: str) -> None:
        norm = normalize(partial)
        self._stable = self._stable + 1 if norm == self._last else 1
        self._last = norm

    def should_start(self, silence_ms: int, partial: str) -> bool:
        return (
            self._cfg.enabled
            and self._handle is None
            and bool(self._last)
            and self._stable >= self._cfg.stable_chunks
            and silence_ms >= self._cfg.start_silence_ms
            and not ends_with_hold_word(partial, self._hold_words)
        )

    def start(self, handle: Speculative, text: str) -> None:
        self._handle, self._text = handle, text

    def resolve(self, final_text: str) -> Speculative | None:
        """The running attempt if it answers ``final_text``; otherwise it is cancelled and None returned."""
        handle = self._handle
        if handle is None:
            return None
        if normalize(final_text) == normalize(self._text):
            self._handle = None
            self.hits += 1
            return handle
        self.cancel()
        return None

    def cancel(self) -> bool:
        """Stop a running attempt (the user resumed or the text changed). True if one was running."""
        handle = self._handle
        if handle is None:
            return False
        handle.cancel()
        self.wasted += 1
        self.wasted_tokens += handle.tokens_used
        self._handle = None
        return True
