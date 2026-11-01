"""Commit policy: given how long the user has been silent, decide whether the turn is over."""

from __future__ import annotations

import re

from ..config import Turn

_WORD = re.compile(r"[a-z']+")


def ends_with_hold_word(text: str, hold_words: list[str]) -> bool:
    words = _WORD.findall(text.lower())
    return bool(words) and words[-1] in hold_words


def commit_after_ms(cfg: Turn, *, eot_prob: float | None, transcript: str) -> int:
    """Silence needed before committing. Unknown probability waits the full ``max_silence_ms``."""
    if cfg.detector == "silence":
        return cfg.silence_only_ms
    if ends_with_hold_word(transcript, cfg.hold_trailing_words) or eot_prob is None:
        return cfg.max_silence_ms
    if eot_prob >= cfg.complete_prob:
        return cfg.min_silence_ms
    if eot_prob <= cfg.incomplete_prob:
        return cfg.max_silence_ms
    return (cfg.min_silence_ms + cfg.max_silence_ms) // 2


def should_commit(silence_ms: int, cfg: Turn, *, eot_prob: float | None, transcript: str) -> bool:
    return silence_ms >= commit_after_ms(cfg, eot_prob=eot_prob, transcript=transcript)
