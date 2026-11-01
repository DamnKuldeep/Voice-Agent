"""Brief-by-default answers, with more room only when the user asks for detail."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from voiceagent.config import LengthPolicy

_HINTS = {
    "brief": "For this reply, answer in one or two short spoken sentences.",
    "detail": (
        "For this reply, explain clearly in under 80 words and finish on a complete sentence."
    ),
}


@lru_cache(maxsize=8)
def _cue_pattern(cues: tuple[str, ...]) -> re.Pattern[str] | None:
    if not cues:
        return None
    alternatives = (r"\s+".join(re.escape(word) for word in cue.split()) for cue in cues)
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


def choose_mode(user_text: str, cfg: LengthPolicy) -> Literal["brief", "detail"]:
    pattern = _cue_pattern(tuple(cfg.detail_cues))
    return "detail" if pattern and pattern.search(user_text) else "brief"


def max_tokens_for(mode: str, cfg: LengthPolicy) -> int:
    return (cfg.detail if mode == "detail" else cfg.brief).max_tokens


def style_hint(mode: str) -> str:
    return _HINTS["detail" if mode == "detail" else "brief"]
