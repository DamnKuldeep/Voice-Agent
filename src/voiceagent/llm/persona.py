"""System prompt loading and chat message assembly."""

from __future__ import annotations

from pathlib import Path

from voiceagent.config import ROOT
from voiceagent.llm.length_policy import style_hint


def load_persona(path: str | Path) -> str:
    file = Path(path)
    if not file.is_absolute():
        file = ROOT / file
    return file.read_text(encoding="utf-8").strip()


def build_messages(
    persona: str,
    history: list[dict[str, str]],
    user_text: str,
    mode: str,
    length_enabled: bool,
) -> list[dict[str, str]]:
    system = f"{persona}\n\n{style_hint(mode)}" if length_enabled else persona
    return [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": user_text},
    ]
