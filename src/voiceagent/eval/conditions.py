"""Experiment conditions from ``--compare`` and the order they run in."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from ..config import Settings, parse_dotted

T = TypeVar("T")

# Sections whose values are baked into the loaded models or clients at build time, so changing any of
# them means building new Components. Everything else is read by the session and can change per run.
COMPONENT_SECTIONS = ("devices", "audio", "asr", "tts", "llm")


@dataclass(frozen=True)
class Condition:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)


def parse_conditions(compare: Sequence[Sequence[str]] | None) -> list[Condition]:
    """One condition per ``--compare k=v [k=v ...]``; none given means the config as it is."""
    if not compare:
        return [Condition("default")]
    labels = [" ".join(pairs) for pairs in compare]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate --compare conditions: {labels}")
    return [Condition(label, parse_dotted(list(pairs))) for label, pairs in zip(labels, compare)]


def interleave(
    items: Sequence[T], conditions: Sequence[Condition], block: int = 1
) -> list[tuple[T, Condition]]:
    """Every condition in turn, ``block`` items at a time (A, B, A, B for block 1), so drift in the
    network or GPU temperature lands on all conditions alike. Conditions that need different models use
    blocks so the models are not reloaded (about 20 s) for every single item."""
    groups = [items[i:i + block] for i in range(0, len(items), block)]
    return [(item, cond) for group in groups for cond in conditions for item in group]


def component_key(settings: Settings) -> str:
    """Equal keys mean one Components object can serve both settings."""
    parts = {name: getattr(settings, name).model_dump() for name in COMPONENT_SECTIONS}
    parts["turn_model"] = settings.turn.detector == "smart_turn"
    return json.dumps(parts, sort_keys=True)
