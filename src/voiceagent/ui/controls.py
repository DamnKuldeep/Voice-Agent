"""Settings drawer logic: definitions for the page and the conversion of page values to overrides."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config import Settings
from .control_catalog import CONTROLS
from .control_spec import GROUPS, Control

BY_KEY: dict[str, Control] = {c.key: c for c in CONTROLS}
RELOAD_KEYS = frozenset(c.key for c in CONTROLS if c.needs_reload)


def read_key(settings: Settings, key: str) -> Any:
    node: Any = settings
    for part in key.split("."):
        node = getattr(node, part)
    return node


def definitions(current: Settings, defaults: Settings) -> dict[str, Any]:
    """Groups and controls for the page; ``defaults`` is what Reset-to-defaults restores."""
    return {
        "groups": [{"id": gid, "label": label} for gid, label in GROUPS],
        "controls": [c.describe(read_key(current, c.key), read_key(defaults, c.key)) for c in CONTROLS],
    }


def overrides_from_values(values: Mapping[str, Any], settings: Settings) -> dict[str, Any]:
    """Nested override dict for ``load_settings(overrides=...)`` from ``{dotted key: raw value}``.

    Only the keys given are overridden; an unknown key or an out-of-range value raises ValueError.
    """
    out: dict[str, Any] = {}
    for key, raw in values.items():
        control = BY_KEY.get(key)
        if control is None:
            raise ValueError(f"unknown setting {key!r}")
        value = control.coerce(raw, read_key(settings, key))
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    chunker = {**settings.chunker.model_dump(), **out.get("chunker", {})}
    if chunker["first_chunk_min_words"] > chunker["first_chunk_max_words"]:
        raise ValueError("First spoken chunk: the minimum cannot be above the maximum")
    return out


def changed_keys(old: Settings, new: Settings) -> list[str]:
    return [c.key for c in CONTROLS if read_key(old, c.key) != read_key(new, c.key)]


def needs_reload(old: Settings, new: Settings) -> bool:
    """True when a changed control is baked into a loaded model or client, so Components must be rebuilt."""
    return any(key in RELOAD_KEYS for key in changed_keys(old, new))
