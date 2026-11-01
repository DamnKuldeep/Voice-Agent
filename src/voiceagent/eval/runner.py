"""Runs (item, condition) pairs through real sessions, one Components object at a time, and writes results."""

from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from ..config import Settings
from ..pipeline.factory import Components
from ..pipeline.trace_log import config_hash
from .conditions import Condition, component_key, interleave

RELOAD_BLOCK = 5  # items per condition between model reloads
from .scoring import markdown_table, summarise

log = logging.getLogger(__name__)

T = TypeVar("T")
Episode = Callable[[T, Settings, Components], Awaitable[dict]]
Builder = Callable[[Settings], Components]
Warmup = Callable[[Settings, Components], Awaitable[Any]]


class ComponentSlot:
    """Holds the one Components object the GPU has room for; builds a new one only when a condition
    changes something baked in at build time, and frees the old one first."""

    def __init__(self, build: Builder, warmup: Warmup | None = None) -> None:
        self._build, self._warmup = build, warmup
        self._key: str | None = None
        self.components: Components | None = None
        self.builds = 0

    async def acquire(self, settings: Settings) -> Components:
        key = component_key(settings)
        if self.components is None or key != self._key:
            self.close()
            self.components = self._build(settings)
            self._key = key
            self.builds += 1
            if self._warmup is not None:
                await self._warmup(settings, self.components)  # first session after a load: excluded
        return self.components

    def close(self) -> None:
        if self.components is not None:
            self.components.close()  # also closes the ASR, whose decode thread pins the weights
            self.components = None
            self._key = None
            gc.collect()


async def run_conditions(
    items: Sequence[T], conditions: Sequence[Condition], settings_for: Callable[[Condition], Settings],
    episode: Episode, slot: ComponentSlot, out_jsonl: Path, item_id: Callable[[T], str],
) -> list[dict]:
    """Interleaved A, B, A, B; every row is appended to ``out_jsonl`` as soon as it exists."""
    settings = {c.label: settings_for(c) for c in conditions}
    rows: list[dict] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    reloads = len({component_key(s) for s in settings.values()}) > 1
    pairs = interleave(items, conditions, RELOAD_BLOCK if reloads else 1)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for index, (item, cond) in enumerate(pairs, 1):
            started = time.perf_counter()
            try:
                # No local name for the Components: one kept across iterations would pin the old models
                # on the GPU while acquire() builds the next ones, and two sets do not fit in 4 GB.
                row = await episode(item, settings[cond.label], await slot.acquire(settings[cond.label]))
            except Exception as exc:  # one broken item must not lose the whole run
                log.exception("item %s under %s failed", item_id(item), cond.label)
                row = {"item": item_id(item), "error": f"{type(exc).__name__}: {exc}"}
            row.update(condition=cond.label, config_hash=config_hash(settings[cond.label]),
                       elapsed_s=round(time.perf_counter() - started, 2))
            rows.append(row)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            print(f"[{index}/{len(pairs)}] {cond.label} {row['item']}: {_one_line(row)}", flush=True)
    return rows


def _one_line(row: dict) -> str:
    if "error" in row:
        return row["error"]
    bits = [f"{k}={v}" for k, v in row.get("checks", {}).items() if v]
    metrics = row.get("metrics", {})
    for name in ("v2v_ms", "stop_after_onset_ms", "asr_final_ms"):
        if name in metrics:
            bits.append(f"{name}={metrics[name]:.0f}")
    if "wer" in row:
        bits.append(f"wer={row['wer']:.2f}")
    if row.get("hypothesis") is not None:
        bits.append(repr(row["hypothesis"][:70]))
    return " ".join(bits) or "ok"


def write_report(rows: list[dict], path: Path, *, title: str, notes: Sequence[str]) -> str:
    """Write the Markdown summary next to the JSONL and return the table."""
    scored = [r for r in rows if "error" not in r]
    table = markdown_table(summarise(scored))
    failed = len(rows) - len(scored)
    header = [f"# {title}", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}.", "", *notes]
    if failed:
        header.append(f"- {failed} item(s) failed and are excluded; see the JSONL `error` field.")
    path.write_text("\n".join([*header, "", table, ""]), encoding="utf-8")
    return table
