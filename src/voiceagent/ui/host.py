"""Shared application state: the one loaded Components object, the one active session, its events."""

from __future__ import annotations

import asyncio
import collections
import threading
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import DEFAULT_CONFIG, Settings, deep_merge, load_settings
from . import controls
from .event_hub import EventHub

if TYPE_CHECKING:
    from ..pipeline.factory import Components
    from ..pipeline.session import ConversationSession

Progress = Callable[[str], None]
BuildFn = Callable[[Settings, Progress], "Components"]
SessionFn = Callable[[Settings, "Components"], "ConversationSession"]

MAX_TAP_S = 20
_LOOP_CALL_TIMEOUT_S = 30


def _build_components(settings: Settings, progress: Progress) -> Components:
    from ..pipeline.factory import build_components

    return build_components(settings, progress)


def _make_session(settings: Settings, components: Components) -> ConversationSession:
    from ..pipeline.session import ConversationSession

    return ConversationSession(settings, components)


class BusyError(RuntimeError):
    """Another apply or reload is still running."""


class SessionHost:
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        build: BuildFn = _build_components,
        make_session: SessionFn = _make_session,
        hub: EventHub | None = None,
    ) -> None:
        self.config_path = config_path
        self.defaults = load_settings(config_path)
        self.settings = self.defaults
        self.hub = hub or EventHub()
        self.components: Components | None = None
        self.phase = "loading"
        self.status = "Loading models..."
        self.ready = threading.Event()
        self.last_utterance = np.zeros(0, dtype=np.float32)
        self._overrides: dict[str, Any] = {}
        self._build = build
        self._make_session = make_session
        self._lock = threading.Lock()
        self._apply_lock = threading.Lock()
        self._session: ConversationSession | None = None
        self._owner: object | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tap: collections.deque[np.ndarray] = collections.deque()
        self._tap_samples = 0
        self._live: dict[str, Any] = {}
        self._reset_live()

    def _reset_live(self) -> None:
        with self._lock:
            self._live = {"state": "idle", "last_final": "", "last_turn": None, "asr_lag_ms": 0.0}

    # Models ----------------------------------------------------------------------------------------

    def _set_phase(self, phase: str, message: str) -> None:
        self.phase, self.status = phase, message
        self.hub.publish(self.status_event())

    def load(self) -> None:
        """Build the models. Blocking; progress and the outcome go to every page as events."""
        self.ready.clear()
        self._set_phase("loading", "Loading models (the first start takes a while)...")
        try:
            self.components = self._build(
                self.settings, lambda msg: self.hub.publish({"type": "progress", "message": msg}))
        except Exception as exc:  # noqa: BLE001 - reported on the page so the process stays up for a retry
            self._set_phase("error", f"Model loading failed: {type(exc).__name__}: {exc}")
            return
        self.ready.set()
        self._set_phase("ready", "Ready")

    def _needs_component(self, old: Settings, new: Settings) -> bool:
        """Switches that need a part the current build skipped: the turn model and the fillers."""
        c = self.components
        if c is None:
            return False
        wants_turn = new.turn.detector == "smart_turn" != old.turn.detector and c.turn_detector is None
        wants_fillers = new.tts.fillers and not old.tts.fillers and not c.fillers
        return wants_turn or wants_fillers

    def apply(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Apply ``{dotted key: value}``; rebuilds the models only when a reload-only switch changed.

        Blocking (a reload takes as long as a cold start). Call from a worker thread, not the loop.
        """
        if not self._apply_lock.acquire(blocking=False):
            raise BusyError("Settings are still being applied")
        try:
            overrides = deep_merge(self._overrides,
                                   controls.overrides_from_values(values, self.settings))
            new = load_settings(self.config_path, overrides)
            old = self.settings
            changed = controls.changed_keys(old, new)
            reload = controls.needs_reload(old, new) or self._needs_component(old, new)
            self._overrides, self.settings = overrides, new
            if reload:
                self._run_on_loop(self._stop_session())
                if self.components is not None:
                    self.components.close()
                    self.components = None
                self.load()
            if self._owner is not None and self.ready.is_set():
                self._run_on_loop(self._replace_session())
            ok = self.ready.is_set()
            message = (f"Applied {', '.join(changed) or 'no changes'}"
                       + (" (models reloaded)" if reload and ok else ""))
            return {"ok": ok, "changed": changed, "reloaded": reload,
                    "message": message if ok else self.status}
        finally:
            self._apply_lock.release()

    # Sessions --------------------------------------------------------------------------------------

    def reset_conversation(self) -> str:
        self._reset_live()
        if self._owner is not None:
            self._run_on_loop(self._replace_session())
        self.hub.publish({"type": "reset"})
        return "Conversation reset"

    def _run_on_loop(self, coro: Coroutine[Any, Any, None]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            coro.close()
            return
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            coro.close()
            raise RuntimeError("blocking host call made on the session loop")
        asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=_LOOP_CALL_TIMEOUT_S)

    async def _stop_session(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.stop()

    async def _replace_session(self) -> None:
        await self._stop_session()
        if self.components is None:
            return
        session = self._make_session(self.settings, self.components)
        session.set_listener(self._on_event)
        await session.start()
        self._session = session

    async def open_session(self, owner: object) -> ConversationSession:
        if not self.ready.is_set():
            raise RuntimeError(self.status)
        self._loop = asyncio.get_running_loop()
        self._owner = owner
        self._reset_live()
        await self._replace_session()
        assert self._session is not None
        self.hub.publish(self.status_event())
        return self._session

    async def close_session(self, owner: object) -> None:
        if owner is not self._owner:
            return
        self._owner = None
        await self._stop_session()
        with self._lock:
            self._live["state"] = "idle"
        self.hub.publish(self.status_event())

    def session_for(self, owner: object) -> ConversationSession | None:
        return self._session if owner is self._owner else None

    @property
    def active(self) -> bool:
        return self._owner is not None

    # Audio and events ------------------------------------------------------------------------------

    def tap_mic(self, rate: int, samples: np.ndarray) -> None:
        """Keep the last seconds of 16 kHz mic audio in memory so one utterance can be saved on request."""
        from ..audio.frames import resample_to_16k

        chunk = resample_to_16k(samples, rate)
        self._tap.append(chunk)
        self._tap_samples += chunk.size
        while self._tap_samples > MAX_TAP_S * 16000 and len(self._tap) > 1:
            self._tap_samples -= self._tap.popleft().size

    def _on_event(self, event: dict[str, Any]) -> None:
        """Session listener; runs on the session loop, so it only records and forwards."""
        kind = event.get("type")
        with self._lock:
            live = self._live
            if kind == "state":
                live["state"] = event["state"]
                if event["state"] == "thinking" and self._tap:
                    self.last_utterance = np.concatenate(list(self._tap))
                    self._tap.clear()
                    self._tap_samples = 0
            elif kind == "final":
                live["last_final"] = event["text"]
            elif kind == "asr_lag":
                live["asr_lag_ms"] = float(event["ms"])
            elif kind == "turn":
                live["last_turn"] = {"metrics": event["metrics"], "flags": event["flags"]}
        self.hub.publish(event)
        if kind == "turn" and self._session is not None:
            self.hub.publish({"type": "summary", "summary": self._session.summary()})

    def status_event(self) -> dict[str, Any]:
        """Everything a page needs to draw itself from scratch; also the first SSE event."""
        with self._lock:
            live = dict(self._live)
        session = self._session
        llm = self.settings.llm
        return {
            "type": "status", "phase": self.phase, "message": self.status,
            "ready": self.ready.is_set(), "active": self.active, "state": live["state"],
            "last_final": live["last_final"], "last_turn": live["last_turn"],
            "asr_lag_ms": live["asr_lag_ms"],
            "summary": session.summary() if session is not None else {},
            "config": {"llm_model": llm.model, "provider_order": list(llm.provider_order),
                       "routing_sort": llm.routing_sort, "asr_chunk_ms": self.settings.asr.chunk_ms,
                       "voice": self.settings.tts.voice, "transport": self.settings.ui.transport},
        }
