"""Fan-out of session events to every open page, as Server-Sent Events.

Events are published from the session's loop thread (audio path), the model-loader thread and request
threads. Publishing never blocks: each page has a bounded queue that drops its oldest event when full,
and level events are latest-wins, so a slow or stalled browser tab can never hold up the audio.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import threading
from collections.abc import AsyncIterator
from typing import Any

QUEUE_SIZE = 256
KEEPALIVE_S = 15.0


class Subscription:
    def __init__(self, loop: asyncio.AbstractEventLoop, size: int) -> None:
        self.loop = loop
        self.dropped = 0
        self._events: collections.deque[dict[str, Any]] = collections.deque()
        self._size = size
        self._level: dict[str, Any] | None = None
        self._wake = asyncio.Event()

    def push(self, event: dict[str, Any]) -> None:
        """Loop thread only."""
        if event.get("type") == "level":
            self._level = event
        else:
            if len(self._events) >= self._size:
                self._events.popleft()
                self.dropped += 1
            self._events.append(event)
        self._wake.set()

    def pending(self) -> int:
        return len(self._events) + (self._level is not None)

    async def get(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Next event, or None after ``timeout`` seconds with nothing to send."""
        while True:
            if self._events:
                return self._events.popleft()
            if self._level is not None:
                event, self._level = self._level, None
                return event
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout)
            except TimeoutError:
                return None


class EventHub:
    def __init__(self, size: int = QUEUE_SIZE) -> None:
        self._size = size
        self._subs: set[Subscription] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> Subscription:
        """Call from a coroutine: the subscription lives on the running loop."""
        sub = Subscription(asyncio.get_running_loop(), self._size)
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.discard(sub)

    @property
    def subscribers(self) -> int:
        with self._lock:
            return len(self._subs)

    def publish(self, event: dict[str, Any]) -> None:
        """Thread-safe and non-blocking; delivery happens on each subscriber's loop."""
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if _on_loop(sub.loop):
                sub.push(event)
            elif not sub.loop.is_closed():
                sub.loop.call_soon_threadsafe(sub.push, event)


def _on_loop(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def to_json(event: dict[str, Any]) -> str:
    return json.dumps(event, default=_jsonable, separators=(",", ":"))


def sse_format(event: dict[str, Any]) -> str:
    return f"data: {to_json(event)}\n\n"


def _jsonable(value: Any) -> Any:
    return getattr(value, "value", str(value))


async def events_after(
    hub: EventHub, first: dict[str, Any], keepalive_s: float = KEEPALIVE_S
) -> AsyncIterator[dict[str, Any] | None]:
    """``first`` (a status snapshot), then every published event; None when a keep-alive is due."""
    sub = hub.subscribe()
    try:
        yield first
        while True:
            yield await sub.get(keepalive_s)
    finally:
        hub.unsubscribe(sub)


async def sse_stream(
    hub: EventHub, first: dict[str, Any], keepalive_s: float = KEEPALIVE_S
) -> AsyncIterator[str]:
    """Body of GET /api/events."""
    # aclosing: when the client leaves, the inner generator must end too, or its subscription stays.
    async with contextlib.aclosing(events_after(hub, first, keepalive_s)) as events:
        async for event in events:
            yield ": keep-alive\n\n" if event is None else sse_format(event)
