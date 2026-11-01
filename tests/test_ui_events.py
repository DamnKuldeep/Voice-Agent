"""The SSE bridge: non-blocking fan-out, bounded queues, latest-wins levels, thread-safe publish."""

from __future__ import annotations

import asyncio
import json
import threading

from voiceagent.ui.event_hub import EventHub, sse_format, sse_stream


async def test_every_subscriber_gets_events_in_order():
    hub = EventHub()
    a, b = hub.subscribe(), hub.subscribe()
    for i in range(3):
        hub.publish({"type": "partial", "text": str(i)})
    for sub in (a, b):
        assert [(await sub.get(0.1))["text"] for _ in range(3)] == ["0", "1", "2"]
    assert await a.get(0.01) is None
    hub.unsubscribe(a)
    hub.publish({"type": "final", "text": "x"})
    assert a.pending() == 0 and b.pending() == 1 and hub.subscribers == 1


async def test_full_queue_drops_oldest_and_levels_are_latest_wins():
    hub = EventHub(size=3)
    sub = hub.subscribe()
    for i in range(5):
        hub.publish({"type": "reply", "text": str(i)})
    for rms in (0.1, 0.2, 0.3):
        hub.publish({"type": "level", "rms": rms})
    assert sub.dropped == 2 and sub.pending() == 4
    got = [await sub.get(0.1) for _ in range(4)]
    assert [e.get("text") for e in got[:3]] == ["2", "3", "4"]
    assert got[3] == {"type": "level", "rms": 0.3}


async def test_publish_from_another_thread_lands_on_the_loop():
    hub = EventHub()
    sub = hub.subscribe()
    worker = threading.Thread(target=hub.publish, args=({"type": "progress", "message": "ASR"},))
    worker.start()
    worker.join()
    assert await sub.get(1.0) == {"type": "progress", "message": "ASR"}


async def test_sse_stream_sends_status_first_then_events_and_keepalives():
    hub = EventHub()
    stream = sse_stream(hub, {"type": "status", "phase": "ready"}, keepalive_s=0.02)
    first = await stream.__anext__()
    assert first.startswith("data: ") and first.endswith("\n\n")
    assert json.loads(first[6:]) == {"type": "status", "phase": "ready"}
    assert await stream.__anext__() == ": keep-alive\n\n"
    hub.publish({"type": "state", "state": "thinking"})
    assert json.loads((await stream.__anext__())[6:])["state"] == "thinking"
    await stream.aclose()
    assert hub.subscribers == 0


def test_sse_format_handles_enums_and_odd_values():
    from voiceagent.core.events import Flag

    text = sse_format({"type": "turn", "flags": {"llm_model": Flag.LLM_MODEL, "odd": object()}})
    flags = json.loads(text[6:])["flags"]
    assert flags["llm_model"] == "llm_model" and flags["odd"].startswith("<object")


def test_publish_after_loop_closed_is_ignored():
    hub = EventHub()
    loop = asyncio.new_event_loop()
    sub = loop.run_until_complete(_subscribe(hub))
    loop.close()
    hub.publish({"type": "state", "state": "idle"})
    assert sub.pending() == 0


async def _subscribe(hub: EventHub):
    return hub.subscribe()
