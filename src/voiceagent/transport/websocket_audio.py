"""Browser audio over a WebSocket, so the agent works through any HTTPS tunnel from any network.

WebRTC media travels outside the tunnel and needs a TURN relay off the local network; a WebSocket is
ordinary HTTP traffic and goes wherever the page goes.

Protocol on ``/ws/audio``:
- client to server, binary: 16 kHz mono int16 little-endian PCM, any length per message;
- server to client, binary: agent audio at ``audio.output_rate`` (24 kHz), mono int16, one 20 ms frame
  per message, paced in real time so the page buffers little and a barge-in leaves only a few frames;
- server to client, text: ``{"type": "error", "message": ...}``.
The page empties its playback buffer itself on the barge-in "stop" event from the event stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..core.interfaces import ConversationHost

if TYPE_CHECKING:
    from fastapi import FastAPI

MIC_RATE = 16000
PATH = "/ws/audio"
_FRAME_S = 0.02
_IDLE_SLEEP_S = 0.01


class WsAudioLink:
    def __init__(self, host: ConversationHost,
                 on_mic: Callable[[int, np.ndarray], None] | None = None) -> None:
        self._host = host
        self._on_mic = on_mic

    def mount(self, app: FastAPI) -> None:
        app.add_api_websocket_route(PATH, self.serve)

    async def serve(self, ws: WebSocket) -> None:
        await ws.accept()
        owner = object()  # each connection is its own owner; the newest one wins, as with WebRTC
        try:
            await self._host.open_session(owner)
        except RuntimeError as exc:  # models still loading or failed to load
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close()
            return
        sender = asyncio.ensure_future(self._send(ws, owner))
        try:
            await self._receive(ws, owner)
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await sender
            await self._host.close_session(owner)

    async def _receive(self, ws: WebSocket, owner: object) -> None:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes")
            if not data:
                continue
            pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
            if self._on_mic is not None:
                self._on_mic(MIC_RATE, pcm)
            session = self._host.session_for(owner)
            if session is not None:
                session.feed_mic(pcm, MIC_RATE)

    async def _send(self, ws: WebSocket, owner: object) -> None:
        # Paced like the WebRTC emit(): frames leave on a real-time schedule, so a barge-in clears the
        # session's playout queue before much of the reply has left this machine.
        next_due = 0.0
        while True:
            session = self._host.session_for(owner)
            if session is None:  # superseded by a newer connection
                await asyncio.sleep(_IDLE_SLEEP_S)
                continue
            now = time.monotonic()
            if now < next_due:
                await asyncio.sleep(next_due - now)
            frame = session.next_playout_frame()
            if frame is None:
                next_due = 0.0
                await asyncio.sleep(_IDLE_SLEEP_S)
                continue
            next_due = max(next_due, time.monotonic()) + _FRAME_S
            await ws.send_bytes((np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
