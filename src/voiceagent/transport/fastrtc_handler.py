"""The only FastRTC import: adapts a browser WebRTC audio stream to a ConversationSession.

``RtcLink`` mounts FastRTC's offer endpoint (``POST /webrtc/offer``) on our FastAPI app; the page's
own JavaScript does the signalling, so no Gradio component is involved.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Self

import numpy as np
from fastrtc import AsyncStreamHandler, Stream, WebRTCError, get_cloudflare_turn_credentials

from ..core.interfaces import ConversationHost

if TYPE_CHECKING:
    from fastapi import FastAPI

MIC_BLOCKED_MESSAGE = (
    "No microphone audio is arriving. Allow microphone access for this page (the padlock or "
    "camera icon in the address bar), check that Windows lets desktop apps use the microphone "
    "(Settings > Privacy > Microphone), close other programs using it, then press Stop and Start."
)
_FRAME_S = 0.02
_IDLE_SLEEP_S = 0.01
# More than one so a reloaded tab can connect before the old peer connection has timed out; the host
# still keeps a single active conversation (the newest connection wins).
MAX_CONNECTIONS = 3
RELAY_TTL_S = 86_400  # relay credentials are fetched once at start-up; restart after a day


class VoiceHandler(AsyncStreamHandler):
    def __init__(
        self,
        host: ConversationHost,
        output_rate: int,
        mic_timeout_s: float = 5.0,
        on_mic: Callable[[int, np.ndarray], None] | None = None,
    ) -> None:
        super().__init__(
            expected_layout="mono", output_sample_rate=output_rate, input_sample_rate=48000
        )
        self._host = host
        self._output_rate = output_rate
        self._mic_timeout_s = mic_timeout_s
        self._on_mic = on_mic
        self._started_at = 0.0
        self._heard_mic = False
        self._next_due = 0.0

    def copy(self) -> Self:
        return type(self)(self._host, self._output_rate, self._mic_timeout_s, self._on_mic)

    async def start_up(self) -> None:
        self._started_at = time.monotonic()
        self._heard_mic = False
        await self._host.open_session(self)

    def shutdown(self) -> None:
        loop = self._loop if hasattr(self, "_loop") else None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._host.close_session(self)))

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        rate, samples = frame
        self._heard_mic = True
        samples = np.asarray(samples).reshape(-1)
        if self._on_mic is not None:
            self._on_mic(rate, samples)
        session = self._host.session_for(self)
        if session is not None:
            session.feed_mic(samples, rate)

    async def emit(self) -> tuple[int, np.ndarray] | None:
        if not self._heard_mic and time.monotonic() - self._started_at > self._mic_timeout_s:
            self._started_at = time.monotonic()
            raise WebRTCError(MIC_BLOCKED_MESSAGE)
        session = self._host.session_for(self)
        # FastRTC's outgoing queue is unbounded and drains as fast as emit returns, so pace the
        # frames in real time; otherwise a barge-in could not clear audio already handed over.
        # The frame is taken only after the wait, so a barge-in during it still clears that frame.
        # (Windows timers can wake one clock tick early; the schedule, not the wake time, sets the pace.)
        if session is None:
            await asyncio.sleep(_IDLE_SLEEP_S)
            return None
        now = time.monotonic()
        if now < self._next_due:
            await asyncio.sleep(self._next_due - now)
        frame = session.next_playout_frame()
        if frame is None:
            self._next_due = 0.0
            await asyncio.sleep(_IDLE_SLEEP_S)
            return None
        self._next_due = max(self._next_due, time.monotonic()) + _FRAME_S
        pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)
        return self._output_rate, pcm


def relay_ice_servers(hf_token: str, ttl_s: int = RELAY_TTL_S) -> list[dict]:
    """Cloudflare TURN servers (fetched with a Hugging Face token), so the audio can reach a phone on
    another network; a tunnel carries only the page and the signalling, never the WebRTC media."""
    servers = (get_cloudflare_turn_credentials(hf_token=hf_token, ttl=ttl_s) or {}).get("iceServers")
    if not servers:
        raise RuntimeError("the TURN service returned no servers")
    return list(servers)


class RtcLink:
    """The FastRTC Stream behind our page: signalling routes plus a way to hang up one connection."""

    def __init__(self, handler: VoiceHandler, ice_servers: list[dict] | None = None) -> None:
        # Without a relay: no ICE servers at all. The page is served from this machine (or the LAN),
        # where host candidates connect, and a STUN lookup would only delay every answer.
        self.ice_servers = list(ice_servers or [])
        self._stream = Stream(
            handler, mode="send-receive", modality="audio", concurrency_limit=MAX_CONNECTIONS,
            time_limit=None, server_rtc_configuration={"iceServers": self.ice_servers}, verbose=False,
        )

    def mount(self, app: FastAPI) -> None:
        self._stream.mount(app)

    @property
    def connections(self) -> int:
        return len(self._stream.pcs)

    async def close(self, webrtc_id: str) -> bool:
        """Close one peer connection; FastRTC then shuts its handler down, which ends the session."""
        pc = self._stream.pcs.get(webrtc_id)
        if pc is None:
            return False
        await pc.close()
        return True
