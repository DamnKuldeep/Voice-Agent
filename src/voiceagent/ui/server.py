"""The web app: our page and static files, a small JSON API and the event stream.

Endpoints that block (applying settings, reloading models, resetting a session) are plain ``def`` so
FastAPI runs them in its thread pool: they wait on the session loop, which is this server's loop.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import controls
from .event_hub import events_after, sse_stream, to_json
from .host import BusyError, SessionHost
from .utterance_store import save_utterance

WEB_DIR = Path(__file__).resolve().parent / "web"
_NO_CACHE = {"Cache-Control": "no-cache"}
_KEEPALIVE = '{"type":"keepalive"}'  # also how a closed socket is noticed while nothing happens

VramReader = Callable[[], dict[str, float]]


class Mountable(Protocol):
    def mount(self, app: FastAPI) -> None: ...


class Rtc(Protocol):
    ice_servers: list[dict]

    def mount(self, app: FastAPI) -> None: ...

    async def close(self, webrtc_id: str) -> bool: ...


class ApplyBody(BaseModel):
    values: dict[str, Any]


class SaveBody(BaseModel):
    text: str


class StopBody(BaseModel):
    webrtc_id: str


def _nvidia_smi_vram() -> dict[str, float]:
    from ..gpu import device_memory_mib

    return device_memory_mib()


def vram_info(reader: VramReader) -> dict[str, float] | None:
    try:
        mem = reader()
        return {"used_mib": mem["used"], "total_mib": mem["total"]}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, KeyError):
        return None  # no GPU reading (no nvidia-smi, or a reader with nothing to report)


def create_app(host: SessionHost, rtc: Rtc | None = None,
               vram: VramReader = _nvidia_smi_vram, ws: Mountable | None = None) -> FastAPI:
    app = FastAPI(title="Voice agent", docs_url=None, redoc_url=None, openapi_url=None)
    for transport in (rtc, ws):
        if transport is not None:
            transport.mount(app)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", headers=_NO_CACHE)

    @app.get("/api/ice")
    def ice() -> dict:
        """ICE servers for the page's RTCPeerConnection: a TURN relay when the link is shared."""
        return {"iceServers": rtc.ice_servers if rtc is not None else []}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {**host.status_event(), "vram": vram_info(vram)}

    @app.get("/api/controls")
    def get_controls() -> dict[str, Any]:
        return controls.definitions(host.settings, host.defaults)

    @app.post("/api/controls")
    def post_controls(body: ApplyBody) -> dict[str, Any]:
        try:
            result = host.apply(body.values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except BusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {**result, **controls.definitions(host.settings, host.defaults)}

    @app.post("/api/load")
    def retry_load() -> dict[str, Any]:
        """Retry after a failed model load (for example the GPU was busy)."""
        if host.phase != "error":
            raise HTTPException(status_code=409, detail=f"Models are {host.phase}")
        threading.Thread(target=host.load, name="model-loader", daemon=True).start()
        return {"message": "Loading models again"}

    @app.post("/api/reset")
    def reset() -> dict[str, str]:
        return {"message": host.reset_conversation()}

    @app.post("/api/save")
    def save(body: SaveBody) -> dict[str, str]:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="Type what you actually said first")
        try:
            return {"id": save_utterance(host.last_utterance, body.text)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stop")
    async def stop(body: StopBody) -> dict[str, bool]:
        return {"closed": rtc is not None and await rtc.close(body.webrtc_id)}

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        return StreamingResponse(
            sse_stream(host.hub, host.status_event()), media_type="text/event-stream",
            headers={**_NO_CACHE, "X-Accel-Buffering": "no"})

    # The same events over a WebSocket: Cloudflare's quick tunnel buffered the event stream, so the
    # shared link showed no transcript or state, while WebSocket messages pass at once.
    @app.websocket("/ws/events")
    async def events_socket(ws: WebSocket) -> None:
        await ws.accept()
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            async with contextlib.aclosing(events_after(host.hub, host.status_event())) as events:
                async for event in events:
                    await ws.send_text(_KEEPALIVE if event is None else to_json(event))

    return app
