"""The web app with a fake host (no models, no GPU): API, static files, SSE, a real WebRTC call."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ui_fakes import make_host
from voiceagent.ui.server import WEB_DIR, create_app

ASSETS = ["/static/style.css", "/static/js/app.js", "/static/js/orb.js", "/static/js/levels.js",
          "/static/js/rtc.js", "/static/js/transcript.js", "/static/js/settings.js",
          "/static/js/perf.js", "/static/js/ws.js", "/static/js/worklets.js"]


class FakeRtc:
    def __init__(self) -> None:
        self.closed: list[str] = []
        self.ice_servers = [{"urls": ["turn:relay.example:3478"], "username": "u", "credential": "c"}]

    def mount(self, app) -> None:
        pass

    async def close(self, webrtc_id: str) -> bool:
        self.closed.append(webrtc_id)
        return webrtc_id == "known"


@pytest.fixture
def api():
    host, builds = make_host()
    rtc = FakeRtc()
    with TestClient(create_app(host, rtc, vram=lambda: {"used": 2048.0, "total": 4096.0})) as client:
        yield client, host, rtc, builds


def test_page_and_every_asset_is_served(api):
    client, *_ = api
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    referenced = set(re.findall(r'(?:href|src)="(/static/[^"]+)"', page.text))
    assert referenced <= set(ASSETS)
    for path in ASSETS:
        res = client.get(path)
        assert res.status_code == 200, path
    assert "javascript" in client.get("/static/js/app.js").headers["content-type"]
    imports = set(re.findall(r'from "\./([a-z]+\.js)"', (WEB_DIR / "js" / "app.js").read_text("utf-8")))
    assert {f"/static/js/{name}" for name in imports} <= set(ASSETS)


def test_web_files_need_no_external_network():
    for path in WEB_DIR.rglob("*.*"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?:src|href)=\"https?://|@import url\(|https?://(?!www\.w3\.org)", text), path


def test_status_includes_vram_and_config(api):
    client, host, *_ = api
    body = client.get("/api/status").json()
    assert body["type"] == "status" and body["phase"] == "ready" and body["ready"]
    assert body["vram"] == {"used_mib": 2048.0, "total_mib": 4096.0}
    assert body["config"]["llm_model"] == host.settings.llm.model


def test_status_without_nvidia_smi():
    host, _ = make_host()

    def broken():
        raise FileNotFoundError("nvidia-smi")

    with TestClient(create_app(host, vram=broken)) as client:
        assert client.get("/api/status").json()["vram"] is None


def test_controls_get_and_apply(api):
    client, host, _, builds = api
    defs = client.get("/api/controls").json()
    assert {g["id"] for g in defs["groups"]} >= {"listening", "interruptions", "advanced"}
    assert any(c["key"] == "barge_in.echo_guard" for c in defs["controls"])
    res = client.post("/api/controls", json={"values": {"barge_in.duck_gain": 0.5}})
    body = res.json()
    assert res.status_code == 200 and body["changed"] == ["barge_in.duck_gain"] and not body["reloaded"]
    assert next(c for c in body["controls"] if c["key"] == "barge_in.duck_gain")["value"] == 0.5
    res = client.post("/api/controls", json={"values": {"asr.chunk_ms": "80"}})
    assert res.json()["reloaded"] and len(builds) == 2 and host.settings.barge_in.duck_gain == 0.5
    bad = client.post("/api/controls", json={"values": {"barge_in.duck_gain": 3}})
    assert bad.status_code == 400 and "outside" in bad.json()["detail"]


def test_apply_while_busy_is_refused(api):
    client, host, *_ = api
    assert host._apply_lock.acquire()
    try:
        assert client.post("/api/controls", json={"values": {}}).status_code == 409
    finally:
        host._apply_lock.release()


def test_reset_save_stop_and_load(api):
    client, _, rtc, _ = api
    assert client.post("/api/reset").json()["message"] == "Conversation reset"
    assert client.post("/api/save", json={"text": "  "}).status_code == 400
    res = client.post("/api/save", json={"text": "hello"})
    assert res.status_code == 400 and "no utterance" in res.json()["detail"]
    assert client.post("/api/stop", json={"webrtc_id": "known"}).json() == {"closed": True}
    assert client.post("/api/stop", json={"webrtc_id": "gone"}).json() == {"closed": False}
    assert rtc.closed == ["known", "gone"]
    assert client.post("/api/load").status_code == 409


def test_retry_load_after_failure():
    attempts: list[int] = []

    def build(_settings, _progress):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        from ui_fakes import fake_components
        return fake_components()

    host, _ = make_host(build=build)
    with TestClient(create_app(host, vram=dict)) as client:
        assert client.get("/api/status").json()["phase"] == "error"
        assert client.post("/api/load").status_code == 200
        assert host.ready.wait(5) and host.phase == "ready"


# A real server on a spare port ---------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    import uvicorn

    from voiceagent.transport.fastrtc_handler import RtcLink, VoiceHandler

    host, _ = make_host()
    handler = VoiceHandler(host, host.settings.audio.output_rate, on_mic=host.tap_mic)
    app = create_app(host, RtcLink(handler), vram=lambda: {"used": 1.0, "total": 2.0})
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           timeout_graceful_shutdown=1))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}", host
    server.should_exit = True
    thread.join(10)


def test_live_server_serves_page_and_assets(live_server):
    base, _ = live_server
    with httpx.Client(base_url=base, timeout=5) as client:
        assert client.get("/").status_code == 200
        for path in ASSETS:
            assert client.get(path).status_code == 200, path
        assert client.get("/api/status").json()["ready"]


def test_live_server_streams_events(live_server):
    base, host = live_server

    async def read_two() -> list[dict]:
        events: list[dict] = []
        async with (httpx.AsyncClient(base_url=base, timeout=5) as client,
                    client.stream("GET", "/api/events") as res):
                assert res.headers["content-type"].startswith("text/event-stream")
                async for line in res.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    events.append(json.loads(line[6:]))
                    if len(events) == 1:
                        host.hub.publish({"type": "progress", "message": "from another thread"})
                    else:
                        return events
        return events

    events = asyncio.run(asyncio.wait_for(read_two(), 10))
    assert events[0]["type"] == "status" and events[1]["message"] == "from another thread"


def test_live_webrtc_call_carries_audio_both_ways(live_server):
    """A Python WebRTC peer stands in for the browser: offer, mic frames up, agent frames down."""
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import AudioStreamTrack

    base, host = live_server

    async def call() -> tuple[int, int, bool]:
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pc.createDataChannel("text")
        pc.addTrack(AudioStreamTrack())
        got: list[int] = []

        @pc.on("track")
        def on_track(track):
            async def pull():
                while True:
                    frame = await track.recv()
                    if frame.to_ndarray().any():
                        got.append(frame.samples)
            asyncio.ensure_future(pull())

        await pc.setLocalDescription(await pc.createOffer())
        async with httpx.AsyncClient(base_url=base, timeout=10) as client:
            res = await client.post("/webrtc/offer", json={
                "sdp": pc.localDescription.sdp, "type": "offer", "webrtc_id": "py-peer"})
            await pc.setRemoteDescription(RTCSessionDescription(**res.json()))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                session = host.session_for(host._owner)
                if session is not None and session.fed and got:
                    break
                await asyncio.sleep(0.05)
            fed = len(session.fed) if session is not None else 0
            closed = (await client.post("/api/stop", json={"webrtc_id": "py-peer"})).json()["closed"]
            for _ in range(100):
                if not host.active:
                    break
                await asyncio.sleep(0.05)
        await pc.close()
        return fed, len(got), closed

    fed, received, closed = asyncio.run(call())
    assert fed > 0, "no mic audio reached the session"
    assert received > 0, "no agent audio came back"
    assert closed and not host.active


# JavaScript syntax --------------------------------------------------------------------------------

def _balanced(source: str) -> bool:
    """Rough fallback when node is missing: brackets balance once strings and comments are removed."""
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/|`(?:\\.|[^`])*`|\"(?:\\.|[^\"\n])*\"|'(?:\\.|[^'\n])*'",
                      "", source, flags=re.DOTALL)
    pairs, stack = {")": "(", "]": "[", "}": "{"}, []
    for ch in stripped:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return False
    return not stack


@pytest.mark.parametrize("path", sorted((WEB_DIR / "js").glob("*.js")), ids=lambda p: p.name)
def test_javascript_parses(path: Path, tmp_path):
    source = path.read_text(encoding="utf-8")
    assert _balanced(source), path.name
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed: only the bracket-balance check ran")
    module = tmp_path / (path.stem + ".mjs")
    module.write_text(source, encoding="utf-8")
    result = subprocess.run([node, "--check", str(module)], capture_output=True, text=True, timeout=30,
                            check=False)
    assert result.returncode == 0, result.stderr


def test_ice_servers_reach_the_page(api):
    client, *_ = api
    assert client.get("/api/ice").json()["iceServers"][0]["urls"] == ["turn:relay.example:3478"]



def test_websocket_audio_carries_mic_up_and_agent_frames_down():
    import numpy as np

    from voiceagent.transport.websocket_audio import PATH, WsAudioLink

    host, _ = make_host()
    tapped: list[int] = []
    app = create_app(host, ws=WsAudioLink(host, on_mic=lambda rate, s: tapped.append(rate)))
    with TestClient(app) as client, client.websocket_connect(PATH) as ws:
        ws.send_bytes(np.zeros(320, dtype="<i2").tobytes())
        frame = np.frombuffer(ws.receive_bytes(), dtype="<i2")
        assert frame.shape == (480,) and np.abs(frame).max() > 0   # 20 ms at 24 kHz of the fake tone
        session = host.session_for(host._owner)
        assert session is not None and session.fed[0] == (16000, 320) and tapped == [16000]
    assert not host.active                                           # closing the socket ends it


def test_websocket_audio_reports_models_not_ready():
    from voiceagent.transport.websocket_audio import PATH, WsAudioLink

    def failing(_settings, _progress):
        raise RuntimeError("CUDA out of memory")

    host, _ = make_host(build=failing)
    with TestClient(create_app(host, ws=WsAudioLink(host))) as client, client.websocket_connect(PATH) as ws:
        assert ws.receive_json()["type"] == "error"


def test_events_websocket_sends_status_then_published_events(api):
    client, host, *_ = api
    with client.websocket_connect("/ws/events") as ws:
        assert ws.receive_json()["type"] == "status"
        host.hub.publish({"type": "state", "state": "thinking"})
        assert ws.receive_json() == {"type": "state", "state": "thinking"}
