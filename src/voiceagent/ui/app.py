"""Entry point: ``python -m voiceagent.ui.app [--config PATH] [--smoke] [--share] [--host H] [--port P]``."""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path

import numpy as np

from ..config import DEFAULT_CONFIG
from .host import SessionHost

SMOKE_QUESTION = "What is the capital of France?"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="voiceagent.ui.app", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true",
                        help="load models, run one spoken question offline, print timings, exit")
    parser.add_argument("--share", action="store_true",
                        help="public HTTPS link through a Cloudflare tunnel, audio over WebSocket "
                             "(for a phone); same as ui.share: true")
    parser.add_argument("--host", default=None, help="overrides ui.host")
    parser.add_argument("--port", type=int, default=None, help="overrides ui.port")
    return parser.parse_args(argv)


async def run_smoke(host: SessionHost) -> int:
    """One scripted turn: Kokoro speaks a question into the session, which should answer aloud."""
    from ..audio.frames import resample_to_16k
    from ..pipeline.session import ConversationSession
    from ..transport.offline import replay
    components = host.components
    assert components is not None
    pcm = np.concatenate(list(components.tts.synthesize(SMOKE_QUESTION)))
    mic = resample_to_16k(pcm, components.tts.sample_rate)
    session = ConversationSession(host.settings, components)
    await session.start()
    try:
        frames = await replay(session, mic, 16000)
    finally:
        await session.stop()
        # The HTTP client belongs to this loop; closing it from the fresh loop Components.close() would
        # have to start fails with "Event loop is closed".
        await components.chat.aclose()
    if not session.traces or not frames:
        print("smoke FAILED: no completed turn or no reply audio")
        return 1
    for trace in session.traces:
        print(f"turn {trace.turn_id}: " + ", ".join(f"{k}={v:.0f}" for k, v in trace.derived().items()))
    print(f"reply audio: {len(frames) * 0.02:.1f} s in {len(frames)} frames")
    return 0


def relay_servers() -> list[dict]:
    """TURN servers for the shared link, or none (then the phone must be on the same Wi-Fi)."""
    import os

    from dotenv import load_dotenv

    from ..config import ROOT
    from ..transport.fastrtc_handler import relay_ice_servers

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("No HF_TOKEN in .env: no audio relay, so the phone must be on the same Wi-Fi as this laptop.")
        return []
    try:
        return relay_ice_servers(token)
    except Exception as exc:  # noqa: BLE001  the link still works on the same Wi-Fi
        print(f"Audio relay unavailable ({type(exc).__name__}); the phone must be on the same Wi-Fi.")
        return []


def serve(host: SessionHost, address: str, port: int, share: bool) -> None:
    """Load the models in the background and serve the page until Ctrl+C."""
    import uvicorn

    from ..transport.fastrtc_handler import RtcLink, VoiceHandler
    from ..transport.websocket_audio import WsAudioLink
    from .server import create_app
    from .tunnel import QuickTunnel, find_cloudflared

    tunnel = None
    if share:
        exe = find_cloudflared()
        if exe is None:
            raise SystemExit(
                r"--share needs cloudflared: put cloudflared.exe next to .venv\Scripts\python.exe or on "
                "PATH (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)")
    # A TURN relay matters only when the shared page is forced onto WebRTC; by default it uses the
    # WebSocket transport, whose audio travels inside the tunnel.
    ice = relay_servers() if share and host.settings.ui.transport == "webrtc" else []
    handler = VoiceHandler(host, host.settings.audio.output_rate, on_mic=host.tap_mic)
    app = create_app(host, RtcLink(handler, ice), ws=WsAudioLink(host, on_mic=host.tap_mic))
    threading.Thread(target=host.load, name="model-loader", daemon=True).start()
    print(f"Voice agent: open http://{address}:{port} in Chrome or Edge (Ctrl+C to quit)")
    if share:
        tunnel = QuickTunnel(exe, port)
        url = tunnel.start()
        print(f"Shared link (HTTPS, works on a phone): {url}")
        print("Anyone with this link can talk to the agent and spend API credit; it ends when you quit.")
        print("Audio goes through the same link (WebSocket), so it works from any network.")
    try:
        # Open event streams never end by themselves, so do not wait long for them on shutdown.
        uvicorn.run(app, host=address, port=port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        if tunnel is not None:
            tunnel.stop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    host = SessionHost(args.config)
    if args.smoke:
        host.load()
        if not host.ready.is_set():
            print(f"smoke FAILED: {host.status}")
            return 1
        try:
            return asyncio.run(run_smoke(host))
        except Exception as exc:  # noqa: BLE001 - any failure must turn into a non-zero exit
            print(f"smoke FAILED: {type(exc).__name__}: {exc}")
            return 1
        finally:
            if host.components is not None:
                host.components.close()
    ui = host.settings.ui
    try:
        serve(host, args.host or ui.host, args.port or ui.port, share=args.share or ui.share)
    except ImportError as exc:
        print(f"The UI needs FastAPI, uvicorn and FastRTC ({exc}). Install with: pip install -e \".[ui]\"")
        return 2
    finally:
        if host.components is not None:
            host.components.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
