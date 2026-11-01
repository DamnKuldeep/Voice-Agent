"""Cloudflare quick tunnel: a temporary public HTTPS address for the page, for trying it on a phone.

Phones allow the microphone only on HTTPS pages. The tunnel needs no account; its address changes on
every start and stops working when the app exits. It carries the page, the API and, with the default
WebSocket transport, the audio too; only a page forced onto WebRTC sends audio outside it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def find_cloudflared() -> str | None:
    """On PATH, or next to the virtualenv's Python (``.venv\\Scripts\\cloudflared.exe`` on Windows)."""
    found = shutil.which("cloudflared")
    if found:
        return found
    local = Path(sys.executable).parent / ("cloudflared.exe" if sys.platform == "win32" else "cloudflared")
    return str(local) if local.exists() else None


class QuickTunnel:
    def __init__(self, exe: str, port: int) -> None:
        self._cmd = [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"]
        self._proc: subprocess.Popen[str] | None = None
        self._found = threading.Event()
        self.url: str | None = None

    def start(self, timeout_s: float = 45.0) -> str:
        self._proc = subprocess.Popen(self._cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                      text=True, encoding="utf-8", errors="replace")
        threading.Thread(target=self._read_log, name="cloudflared-log", daemon=True).start()
        if not self._found.wait(timeout_s) or self.url is None:
            self.stop()
            raise RuntimeError("cloudflared did not report a public address in time")
        return self.url

    def _read_log(self) -> None:
        # Keep draining after the address is found, or cloudflared blocks on a full pipe.
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            if self.url is None and (match := _URL.search(line)):
                self.url = match.group(0)
                self._found.set()
        self._found.set()

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
