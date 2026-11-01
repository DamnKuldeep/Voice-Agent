"""A SessionHost wired to fake models and a fake session, shared by the UI tests."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

from voiceagent.ui.event_hub import EventHub
from voiceagent.ui.host import SessionHost


class FakeSession:
    def __init__(self) -> None:
        # 3 s of a 440 Hz tone at 24 kHz: long enough to outlast connection set-up, and unlike a constant
        # it survives Opus, whose encoder removes DC.
        t = np.arange(150 * 480, dtype=np.float32) / 24000
        self.frames = list((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).reshape(150, 480))
        self.fed: list[tuple[int, int]] = []
        self.listener = None
        self.started = self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def feed_mic(self, samples: np.ndarray, rate: int) -> None:
        self.fed.append((rate, samples.shape[0]))

    def next_playout_frame(self) -> np.ndarray | None:
        return self.frames.pop(0) if self.frames else None

    def set_listener(self, fn) -> None:
        self.listener = fn

    def summary(self) -> dict:
        return {"v2v_ms": {"n": 1.0, "mean": 1.0, "p50": 1.0, "p90": 1.0, "max": 1.0}}


class RecordingHub(EventHub):
    def __init__(self, sink: Callable[[dict[str, Any]], None]) -> None:
        super().__init__()
        self._sink = sink

    def publish(self, event: dict[str, Any]) -> None:
        self._sink(event)
        super().publish(event)


def fake_components() -> SimpleNamespace:
    return SimpleNamespace(close=lambda: None, turn_detector=object(), fillers=[])


def make_host(
    build: Callable[..., Any] | None = None,
    publish: Callable[[dict[str, Any]], None] | None = None,
    load: bool = True,
) -> tuple[SessionHost, list[int]]:
    builds: list[int] = []

    def counting_build(_settings, _progress):
        builds.append(1)
        return fake_components()

    hub = RecordingHub(publish) if publish is not None else EventHub()
    host = SessionHost(build=build or counting_build, make_session=lambda s, c: FakeSession(), hub=hub)
    if load:
        host.load()
    return host, builds
