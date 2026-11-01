"""Microphone intake: any rate and layout in, exact 16 kHz VAD frames out, in order, on the event loop.

``feed`` is called from the transport's thread; frames cross to the loop through a bounded queue, so a
loop that falls far behind drops audio instead of growing without limit.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

import numpy as np

from ..audio.frames import FrameCutter, resample_to_16k

log = logging.getLogger(__name__)

FRAME_QUEUE_SIZE = 400   # about 13 s of 32 ms frames
LEVEL_EVERY = 3          # frames between level events (about 10 per second)


class MicInput:
    def __init__(
        self, frame_samples: int, on_frame: Callable[[np.ndarray], None], emit: Callable[[dict], None],
    ) -> None:
        self._cutter = FrameCutter(frame_samples)
        self._cut_lock = threading.Lock()
        self._on_frame, self._emit = on_frame, emit
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frames: asyncio.Queue[np.ndarray] = asyncio.Queue(FRAME_QUEUE_SIZE)
        self._task: asyncio.Task[None] | None = None
        self._count = 0
        self.dropped_frames = 0

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        self._loop = None
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait({self._task})
            self._task = None

    def feed(self, samples: np.ndarray, rate: int) -> None:
        loop = self._loop
        if loop is None or not len(samples):
            return
        mono = resample_to_16k(samples, rate)
        with self._cut_lock:
            frames = self._cutter.push(mono)
        for frame in frames:
            try:
                loop.call_soon_threadsafe(self._enqueue, frame)
            except RuntimeError:  # the loop closed between the check above and now
                return

    def _enqueue(self, frame: np.ndarray) -> None:
        try:
            self._frames.put_nowait(frame)
        except asyncio.QueueFull:
            self.dropped_frames += 1

    async def _run(self) -> None:
        while True:
            frame = await self._frames.get()
            self._count += 1
            if self._count % LEVEL_EVERY == 0:
                self._emit({"type": "level", "rms": float(np.sqrt(np.mean(frame * frame)))})
            try:
                self._on_frame(frame)
            except Exception:
                log.exception("frame handling failed")
