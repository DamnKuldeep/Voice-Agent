"""Runs the blocking streaming ASR off the event loop, in whole chunks, strictly in order.

Every queued item carries the generation (utterance) it belongs to. ``begin_turn`` purges everything
queued before it, and an item already in flight for an old generation has its result ignored, so a reset
clears the backlog at once instead of after the GPU has chewed through audio nobody will read.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any

import numpy as np

from ..core.interfaces import StreamingASR

log = logging.getLogger(__name__)

RATE = 16000
MAX_BACKLOG_MS = 10240   # about ten seconds of queued audio; beyond that the recogniser is hopelessly behind
CONTROL_SLOTS = 16       # room for resets and finishes on top of the audio bound
LAG_REPORT_MS = 300      # backlog worth telling the UI about
LAG_REPORT_EVERY_S = 0.5


class AsrStage:
    def __init__(
        self, asr: StreamingASR, pool: Executor, chunk_ms: int,
        on_partial: Callable[[str], None], on_error: Callable[[Exception], None],
        on_lag: Callable[[float], None] | None = None,
    ) -> None:
        self._asr, self._pool = asr, pool
        self._chunk_samples = chunk_ms * RATE // 1000
        self._on_partial, self._on_error, self._on_lag = on_partial, on_error, on_lag
        max_chunks = max(1, MAX_BACKLOG_MS // chunk_ms)
        self._max_backlog = max_chunks * self._chunk_samples
        self._queue: asyncio.Queue[tuple[str, Any, asyncio.Future[str] | None, int]] = asyncio.Queue(
            max_chunks + CONTROL_SLOTS)
        self._pending = np.zeros(0, dtype=np.float32)
        self._generation = 0
        self._backlog = 0          # samples of the current generation queued or being decoded
        self._max_lag_ms = 0.0
        self._lag_reported_at: float | None = None
        self._worker: asyncio.Task[None] | None = None
        self.dropped_chunks = 0
        self.skipped_chunks = 0

    @property
    def backlog_ms(self) -> float:
        """Audio handed to the recogniser that it has not decoded yet."""
        return self._backlog * 1000 / RATE

    def take_max_lag_ms(self) -> float:
        """Largest backlog since the last call."""
        value, self._max_lag_ms = self._max_lag_ms, self.backlog_ms
        return value

    def start(self) -> None:
        self._worker = asyncio.ensure_future(self._work())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.wait({self._worker})
            self._worker = None
        self._purge()
        await asyncio.get_running_loop().run_in_executor(self._pool, self._asr.reset)

    def begin_turn(self) -> None:
        """Discard whatever the recogniser holds and start a fresh utterance."""
        self._generation += 1
        self._pending = np.zeros(0, dtype=np.float32)
        self._backlog = 0
        self._purge()
        self._report_lag()
        self._enqueue("reset", None, None)

    def feed(self, audio: np.ndarray) -> None:
        self._pending = np.concatenate([self._pending, audio])
        while len(self._pending) >= self._chunk_samples:
            chunk, self._pending = self._pending[:self._chunk_samples], self._pending[self._chunk_samples:]
            self._push(chunk)

    async def finish(self) -> str:
        if len(self._pending):
            self._push(self._pending)
            self._pending = np.zeros(0, dtype=np.float32)
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._enqueue("finish", None, future)
        return await future

    def _push(self, chunk: np.ndarray) -> None:
        if self._backlog + len(chunk) > self._max_backlog:
            self.dropped_chunks += 1
            log.warning("ASR backlog above %d ms, dropping a chunk", MAX_BACKLOG_MS)
            return
        if self._enqueue("push", chunk, None):
            self._backlog += len(chunk)
            self._report_lag()

    def _enqueue(self, kind: str, payload: Any, future: asyncio.Future[str] | None) -> bool:
        try:
            self._queue.put_nowait((kind, payload, future, self._generation))
        except asyncio.QueueFull:
            self.dropped_chunks += 1
            log.warning("ASR queue full, dropping a %s", kind)
            if future is not None:
                future.set_exception(RuntimeError("ASR queue full"))
            return False
        return True

    def _purge(self) -> None:
        """Drop everything queued: after a reset none of it can matter. A waiting finish gets ""."""
        while not self._queue.empty():
            kind, _payload, future, _generation = self._queue.get_nowait()
            if kind == "push":
                self.skipped_chunks += 1
            if future is not None and not future.done():
                future.set_result("")

    def _report_lag(self) -> None:
        lag = self.backlog_ms
        self._max_lag_ms = max(self._max_lag_ms, lag)
        if self._on_lag is None:
            return
        now = time.monotonic()
        if lag > LAG_REPORT_MS:
            if self._lag_reported_at is None or now - self._lag_reported_at >= LAG_REPORT_EVERY_S:
                self._lag_reported_at = now
                self._on_lag(lag)
        elif self._lag_reported_at is not None:
            self._lag_reported_at = None
            self._on_lag(0.0)

    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            kind, payload, future, generation = await self._queue.get()
            try:
                if kind == "push":
                    text = await loop.run_in_executor(self._pool, self._asr.push, payload)
                    if generation == self._generation:
                        self._backlog = max(0, self._backlog - len(payload))
                        self._report_lag()
                        self._on_partial(text)
                elif kind == "finish":
                    text = await loop.run_in_executor(self._pool, self._asr.finish)
                    if future is not None and not future.done():
                        future.set_result(text)
                else:
                    await loop.run_in_executor(self._pool, self._asr.reset)
            except Exception as exc:  # noqa: BLE001  reported to the session, never fatal to the worker
                log.warning("ASR %s failed: %s", kind, exc)
                if future is not None and not future.done():
                    future.set_exception(exc)
                else:
                    self._on_error(exc)
