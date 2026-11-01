"""One reply: LLM stream -> speech chunker -> TTS -> playout, with cancellation at every stage.

Three stages run as tasks joined by queues: the LLM feed emits speakable chunks, the synthesis stage
runs the TTS in its own single-thread executor (so chunk n+1 is synthesised while chunk n plays), and
the writer feeds the bounded playout queue. A held (speculative) responder runs the LLM but waits for
``release`` before any speech is synthesised. A cleared ``synth_gate`` (a possible interruption is being
recognised) holds back new syntheses while the audio already made keeps playing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from typing import Any

import numpy as np

from ..config import Settings
from ..core.events import Mark
from ..core.interfaces import ChatModel, SpeechSynthesizer
from ..core.metrics import TurnTrace
from ..llm.chunker import SpeechChunker
from .playout import Playout

log = logging.getLogger(__name__)

LLM_FAILURE_REPLY = "Sorry, I'm having trouble thinking right now. Could you ask me again?"
UNCAPPED_TOKENS = 1024  # safety bound when the length policy is switched off
WRITE_POLL_S = 0.01
_DONE = "done"


class Responder:
    def __init__(
        self, settings: Settings, chat: ChatModel, tts: SpeechSynthesizer, playout: Playout,
        pool: Executor, turn_id: int, fillers: Sequence[np.ndarray] = (),
        synth_gate: asyncio.Event | None = None,
    ) -> None:
        self._cfg, self._chat, self._tts = settings, chat, tts
        self._playout, self._pool, self._fillers = playout, pool, fillers
        self._gate = synth_gate
        self.trace = TurnTrace(turn_id)
        self.first_audio = asyncio.Event()
        self.text = ""
        self.finish_reason: str | None = None
        self.model: str | None = None
        self.usage: dict[str, Any] | None = None
        self.error: Exception | None = None
        self.tokens_used = 0
        self._released = asyncio.Event()
        self._stop = threading.Event()
        self._llm_cancel = asyncio.Event()
        self._chunks: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)
        self._audio: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()  # bounded by the prefetch slots
        self._slots = asyncio.Semaphore(settings.tts.prefetch_chunks + 1)
        self._sink: TurnTrace | None = None
        self._on_text: Callable[[str], None] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._main: asyncio.Task[None] | None = None
        self._texts: list[str] = []
        self._whole: list[str] = []  # chunks held back when tts.stream is off

    def start(self, messages: list[dict[str, str]], max_tokens: int, *, hold: bool) -> None:
        self._begin(self._feed_llm(messages, max_tokens), hold)

    def start_canned(self, text: str) -> None:
        self._begin(self._feed_text(text), hold=False)

    def adopt(self, sink: TurnTrace, on_text: Callable[[str], None]) -> None:
        """Make this the reply the user hears: mirror its marks into ``sink`` from now on."""
        self._sink, self._on_text = sink, on_text
        for name, at in self.trace.marks.items():
            sink.mark(name, at)
        if self.text:
            on_text(self.text)

    def release(self) -> None:
        self._released.set()

    @property
    def task(self) -> asyncio.Task[None]:
        assert self._main is not None
        return self._main

    async def wait(self) -> None:
        await asyncio.wait({self.task})

    def cancel(self) -> None:
        self._stop.set()
        self._llm_cancel.set()
        for task in self._all_tasks():
            task.cancel()

    async def aclose(self) -> None:
        """Cancel and wait until every stage task has really ended."""
        self.cancel()
        tasks = self._all_tasks()
        if tasks:
            await asyncio.wait(tasks)

    def _all_tasks(self) -> list[asyncio.Task[None]]:
        return [task for task in [self._main, *self._tasks] if task is not None]

    def _begin(self, feeder: Any, hold: bool) -> None:
        if not hold:
            self._released.set()
        self._tasks = [
            asyncio.ensure_future(self._guard(feeder)),
            asyncio.ensure_future(self._guard(self._synthesize())),
            asyncio.ensure_future(self._guard(self._write())),
        ]
        self._main = asyncio.ensure_future(self._join())

    async def _join(self) -> None:
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            self.cancel()
            raise

    async def _guard(self, stage: Any) -> None:
        try:
            await stage
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("reply stage failed")
            self.error = exc
            self.cancel()

    def _mark(self, name: Mark) -> None:
        self.trace.mark(name)
        if self._sink is not None:
            self._sink.mark(name, self.trace.marks[name])

    async def _emit(self, chunk: str) -> None:
        if not self._texts:
            self._mark(Mark.FIRST_CHUNK_READY)
        self._texts.append(chunk)
        self.text = " ".join(self._texts)
        if self._on_text is not None:
            self._on_text(self.text)
        await self._chunks.put(chunk)

    async def _feed_text(self, text: str) -> None:
        await self._emit(text)
        await self._chunks.put(None)

    async def _feed_llm(self, messages: list[dict[str, str]], max_tokens: int) -> None:
        chunker = SpeechChunker(self._cfg.chunker)
        streaming = self._cfg.llm.stream
        held: list[str] = []
        self._mark(Mark.LLM_REQUEST)
        stream = self._chat.stream(messages, max_tokens=max_tokens, cancel=self._llm_cancel)
        try:
            async for delta in stream:
                if not self.tokens_used:
                    self._mark(Mark.LLM_FIRST_TOKEN)
                self.tokens_used += 1  # deltas, an estimate until the provider reports usage
                if streaming:
                    for chunk in chunker.feed(delta):
                        await self._emit(chunk)
                else:
                    held.append(delta)
        except Exception as exc:  # noqa: BLE001  fall through to the spoken fallback below
            log.warning("LLM stream failed: %s", exc)
            self.error = exc
        if self._llm_cancel.is_set():
            return
        # The stream's own result, not the client's: a speculative call may overlap this one.
        if (result := getattr(stream, "result", None)) is not None:
            self.finish_reason, self.model, self.usage = result.finish_reason, result.model, result.usage
        if self.usage and self.usage.get("completion_tokens"):
            self.tokens_used = int(self.usage["completion_tokens"])
        for chunk in chunker.feed("".join(held)):
            await self._emit(chunk)
        for chunk in chunker.flush(truncated=self.finish_reason == "length" or self.error is not None):
            await self._emit(chunk)
        if not self._texts:
            await self._emit(LLM_FAILURE_REPLY)
        await self._chunks.put(None)

    async def _synthesize(self) -> None:
        loop = asyncio.get_running_loop()
        streaming = self._cfg.tts.stream
        try:
            while (text := await self._chunks.get()) is not None:
                if not streaming:
                    self._whole.append(text)
                    continue
                await self._released.wait()
                await self._slots.acquire()
                if self._gate is not None:
                    await self._gate.wait()
                await self._audio.put(("begin", text))
                await loop.run_in_executor(self._pool, self._synth_blocking, loop, text, False)
                await self._audio.put(("end", None))
            if not streaming and self._whole:
                await self._released.wait()
                if self._gate is not None:
                    await self._gate.wait()
                text = " ".join(self._whole)
                await self._audio.put(("begin", text))
                await loop.run_in_executor(self._pool, self._synth_blocking, loop, text, True)
                await self._audio.put(("end", None))
        finally:
            await self._audio.put((_DONE, None))

    def _synth_blocking(self, loop: asyncio.AbstractEventLoop, text: str, collect: bool) -> None:
        parts: list[np.ndarray] = []
        for samples in self._tts.synthesize(text):
            if self._stop.is_set():
                return
            if collect:
                parts.append(samples)
            else:
                loop.call_soon_threadsafe(self._audio.put_nowait, ("pcm", samples))
        if collect and parts:
            loop.call_soon_threadsafe(self._audio.put_nowait, ("pcm", np.concatenate(parts)))

    async def _write(self) -> None:
        await self._released.wait()
        loop = asyncio.get_running_loop()
        filler_at = loop.time() + self._cfg.tts.filler_after_ms / 1000 if self._fillers else None
        chunk_text: str | None = None
        while True:
            waiting_for_audio = filler_at is not None and not self.first_audio.is_set()
            try:
                item = await asyncio.wait_for(
                    self._audio.get(), max(0.0, filler_at - loop.time()) if waiting_for_audio else None)
            except TimeoutError:
                filler_at = None
                await self._play_filler()
                continue
            kind, payload = item
            if kind == _DONE:
                break
            if kind == "begin":
                chunk_text = payload
            elif kind == "pcm":
                if chunk_text is not None:
                    self._playout.begin_chunk(chunk_text)
                    chunk_text = None
                if not self.first_audio.is_set():
                    self._mark(Mark.TTS_FIRST_AUDIO)
                    self.first_audio.set()
                await self._push(payload)
            else:
                self._playout.end_chunk()
                self._slots.release()
        self._playout.finish()

    async def _play_filler(self) -> None:
        filler = self._fillers[self.trace.turn_id % len(self._fillers)]
        self._playout.begin_chunk("")
        await self._push(filler)
        self._playout.end_chunk()

    async def _push(self, samples: np.ndarray) -> None:
        while len(samples):
            taken = self._playout.write(samples)
            samples = samples[taken:]
            if len(samples):
                await asyncio.sleep(WRITE_POLL_S)
