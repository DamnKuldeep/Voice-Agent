"""Scripted stand-ins for the VAD, ASR, LLM, TTS and turn model, plus a driver that plays audio into a session."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from voiceagent.config import Settings, load_settings
from voiceagent.llm.call_result import CallResult, ChatStream
from voiceagent.pipeline.factory import Components
from voiceagent.pipeline.session import ConversationSession

RATE = 16000
STEP_MS = 20
LONG_REPLY = (
    "This is a long answer about many different things. It keeps going and going for quite a while. "
    "Here is a third sentence to make it longer still. And a fourth sentence closes it out."
)


class FakeVAD:
    def reset(self) -> None:
        pass

    def process(self, frame: np.ndarray) -> float:
        return 0.9 if float(np.abs(frame).max()) > 0.1 else 0.05


class FakeASR:
    """Reveals one word per 160 ms of audio pushed; ``finish`` returns the next scripted transcript."""

    def __init__(self, transcripts: list[str], finals: list[str] | None = None) -> None:
        self.transcripts = list(transcripts)
        self.finals = finals
        self.fail_finish = False
        self.pushes: list[int] = []
        self.audio: list[np.ndarray] = []   # everything pushed, in order
        self.resets = 0
        self.push_delay_s = 0.0
        self.finish_delay_s = 0.0
        self.samples = 0

    def reset(self) -> None:
        self.samples = 0
        self.resets += 1

    def push(self, audio: np.ndarray) -> str:
        time.sleep(self.push_delay_s)
        self.samples += len(audio)
        self.pushes.append(len(audio))
        self.audio.append(audio.copy())
        words = (self.transcripts[0] if self.transcripts else "").split()
        return " ".join(words[: self.samples // 2560])

    def finish(self) -> str:
        time.sleep(self.finish_delay_s)
        if self.fail_finish:
            raise RuntimeError("asr broke")
        if self.samples == 0:
            return ""
        text = self.finals.pop(0) if self.finals else ""
        script = self.transcripts.pop(0) if self.transcripts else ""
        self.samples = 0
        return text or script


class FakeTurnModel:
    def __init__(self, prob: float) -> None:
        self.prob = prob

    def end_of_turn_probability(self, audio: np.ndarray) -> float:
        return self.prob


class FakeChat:
    """Streams ``reply`` word by word; like the OpenRouter client, each stream carries its own result."""

    def __init__(self, reply: str = "Paris is the capital of France.", delay: float = 0.0,
                 fail: bool = False, finish_reason: str = "stop", per_delta: float = 0.002) -> None:
        self.reply, self.delay, self.fail = reply, delay, fail
        self.per_delta = per_delta
        self.finish_reason = finish_reason
        self.calls: list[list[dict[str, str]]] = []
        self.cancels: list[asyncio.Event] = []

    def stream(self, messages, *, max_tokens, cancel) -> ChatStream:
        result = CallResult(model="fake/model", finish_reason=self.finish_reason,
                            usage={"prompt_tokens": 10, "completion_tokens": 7})
        return ChatStream(self._deltas(messages, cancel), result)

    async def _deltas(self, messages, cancel):
        self.calls.append(messages)
        self.cancels.append(cancel)
        if self.fail:
            raise RuntimeError("llm down")
        await asyncio.sleep(self.delay)
        for word in self.reply.split(" "):
            if cancel.is_set():
                return
            yield word + " "
            await asyncio.sleep(self.per_delta)


class FakeTTS:
    sample_rate = 24000

    def __init__(self, ms_per_word: int = 100, fail: bool = False) -> None:
        self.ms_per_word = ms_per_word
        self.fail = fail
        self.texts: list[str] = []

    def warmup(self) -> None:
        pass

    def synthesize(self, text: str):
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("tts broke")
        for _ in text.split():
            yield np.full(24 * self.ms_per_word, 0.2, dtype=np.float32)
            time.sleep(0.001)


def make_settings(**sections: dict[str, Any]) -> Settings:
    return load_settings(overrides=sections or None)


def make_components(
    *, transcripts: list[str] | None = None, finals: list[str] | None = None, chat: FakeChat | None = None,
    tts: FakeTTS | None = None, turn_prob: float | None = None,
) -> Components:
    return Components(
        vad=FakeVAD(),
        turn_detector=FakeTurnModel(turn_prob) if turn_prob is not None else None,
        asr=FakeASR(transcripts if transcripts is not None else ["what is the capital of france"], finals),
        chat=chat or FakeChat(),
        tts=tts or FakeTTS(),
    )


def speech(seconds: float) -> np.ndarray:
    return np.full(int(seconds * RATE), 0.5, dtype=np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


class Driver:
    """Feeds audio in 20 ms steps (faster than real time) and pulls output frames at the same cadence."""

    def __init__(self, session: ConversationSession, step_s: float = 0.003) -> None:
        self.session = session
        self.step_s = step_s
        self.fed_ms = 0
        self.output: list[np.ndarray] = []
        self.events: list[tuple[int, dict]] = []
        session.set_listener(lambda event: self.events.append((self.fed_ms, event)))

    async def feed(self, audio: np.ndarray) -> None:
        step = RATE * STEP_MS // 1000
        for start in range(0, len(audio), step):
            self.session.feed_mic(audio[start:start + step], RATE)
            frame = self.session.next_playout_frame()
            if frame is not None:
                self.output.append(frame)
            self.fed_ms += STEP_MS
            await asyncio.sleep(self.step_s)

    async def until(self, condition: Callable[[], bool], timeout_s: float = 15.0) -> None:
        """Feed silence until ``condition`` holds."""
        deadline = time.perf_counter() + timeout_s
        while not condition():
            assert time.perf_counter() < deadline, f"timed out in state {self.session.state}"
            await self.feed(silence(0.1))

    def states(self) -> list[str]:
        return [e["state"] for _, e in self.events if e["type"] == "state"]

    def at(self, state: str) -> int:
        return next(t for t, e in self.events if e["type"] == "state" and e["state"] == state)

    def of_type(self, kind: str) -> list[dict]:
        return [e for _, e in self.events if e["type"] == kind]


class FakeAsrBackend:
    """Constructor-compatible stand-in for a real ASR adapter, used to test backend selection."""

    def __init__(self, settings: Any, device: str) -> None:
        self.device = device
