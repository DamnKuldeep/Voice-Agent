"""Drive a session from an audio array instead of a browser: evals, the --smoke check and tests."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..pipeline.session import ConversationSession

STEP_MS = 20  # the cadence a WebRTC transport pushes and pulls audio


@dataclass
class Cue:
    """A clip mixed into the mic once ``when()`` first holds, ``delay_s`` later (e.g. "agent is speaking").

    The replay fills in when the clip's first sample was fed: ``fired_at`` on the perf_counter clock the
    traces use, ``fired_audio_s`` in seconds of stream fed so far.
    """

    when: Callable[[], bool]
    clip: np.ndarray
    delay_s: float = 0.0
    fired_at: float | None = None
    fired_audio_s: float | None = None
    _due_step: int | None = field(default=None, init=False, repr=False)


@dataclass(frozen=True)
class Echo:
    """Agent output fed back into the mic, as from speakers with no echo cancellation."""

    gain_db: float = -15.0
    delay_ms: float = 60.0


class _EchoLine:
    """Delay line from the playout side (any rate) to the mic side (``rate``)."""

    def __init__(self, echo: Echo, step: int) -> None:
        self._gain = float(10 ** (echo.gain_db / 20))
        self._delay = round(echo.delay_ms / STEP_MS * step)
        self._step = step
        self._buf = np.zeros(0, dtype=np.float32)
        self._read = 0  # stream position of self._buf[0]

    def add(self, frame: np.ndarray, played_at: int) -> None:
        """``frame`` is one 20 ms output frame that started playing at stream sample ``played_at``."""
        mic = np.interp(np.linspace(0, len(frame) - 1, self._step), np.arange(len(frame)), frame)
        at = max(played_at + self._delay - self._read, 0)
        if len(self._buf) < at + len(mic):
            self._buf = np.concatenate([self._buf, np.zeros(at + len(mic) - len(self._buf), np.float32)])
        self._buf[at:at + len(mic)] += (mic * self._gain).astype(np.float32)

    def take(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        head = self._buf[:n]
        out[:len(head)] = head
        self._buf = self._buf[n:]
        self._read += n
        return out


def _timeline(mic_audio: np.ndarray, rate: int, inject: list[tuple[float, np.ndarray]] | None,
              step: int) -> np.ndarray:
    """``mic_audio`` with each injected clip mixed in at its offset, padded to whole steps."""
    mic = np.asarray(mic_audio, dtype=np.float32).reshape(-1)
    length = len(mic)
    for offset, clip in inject or []:
        length = max(length, int(offset * rate) + len(clip))
    out = np.zeros(-(-length // step) * step, dtype=np.float32)
    out[: len(mic)] += mic
    for offset, clip in inject or []:
        start = int(offset * rate)
        out[start:start + len(clip)] += np.asarray(clip, dtype=np.float32).reshape(-1)
    return out


async def replay(
    session: ConversationSession, mic_audio: np.ndarray, rate: int, *,
    tail_seconds: float = 6.0, realtime: bool = True,
    inject: list[tuple[float, np.ndarray]] | None = None,
    cues: list[Cue] | None = None, echo: Echo | None = None,
    until: Callable[[], bool] | None = None, speed: float = 1.0,
) -> list[np.ndarray]:
    """Feed the audio in 20 ms steps while pulling output frames at the same cadence.

    ``inject`` mixes extra clips at fixed offsets in seconds; ``cues`` mix clips when a condition first
    holds; ``echo`` loops the agent's output back into the mic. Once all of that audio has been fed,
    silence follows for up to ``tail_seconds``, ending early as soon as ``until()`` holds (it is asked
    once per silent step, so it can count steps of audio time).
    ``realtime`` paces the steps at ``speed`` times real time; ``realtime=False`` only yields between
    steps, which suits fake components.
    """
    step = rate * STEP_MS // 1000
    signal = _timeline(mic_audio, rate, inject, step)
    line = _EchoLine(echo, step) if echo is not None else None
    cues = cues or []
    playing: list[list] = []  # [clip, next sample]
    tail_steps = round(tail_seconds * 1000 / STEP_MS)
    quiet = 0
    output: list[np.ndarray] = []
    started = time.perf_counter()
    index = 0
    while True:
        start = index * step
        for cue in cues:
            if cue._due_step is None and cue.when():
                cue._due_step = index + round(cue.delay_s * 1000 / STEP_MS)
            if cue._due_step == index and cue.fired_at is None:
                cue.fired_at, cue.fired_audio_s = time.perf_counter(), start / rate
                playing.append([np.asarray(cue.clip, dtype=np.float32).reshape(-1), 0])
        waiting = any(c._due_step is not None and c.fired_at is None for c in cues)
        if start >= len(signal) and not playing and not waiting:
            if quiet >= tail_steps or (until is not None and until()):
                break
            quiet += 1
        else:
            quiet = 0
        chunk = signal[start:start + step].copy() if start < len(signal) else np.zeros(step, np.float32)
        for item in playing:
            part = item[0][item[1]:item[1] + step]
            chunk[:len(part)] += part
            item[1] += step
        playing = [item for item in playing if item[1] < len(item[0])]
        if line is not None:
            chunk += line.take(step)
        session.feed_mic(np.clip(chunk, -1.0, 1.0), rate)
        frame = session.next_playout_frame()
        if frame is not None:
            output.append(frame)
            if line is not None:
                line.add(frame, start)
        index += 1
        if realtime:
            due = started + index * STEP_MS / 1000 / speed
            await asyncio.sleep(max(0.0, due - time.perf_counter()))
        else:
            await asyncio.sleep(0)
    return output
