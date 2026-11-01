"""Barge-in confirmation window: decides when speech during playback is a real interruption."""

from __future__ import annotations

import re
from enum import Enum

from ..config import BargeIn

BACKCHANNEL_WORDS = frozenset({
    "yeah", "yes", "yep", "yup", "uh", "huh", "mm", "hmm", "mhm", "okay", "ok", "right", "sure",
    "got", "it", "i", "see", "alright", "cool", "go", "on",
})
MAX_BACKCHANNEL_WORDS = 3


class BargeEvent(Enum):
    ONSET = "onset"          # first speech frame of a candidate
    DUCK = "duck"            # min_speech_ms of speech by VAD alone: lower the volume while words decide
    CHECK = "check"          # short speech ended: the transcript decides (see resolve)
    CONFIRMED = "confirmed"  # a real interruption: stop talking
    DISMISSED = "dismissed"  # noise or backchannel: keep talking


def is_backchannel(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return 0 < len(words) <= MAX_BACKCHANNEL_WORDS and all(w in BACKCHANNEL_WORDS for w in words)


class BargeInDetector:
    def __init__(self, cfg: BargeIn, end_silence_ms: int) -> None:
        self._cfg = cfg
        self._end_silence_ms = end_silence_ms
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._awaiting = False
        self._ducked = False
        self._speech_ms = 0
        self._silence_ms = 0

    @property
    def active(self) -> bool:
        """A candidate is in progress, so the caller should be collecting its audio."""
        return self._active

    @property
    def ducked(self) -> bool:
        return self._ducked

    def process(self, speech_prob: float, frame_ms: int) -> tuple[BargeEvent, ...]:
        if self._awaiting:
            return ()
        cfg = self._cfg
        if speech_prob >= cfg.vad_threshold:
            events = [] if self._active else [BargeEvent.ONSET]
            self._active = True
            self._speech_ms += frame_ms
            self._silence_ms = 0
            long_enough = self._speech_ms >= cfg.min_speech_ms
            if long_enough and (not cfg.ignore_backchannels or self._speech_ms > cfg.backchannel_max_ms):
                self.reset()
                events.append(BargeEvent.CONFIRMED)
            elif long_enough and not self._ducked:
                self._ducked = True
                events.append(BargeEvent.DUCK)
            return tuple(events)
        if not self._active:
            return ()
        self._silence_ms += frame_ms
        if self._silence_ms < self._end_silence_ms:
            return ()
        if self._speech_ms >= cfg.min_speech_ms:
            self._awaiting = True
            return (BargeEvent.CHECK,)
        self.reset()
        return (BargeEvent.DISMISSED,)

    def on_partial(self, transcript: str) -> BargeEvent | None:
        """Confirm before the backchannel window ends: enough speech, and words that are not "uh-huh".

        Waiting out backchannel_max_ms for every real interruption made the agent talk over the user for
        about 0.7 s; the words heard so far already tell the two apart.
        """
        if not self._active or self._awaiting or self._speech_ms < self._cfg.min_speech_ms:
            return None
        if not transcript.strip() or is_backchannel(transcript):
            return None
        self.reset()
        return BargeEvent.CONFIRMED

    def resolve(self, transcript: str) -> BargeEvent:
        """Settle a CHECK. Silence from the recogniser means noise, not an interruption."""
        self.reset()
        if not transcript.strip() or is_backchannel(transcript):
            return BargeEvent.DISMISSED
        return BargeEvent.CONFIRMED
