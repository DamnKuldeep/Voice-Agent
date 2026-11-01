"""The listening set: one recorded sentence per session, scored on what the session heard and when."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..config import Settings
from ..pipeline.factory import Components
from ..pipeline.session import ConversationSession
from ..transport.offline import STEP_MS, replay
from .dataset import RATE, Utterance
from .observer import Observer, settled
from .scoring import transcript_scores

LEAD_S = 0.6      # silence before the sentence, as when someone starts talking after a pause
SETTLE_S = 1.5    # idle this long after a reply before the clip is over (a late second commit shows up)
MAX_TAIL_S = 30.0


def steps(seconds: float) -> int:
    return max(1, round(seconds * 1000 / STEP_MS))


async def drive(
    settings: Settings, components: Components, mic: np.ndarray, observer: Observer,
    until: Callable[[], bool], *, realtime: bool = True, speed: float = 1.0,
    tail_s: float = MAX_TAIL_S, **replay_args,
) -> ConversationSession:
    """Run one fresh session (empty history, no trace file) over ``mic`` and return it stopped."""
    session = ConversationSession(settings, components, trace_dir=None)
    session.set_listener(observer)
    await session.start()
    try:
        await replay(session, mic, RATE, tail_seconds=tail_s, realtime=realtime, speed=speed,
                     until=until, **replay_args)
    finally:
        await session.stop()
    return session


def with_lead(audio: np.ndarray, lead_s: float = LEAD_S) -> np.ndarray:
    return np.concatenate([np.zeros(int(lead_s * RATE), dtype=np.float32), audio.astype(np.float32)])


def reply_finished(observer: Observer, settle_s: float = SETTLE_S) -> Callable[[], bool]:
    """Stop once something was committed and the session has been idle for ``settle_s``."""
    return settled(lambda: observer.commits > 0 and observer.state == "idle", steps(settle_s))


async def run_listening(
    item: Utterance, audio: np.ndarray, settings: Settings, components: Components, *,
    realtime: bool = True, speed: float = 1.0, settle_s: float = SETTLE_S,
) -> dict:
    obs = Observer()
    await drive(settings, components, with_lead(audio), obs, reply_finished(obs, settle_s),
                realtime=realtime, speed=speed)
    return score_listening(item, obs)


def score_listening(item: Utterance, obs: Observer) -> dict:
    hypothesis = " ".join(t["transcript"] for t in obs.turns if t.get("transcript"))
    last = obs.turns[-1] if obs.turns else {}
    scores = transcript_scores(item.text, hypothesis)
    metrics = dict(last.get("metrics", {}))
    if obs.asr_lag_max_ms is not None:
        metrics["asr_lag_max_ms"] = obs.asr_lag_max_ms
    flags = last.get("flags", {})
    return {
        "set": "listening", "item": item.id, "source": item.source,
        "reference": item.text, "hypothesis": hypothesis,
        "commits": obs.commits, "model": flags.get("llm_model"),
        "wer_errors": scores["wer_errors"], "wer_words": scores["wer_words"], "wer": scores["wer"],
        "checks": {
            "no_turn": obs.commits == 0,
            "split_commit": obs.commits > 1,
            "last_word_missing": scores["last_word_missing"],
            "first_word_missing": scores["first_word_missing"],
        },
        "metrics": metrics,
        "turns": obs.turns,
        "events": dict(obs.counts),
    }
