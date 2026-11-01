"""The interruption set: the agent gives a long answer and something talks over it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np

from ..audio.frames import resample_to_16k
from ..config import Settings
from ..core.events import Mark
from ..pipeline.factory import Components
from ..pipeline.session import ConversationSession
from ..transport.offline import Cue, Echo
from .dataset import load_audio
from .listening import SETTLE_S, drive, steps, with_lead
from .observer import Observer, settled
from .scoring import content_overlap, transcript_scores
from .stimuli import ScenarioSet, Trial, cough, speech_onset_s

VAD_FRAME, RATE = 512, 16000
# Stop-latency thresholds reported: the 500 ms target, and 1.5 s, past which a stop feels late.
STOP_WITHIN_MS = (500, 1500)


class Stimuli:
    """Synthesises (with the loaded Kokoro) and caches what the trials say, per Components object."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str], np.ndarray] = {}

    async def speak(self, components: Components, text: str) -> np.ndarray:
        key = (id(components), text)
        if key not in self._cache:
            tts = components.tts
            pcm = await asyncio.get_running_loop().run_in_executor(
                components.tts_pool, lambda: np.concatenate(list(tts.synthesize(text))))
            self._cache[key] = resample_to_16k(pcm, tts.sample_rate)
        return self._cache[key]

    async def injected(self, components: Components, trial: Trial) -> np.ndarray | None:
        kind = trial.scenario.kind
        if kind == "interrupt":
            assert trial.clip is not None
            return load_audio(trial.clip)
        if kind == "backchannel":
            return await self.speak(components, trial.text)
        if kind == "noise":
            return cough(trial.seed)
        return None

    @staticmethod
    def onset_s(components: Components, sound: np.ndarray) -> float:
        """Where the injected sound's speech starts, by the same VAD the agent uses (idle between trials)."""
        vad = components.vad
        vad.reset()
        probs = [vad.process(sound[i:i + VAD_FRAME]) for i in range(0, len(sound) - VAD_FRAME + 1, VAD_FRAME)]
        vad.reset()
        return speech_onset_s(probs, VAD_FRAME / RATE)


async def run_trial(
    trial: Trial, spec: ScenarioSet, stimuli: Stimuli, settings: Settings, components: Components, *,
    realtime: bool = True, speed: float = 1.0,
) -> dict:
    question = with_lead(await stimuli.speak(components, spec.question))
    sound = await stimuli.injected(components, trial)
    obs = Observer()
    scenario = trial.scenario
    cue = None if sound is None else Cue(lambda: obs.state == "speaking", sound, scenario.offset_s)
    echo = Echo(scenario.gain_db, scenario.delay_ms) if scenario.kind == "echo" else None
    session = await drive(settings, components, question, obs, _until(scenario.kind, spec, obs, cue),
                          realtime=realtime, speed=speed, cues=[cue] if cue else None, echo=echo)
    onset = None if sound is None else stimuli.onset_s(components, sound)
    return score_trial(trial, obs, session, cue, onset, speed)


def _until(kind: str, spec: ScenarioSet, obs: Observer, cue: Cue | None) -> Callable[[], bool]:
    """When a trial has shown what it can: replay asks this once per silent 20 ms step."""
    if cue is None:  # echo: play a stretch of the reply with the loop-back on
        done = settled(lambda: obs.spoke_at is not None and obs.state == "idle", steps(SETTLE_S))
        watched = settled(lambda: obs.spoke_at is not None, steps(spec.echo_observe_s))
    else:
        def fired() -> bool:
            return cue.fired_at is not None

        if kind == "interrupt":  # honoured: wait for the answer to the interruption
            done = settled(lambda: fired() and obs.interrupted and obs.state == "idle", steps(SETTLE_S))
        else:
            done = settled(lambda: False, 1)
        watched = settled(lambda: fired() and not obs.interrupted, steps(spec.observe_after_s))
    return lambda: any([done(), watched()])  # a list, so both counters advance on every step


def score_trial(trial: Trial, obs: Observer, session: ConversationSession, cue: Cue | None,
                speech_at_s: float | None, speed: float = 1.0) -> dict:
    """Latencies are wall-clock ms from the injected speech's onset; they mean real time only at speed 1."""
    kind = trial.scenario.kind
    spoke = obs.spoke_at is not None and (cue is None or cue.fired_at is not None)
    cut = next((i for i, t in enumerate(obs.turns) if t.get("flags", {}).get("interrupted")), None)
    stopped = cut is not None
    metrics = dict(obs.turns[0]["metrics"]) if obs.turns else {}
    if obs.asr_lag_max_ms is not None:
        metrics["asr_lag_max_ms"] = obs.asr_lag_max_ms
    row: dict = {"set": "interruptions", "item": trial.id, "scenario": trial.scenario.id, "kind": kind,
                 "agent_spoke": spoke, "stopped": stopped, "turns": obs.turns,
                 "barge_partials": obs.barge_partials, "events": dict(obs.counts)}
    onset = None
    if cue is not None and cue.fired_at is not None and speech_at_s is not None:
        onset = cue.fired_at + speech_at_s / speed
        row["injected_at_audio_s"] = cue.fired_audio_s
    stop_mark = _stop_mark(session, cut)
    if onset is not None and stop_mark is not None:
        metrics["stop_after_onset_ms"] = (stop_mark - onset) * 1000
    if onset is not None and (duck := obs.first("barge", "duck")) is not None:
        metrics["duck_after_onset_ms"] = (duck - onset) * 1000
    checks: dict[str, bool | None] = {}
    if kind == "interrupt":
        stop_ms = metrics.get("stop_after_onset_ms")
        for bound in STOP_WITHIN_MS:
            checks[f"missed_within_{bound}ms"] = None if not spoke else not (
                stopped and stop_ms is not None and stop_ms <= bound)
        later = obs.turns[cut + 1:] if cut is not None else []
        hypothesis = " ".join(t["transcript"] for t in later if t.get("transcript"))
        reply = " ".join(t["reply"] for t in later if t.get("reply"))
        assert trial.clip is not None
        row.update(reference=trial.clip.text, hypothesis=hypothesis, next_reply=reply,
                   clip=trial.clip.id)
        if stopped:
            scores = transcript_scores(trial.clip.text, hypothesis)
            row.update(wer_errors=scores["wer_errors"], wer_words=scores["wer_words"], wer=scores["wer"])
            checks["interrupt_first_word_missing"] = scores["first_word_missing"]
            checks["interrupt_last_word_missing"] = scores["last_word_missing"]
            checks["next_reply_given"] = bool(reply)
            checks["next_reply_overlaps_heuristic"] = content_overlap(trial.clip.text, reply)
        for name, value in (later[-1]["metrics"] if later else {}).items():
            metrics[f"next_{name}"] = value
    else:
        checks["false_interrupt"] = stopped if spoke else None
        checks[f"false_interrupt_{kind}"] = stopped if spoke else None
        row["injected_text"] = trial.text
    row["checks"], row["metrics"] = checks, metrics
    return row


def _stop_mark(session: ConversationSession, cut: int | None) -> float | None:
    if cut is None or cut >= len(session.traces):
        return None
    return session.traces[cut].marks.get(Mark.PLAYBACK_STOPPED)
