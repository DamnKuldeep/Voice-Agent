"""Interruption scenarios (evals/sets/interruptions.yaml) and the sounds they inject."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from .dataset import RATE, SETS_DIR, Utterance, balanced_subset, load_manifest

Kind = Literal["interrupt", "backchannel", "noise", "echo"]
ONSET_RUN_FRAMES = 3   # about 100 ms of speech by VAD: a click or a breath before the words does not count


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: Kind
    offset_s: float = 1.0       # after the agent starts speaking
    gain_db: float = -15.0      # echo only
    delay_ms: float = 60.0      # echo only


@dataclass(frozen=True)
class ScenarioSet:
    question: str
    clip_manifest: str
    clip_source: str
    backchannels: list[str]
    observe_after_s: float      # how long to watch after the injected sound ends
    echo_observe_s: float       # how much of the reply to play with echo on
    scenarios: list[Scenario]


@dataclass(frozen=True)
class Trial:
    id: str
    scenario: Scenario
    clip: Utterance | None = None   # real interruption: the recording and its reference text
    text: str = ""                  # backchannel: what Kokoro says
    seed: int = 0                   # noise: which burst


def load_scenarios(name: str | Path) -> ScenarioSet:
    path = Path(name) if Path(name).is_absolute() else SETS_DIR / name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["scenarios"] = [Scenario(**s) for s in data["scenarios"]]
    return ScenarioSet(**data)


def expand(spec: ScenarioSet, repeats: int, clips: list[Utterance] | None = None) -> list[Trial]:
    """``repeats`` rounds of every scenario; each real interruption gets a different recording."""
    if clips is None:
        pool = [c for c in load_manifest(spec.clip_manifest) if c.source == spec.clip_source]
        wanted = repeats * sum(s.kind == "interrupt" for s in spec.scenarios)
        clips = balanced_subset(pool, wanted)
    trials: list[Trial] = []
    used = 0
    for rep in range(repeats):
        for s in spec.scenarios:
            trial_id = f"{s.id}#{rep}"
            if s.kind == "interrupt":
                if not clips:
                    raise ValueError(f"no recordings from {spec.clip_source} for {s.id}")
                trials.append(Trial(trial_id, s, clip=clips[used % len(clips)]))
                used += 1
            elif s.kind == "backchannel":
                trials.append(Trial(trial_id, s, text=spec.backchannels[rep % len(spec.backchannels)]))
            else:
                trials.append(Trial(trial_id, s, seed=rep))
    return trials


def cough(seed: int, rate: int = RATE) -> np.ndarray:
    """Two short noise bursts with a sharp attack, like a cough: loud enough to trip a VAD now and then."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(0.55 * rate), dtype=np.float32)
    for start_s, length_s in ((0.0, 0.18), (0.3, 0.22)):
        n = int(length_s * rate)
        envelope = np.exp(-np.linspace(0, 5, n)) * np.minimum(1.0, np.linspace(0, 20, n))
        burst = np.convolve(rng.standard_normal(n), np.ones(4) / 4, mode="same")  # takes the hiss off
        start = int(start_s * rate)
        out[start:start + n] = (0.4 * envelope * burst / np.abs(burst).max()).astype(np.float32)
    return out


def speech_onset_s(probs: Sequence[float], frame_s: float, threshold: float = 0.5) -> float:
    """Where speech starts in a recording, from its per-frame VAD probabilities.

    An energy threshold was used first and fired on a noise burst 2.1 s before the words of one
    Common Voice clip, which made a prompt stop look like a missed interruption.
    """
    run = 0
    for index, prob in enumerate(probs):
        run = run + 1 if prob >= threshold else 0
        if run == ONSET_RUN_FRAMES:
            return (index - ONSET_RUN_FRAMES + 1) * frame_s
    return 0.0
