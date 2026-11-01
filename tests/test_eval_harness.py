"""Pure harness logic: subsets, conditions, ordering, scoring, summaries, scenarios. No models."""

import numpy as np
import pytest

from voiceagent.config import load_settings
from voiceagent.eval.conditions import Condition, component_key, interleave, parse_conditions
from voiceagent.eval.dataset import Utterance, balanced_subset
from voiceagent.eval.observer import Observer, settled
from voiceagent.eval.scoring import (
    content_overlap,
    first_word_missing,
    last_word_missing,
    markdown_table,
    summarise,
    transcript_scores,
)
from voiceagent.eval.stimuli import ScenarioSet, cough, expand, load_scenarios, speech_onset_s


def utt(i: int, source: str) -> Utterance:
    return Utterance(f"{source}_{i}", f"{source}_{i}.wav", "text", source)


def test_balanced_subset_round_robins_sources_in_manifest_order():
    items = [utt(i, "b") for i in range(3)] + [utt(i, "a") for i in range(1)] + [utt(i, "c") for i in range(4)]
    picked = [u.id for u in balanced_subset(items, 6)]
    assert picked == ["a_0", "b_0", "c_0", "b_1", "c_1", "b_2"]
    assert balanced_subset(items, 6) == balanced_subset(list(items), 6)
    assert len(balanced_subset(items, 100)) == len(items)


def test_parse_conditions_and_default():
    assert parse_conditions(None) == [Condition("default")]
    conds = parse_conditions([["asr.chunk_ms=560"], ["asr.chunk_ms=160", "vad.gate_asr=false"]])
    assert conds[0] == Condition("asr.chunk_ms=560", {"asr": {"chunk_ms": 560}})
    assert conds[1].overrides == {"asr": {"chunk_ms": 160}, "vad": {"gate_asr": False}}
    with pytest.raises(ValueError):
        parse_conditions([["a=1"], ["a=1"]])


def test_interleave_is_item_major():
    a, b = Condition("A"), Condition("B")
    assert interleave([1, 2], [a, b]) == [(1, a), (1, b), (2, a), (2, b)]


def test_interleave_in_blocks_for_conditions_that_reload_models():
    a, b = Condition("a"), Condition("b")
    assert interleave([1, 2, 3], [a, b], block=2) == [(1, a), (2, a), (1, b), (2, b), (3, a), (3, b)]


def test_component_key_changes_only_for_settings_baked_into_the_models():
    base = load_settings()
    assert component_key(base) == component_key(load_settings(overrides={"vad": {"gate_asr": False}}))
    assert component_key(base) == component_key(load_settings(overrides={"barge_in": {"min_speech_ms": 400}}))
    assert component_key(base) != component_key(load_settings(overrides={"asr": {"chunk_ms": 80}}))
    assert component_key(base) != component_key(load_settings(overrides={"llm": {"model": "x/y"}}))


@pytest.mark.parametrize(("hyp", "missing"), [
    ("tell me about the history of plastic", False),
    ("tell me about the history of plas", True),     # clipped mid-word
    ("tell me about the history of", True),          # dropped
    ("Tell me about the history of plastic, please", False),
])
def test_last_word_missing(hyp, missing):
    assert last_word_missing("Tell me about the history of plastic.", hyp) is missing


def test_first_word_missing_and_empty_reference():
    assert first_word_missing("Stop, tell me about Delhi", "tell me about delhi")
    assert not first_word_missing("Stop, tell me about Delhi", "so stop tell me")
    assert not last_word_missing("", "anything") and not first_word_missing("", "")


def test_transcript_scores_and_overlap():
    s = transcript_scores("the burst tyre", "the burst")
    assert (s["wer_errors"], s["wer_words"]) == (1, 3) and s["last_word_missing"]
    assert content_overlap("What about the burst tyre?", "A burst tyre is dangerous.")
    assert not content_overlap("What about the tyre?", "The weather is nice.")


def rows():
    return [
        {"condition": "A", "wer_errors": 1, "wer_words": 10, "checks": {"split": True, "x": None},
         "metrics": {"v2v_ms": 800.0}},
        {"condition": "A", "wer_errors": 0, "wer_words": 10, "checks": {"split": False},
         "metrics": {"v2v_ms": 1000.0, "new_metric_ms": 5.0}},
        {"condition": "B", "checks": {"split": False}, "metrics": {}},
    ]


def test_summarise_counts_rates_over_applicable_items_and_picks_up_new_metrics():
    out = summarise(rows())
    assert out["A"]["n"] == 2 and out["A"]["wer"] == (0.05, 20)
    assert out["A"]["checks"] == {"split": (1, 2)}
    assert out["A"]["metrics"]["v2v_ms"] == (900.0, 980.0, 2)
    assert out["A"]["metrics"]["new_metric_ms"] == (5.0, 5.0, 1)
    assert out["B"]["wer"] is None and out["B"]["metrics"] == {}


def test_markdown_table_puts_conditions_side_by_side():
    table = markdown_table(summarise(rows()))
    lines = table.splitlines()
    assert lines[0] == "| quantity | `A` | `B` |"
    assert "| WER (reference words) | 5.0% (20) | - |" in lines
    assert "| split rate | 50% (1/2) | 0% (0/1) |" in lines
    assert "| v2v_ms p50 / p90 (n) | 900 / 980 (2) | - |" in lines


def test_observer_counts_commits_and_tracks_state_and_first_events():
    obs = Observer()
    for event in [{"type": "level", "rms": 0.1}, {"type": "state", "state": "listening"},
                  {"type": "state", "state": "thinking"}, {"type": "state", "state": "speaking"},
                  {"type": "barge", "phase": "duck"}, {"type": "partial", "text": "stop", "barge": True},
                  {"type": "asr_lag", "ms": 450.0}, {"type": "asr_lag", "ms": 0.0},
                  {"type": "turn", "metrics": {}, "flags": {"interrupted": True}, "transcript": "q",
                   "reply": "r"}]:
        obs(event)
    assert obs.commits == 1 and obs.state == "speaking" and obs.spoke_at is not None
    assert obs.first("barge", "duck") is not None and obs.first("barge", "stop") is None
    assert obs.barge_partials == ["stop"] and obs.asr_lag_max_ms == 450.0 and obs.interrupted
    assert "level" not in obs.counts


def test_settled_needs_consecutive_holds():
    values = iter([True, True, False, True, True, True])
    check = settled(lambda: next(values), 3)
    assert [check() for _ in range(6)] == [False, False, False, False, False, True]


def test_shipped_scenarios_load_and_expand_with_distinct_clips():
    spec = load_scenarios("interruptions.yaml")
    kinds = {s.kind for s in spec.scenarios}
    assert kinds == {"interrupt", "backchannel", "noise", "echo"}
    clips = [utt(i, "cv") for i in range(5)]
    trials = expand(spec, 2, clips)
    assert len(trials) == 2 * len(spec.scenarios)
    real = [t.clip.id for t in trials if t.scenario.kind == "interrupt"]
    assert len(real) == len(set(real)) == 4
    said = [t.text for t in trials if t.scenario.kind == "backchannel"]
    assert said == spec.backchannels[:2]
    assert len({t.id for t in trials}) == len(trials)


def test_expand_without_recordings_for_a_real_interruption_fails_clearly():
    spec = ScenarioSet("q", "m.yaml", "none", ["yeah"], 1.0, 1.0,
                       load_scenarios("interruptions.yaml").scenarios[:1])
    with pytest.raises(ValueError, match="no recordings"):
        expand(spec, 1, [])


def test_cough_is_deterministic_bounded_and_short():
    a, b = cough(0), cough(0)
    assert np.array_equal(a, b) and not np.array_equal(a, cough(1))
    assert np.abs(a).max() <= 0.41 and len(a) == int(0.55 * 16000)


def test_speech_onset_needs_a_run_of_speech_frames():
    # a lone high frame (a click) at 0.1 s is ignored; speech starts with the run at 0.5 s
    probs = [0.0] * 3 + [0.9] + [0.0] * 11 + [0.8] * 5
    assert speech_onset_s(probs, frame_s=1 / 30) == pytest.approx(0.5)
    assert speech_onset_s([0.1] * 20, frame_s=0.032) == 0.0
