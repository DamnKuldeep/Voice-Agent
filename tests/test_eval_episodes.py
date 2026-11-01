"""The harness driving real ConversationSessions built from scripted fake components."""

import json

import numpy as np
import soundfile as sf

from pipeline_fakes import LONG_REPLY, FakeChat, make_components, make_settings, speech
from voiceagent.eval.conditions import parse_conditions
from voiceagent.eval.dataset import Utterance
from voiceagent.eval.interruptions import Stimuli, run_trial
from voiceagent.eval.listening import run_listening
from voiceagent.eval.runner import ComponentSlot, run_conditions, write_report
from voiceagent.eval.stimuli import Scenario, ScenarioSet, Trial

SPEED = 6.0  # fake components keep up with audio fed about six times faster than real time
QUESTION = "what is the capital of france"


async def test_listening_episode_scores_a_clean_single_commit():
    comps = make_components(transcripts=[QUESTION])
    item = Utterance("u1", "u1.wav", "What is the capital of France?", "fake")
    try:
        row = await run_listening(item, speech(1.2), make_settings(), comps, speed=SPEED, settle_s=0.3)
    finally:
        comps.close()
    assert row["commits"] == 1 and row["hypothesis"] == QUESTION and row["wer"] == 0.0
    assert row["checks"] == {"no_turn": False, "split_commit": False, "last_word_missing": False,
                             "first_word_missing": False}
    assert {"v2v_ms", "endpoint_ms", "asr_final_ms"} <= set(row["metrics"])
    assert row["model"] == "fake/model"
    json.dumps(row)  # the row goes to JSONL as it is


async def test_listening_episode_reports_a_truncated_ending():
    comps = make_components(transcripts=["what is the capital of fra"])
    item = Utterance("u1", "u1.wav", "What is the capital of France?", "fake")
    try:
        row = await run_listening(item, speech(1.2), make_settings(), comps, speed=SPEED, settle_s=0.3)
    finally:
        comps.close()
    assert row["checks"]["last_word_missing"] and row["wer_errors"] == 1


def spec(*scenarios: Scenario) -> ScenarioSet:
    return ScenarioSet("tell me about the history of the taj mahal in detail", "unused.yaml", "fake",
                       ["yeah"], observe_after_s=0.6, echo_observe_s=1.0, scenarios=list(scenarios))


async def test_real_interruption_trial_measures_stop_and_understanding(tmp_path):
    wav = tmp_path / "stop.wav"
    sf.write(wav, np.concatenate([np.zeros(3200, np.float32), speech(1.0)]), 16000)
    clip = Utterance("c1", str(wav), "No, tell me about Spain.", "fake")
    scenario = Scenario("interrupt_1s", "interrupt", offset_s=0.5)
    comps = make_components(transcripts=["tell me about the taj mahal", "no tell me about spain"],
                            chat=FakeChat(reply=LONG_REPLY))
    try:
        row = await run_trial(Trial("interrupt_1s#0", scenario, clip=clip), spec(scenario), Stimuli(),
                              make_settings(), comps, speed=SPEED)
    finally:
        comps.close()
    assert row["agent_spoke"] and row["stopped"]
    assert row["hypothesis"] == "no tell me about spain" and row["wer"] == 0.0
    assert row["checks"]["missed_within_1500ms"] is False
    assert row["checks"]["next_reply_given"] and row["checks"]["interrupt_last_word_missing"] is False
    assert 0 < row["metrics"]["stop_after_onset_ms"] < 1500
    assert "barge_in_stop_ms" in row["metrics"] and "next_v2v_ms" in row["metrics"]


async def test_backchannel_trial_that_is_ignored_is_not_a_false_interrupt():
    scenario = Scenario("backchannel_1s", "backchannel", offset_s=0.5)
    comps = make_components(transcripts=["tell me about the taj mahal"], chat=FakeChat(reply=LONG_REPLY))
    try:
        row = await run_trial(Trial("backchannel_1s#0", scenario, text="yeah"), spec(scenario), Stimuli(),
                              make_settings(), comps, speed=SPEED)
    finally:
        comps.close()
    assert row["agent_spoke"] and not row["stopped"]
    assert row["checks"] == {"false_interrupt": False, "false_interrupt_backchannel": False}
    assert row["injected_at_audio_s"] > 0


async def test_runner_interleaves_rebuilds_only_when_needed_and_writes_outputs(tmp_path):
    built: list[int] = []
    warmed: list[int] = []

    def build(settings):
        built.append(settings.asr.chunk_ms)
        return make_components()

    async def warmup(settings, components):
        warmed.append(settings.asr.chunk_ms)

    async def episode(item, settings, components):
        if item == "bad":
            raise RuntimeError("boom")
        return {"item": item, "checks": {"split_commit": settings.vad.gate_asr}, "metrics": {"v2v_ms": 1.0}}

    out = tmp_path / "run.jsonl"
    slot = ComponentSlot(build, warmup)
    same = parse_conditions([["vad.gate_asr=true"], ["vad.gate_asr=false"]])
    rows = await run_conditions(["a", "bad"], same, lambda c: make_settings(**{"asr": {"chunk_ms": 160}, **c.overrides}), episode, slot,
                                out, str)
    assert built == [160] and warmed == [160]
    assert [(r["item"], r["condition"]) for r in rows] == [
        ("a", "vad.gate_asr=true"), ("a", "vad.gate_asr=false"),
        ("bad", "vad.gate_asr=true"), ("bad", "vad.gate_asr=false")]
    assert "error" in rows[2] and len(out.read_text().splitlines()) == 4

    chunks = parse_conditions([["asr.chunk_ms=160"], ["asr.chunk_ms=560"]])
    await run_conditions(["a", "b"], chunks, lambda c: make_settings(**{"asr": {"chunk_ms": 160}, **c.overrides}), episode, slot, out, str)
    slot.close()
    assert built == [160, 560] and slot.components is None  # blocks: the loaded 160 serves a and b first

    table = write_report(rows, tmp_path / "run.md", title="t", notes=["- note"])
    assert "| split_commit rate | 100% (1/1) | 0% (0/1) |" in table
    assert "2 item(s) failed" in (tmp_path / "run.md").read_text()
