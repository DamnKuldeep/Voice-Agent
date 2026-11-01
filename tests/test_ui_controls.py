"""UI logic that needs no GPU, network or browser: settings metadata, overrides, host, handler."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import yaml

from ui_fakes import make_host
from voiceagent.config import load_settings
from voiceagent.ui import controls
from voiceagent.ui.control_catalog import CONTROLS
from voiceagent.ui.control_spec import GROUPS
from voiceagent.ui.utterance_store import RATE, save_utterance, trim_leading_silence

# The switches the latency and listening experiments compare must all be reachable from the page.
REQUIRED_KEYS = {
    "vad.gate_asr", "turn.detector", "speculation.enabled", "llm.stream", "tts.stream",
    "chunker.first_chunk_min_words", "chunker.first_chunk_max_words", "length_policy.enabled",
    "llm.routing_sort", "llm.model", "asr.chunk_ms", "barge_in.enabled",
    "barge_in.min_speech_ms", "barge_in.vad_threshold", "tts.fillers",
    "vad.end_hangover_ms", "asr.tail_ms", "barge_in.duck_gain", "barge_in.echo_guard",
    "llm.provider_order",
}


@pytest.fixture
def settings():
    return load_settings()


def test_every_control_key_exists_and_is_described(settings):
    group_ids = {gid for gid, _ in GROUPS}
    for control in CONTROLS:
        controls.read_key(settings, control.key)
        assert control.group in group_ids, control.key
        assert control.label.strip() and len(control.help.strip()) > 20, control.key
    assert len({c.key for c in CONTROLS}) == len(CONTROLS)
    assert REQUIRED_KEYS <= {c.key for c in CONTROLS}
    assert all(any(c.group == gid for c in CONTROLS) for gid in group_ids)


def test_config_defaults_sit_on_the_widget_grid(settings):
    """A slider snaps to min + n * step, so an off-grid default would silently change on Apply."""
    for control in CONTROLS:
        value = controls.read_key(settings, control.key)
        if control.kind == "slider":
            assert control.minimum <= value <= control.maximum, control.key
            steps = (value - control.minimum) / control.step
            assert abs(steps - round(steps)) < 1e-6, control.key
        if control.kind == "select":
            assert value in control.choices, control.key


def test_definitions_are_json_with_current_and_default(settings):
    current = load_settings(overrides={"asr": {"chunk_ms": 80}})
    defs = json.loads(json.dumps(controls.definitions(current, settings)))
    assert [g["id"] for g in defs["groups"]] == [gid for gid, _ in GROUPS]
    chunk = next(c for c in defs["controls"] if c["key"] == "asr.chunk_ms")
    assert chunk["value"] == 80 and chunk["default"] == 560 and chunk["needs_reload"]
    assert chunk["choices"] == [80, 160, 560] and "560" in chunk["help"]


def test_overrides_are_partial_nested_and_cast(settings):
    overrides = controls.overrides_from_values({
        "barge_in.min_speech_ms": "300.0", "asr.chunk_ms": "80", "llm.stream": "false",
        "llm.provider_order": " Groq, , Cerebras ", "barge_in.duck_gain": 0.5, "tts.voice": " af_bella ",
    }, settings)
    assert overrides["barge_in"] == {"min_speech_ms": 300, "duck_gain": 0.5}
    assert isinstance(overrides["barge_in"]["min_speech_ms"], int)
    assert overrides["llm"] == {"stream": False, "provider_order": ["Groq", "Cerebras"]}
    merged = load_settings(overrides=overrides)
    assert merged.asr.chunk_ms == 80 and merged.tts.voice == "af_bella"
    assert merged.vad == settings.vad
    assert controls.overrides_from_values({}, settings) == {}


@pytest.mark.parametrize("values", [
    {"no.such_key": 1},
    {"asr.chunk_ms": 333},
    {"barge_in.duck_gain": 1.5},
    {"tts.voice": "   "},
    {"chunker.first_chunk_min_words": 10},
    {"chunker.first_chunk_min_words": 6, "chunker.first_chunk_max_words": 5},
])
def test_bad_values_are_rejected(settings, values):
    with pytest.raises(ValueError):
        controls.overrides_from_values(values, settings)


def test_reload_only_for_model_baked_switches(settings):
    def changed(key, value):
        return load_settings(overrides=controls.overrides_from_values({key: value}, settings))

    for key, value in [("tts.stream", False), ("turn.detector", "silence"), ("asr.tail_ms", 320),
                       ("barge_in.echo_guard", False), ("llm.history_turns", 3)]:
        assert not controls.needs_reload(settings, changed(key, value)), key
    for key, value in [("asr.chunk_ms", 80), ("tts.voice", "af_bella"), ("devices.cpu_threads", 4),
                       ("llm.model", "openai/gpt-oss-20b"), ("llm.provider_order", "Groq")]:
        assert controls.needs_reload(settings, changed(key, value)), key


def test_utterance_store_writes_only_when_called(tmp_path):
    audio = np.concatenate([np.zeros(RATE), np.full(RATE, 0.3)]).astype(np.float32)
    assert trim_leading_silence(audio).size < audio.size
    manifest = tmp_path / "sets" / "utterances.yaml"
    manifest.parent.mkdir()
    manifest.write_text(yaml.safe_dump({"utterances": [{"id": "read_01"}]}), encoding="utf-8")
    uid = save_utterance(audio, " hello there ", tmp_path / "audio", manifest)
    items = yaml.safe_load(manifest.read_text(encoding="utf-8"))["utterances"]
    assert [i["id"] for i in items] == ["read_01", uid]
    assert items[1]["text"] == "hello there"
    assert (tmp_path / "audio" / f"{uid}.wav").exists()
    with pytest.raises(ValueError):
        save_utterance(np.zeros(0, dtype=np.float32), "x", tmp_path / "audio", manifest)


async def test_host_one_active_session_and_owner_checks():
    host, _ = make_host()
    first, second = object(), object()
    s1 = await host.open_session(first)
    s2 = await host.open_session(second)
    assert s1.stopped and s2.started and host.active
    assert host.session_for(first) is None and host.session_for(second) is s2
    await host.close_session(first)
    assert host.session_for(second) is s2
    await host.close_session(second)
    assert host.session_for(second) is None and s2.stopped and not host.active


async def test_host_refuses_session_until_models_ready():
    def build(_settings, _progress):
        raise RuntimeError("no gpu")

    host, _ = make_host(build=build)
    assert host.phase == "error" and "no gpu" in host.status
    with pytest.raises(RuntimeError):
        await host.open_session(object())


def test_host_load_reports_progress_and_phase():
    seen: list[dict] = []

    def build(_settings, progress):
        progress("Loading speech recognition")
        from types import SimpleNamespace
        return SimpleNamespace(close=lambda: None, turn_detector=None, fillers=[])

    make_host(build=build, publish=seen.append)
    assert [e["type"] for e in seen] == ["status", "progress", "status"]
    assert seen[1]["message"] == "Loading speech recognition"
    assert seen[0]["phase"] == "loading" and seen[-1]["phase"] == "ready" and seen[-1]["ready"]


async def test_host_records_and_forwards_session_events():
    seen: list[dict] = []
    host, _ = make_host(publish=seen.append)
    session = await host.open_session(object())
    emit = session.listener
    emit({"type": "state", "state": "speaking"})
    emit({"type": "final", "text": "hello"})
    emit({"type": "asr_lag", "ms": 640.0})
    emit({"type": "turn", "metrics": {"v2v_ms": 1.0}, "flags": {"interrupted": True},
          "transcript": "hello", "reply": "hi"})
    status = host.status_event()
    assert status["state"] == "speaking" and status["last_final"] == "hello"
    assert status["asr_lag_ms"] == 640.0 and status["last_turn"]["flags"] == {"interrupted": True}
    kinds = [e["type"] for e in seen]
    assert kinds[-5:] == ["state", "final", "asr_lag", "turn", "summary"]


async def test_apply_swaps_session_accumulates_and_reloads_only_when_needed():
    host, builds = make_host()
    first = await host.open_session(object())
    result = await asyncio.to_thread(host.apply, {"tts.stream": False})
    assert result["ok"] and result["changed"] == ["tts.stream"] and not result["reloaded"]
    assert len(builds) == 1 and first.stopped and host.session_for(host._owner) is not first
    result = await asyncio.to_thread(host.apply, {"asr.chunk_ms": 80})
    assert result["reloaded"] and len(builds) == 2
    assert host.settings.asr.chunk_ms == 80 and host.settings.tts.stream is False
    result = await asyncio.to_thread(host.apply, {"asr.chunk_ms": 80})
    assert result["changed"] == [] and not result["reloaded"]


async def test_apply_reloads_when_a_skipped_component_is_switched_on():
    host, builds = make_host()
    host.components.turn_detector = None
    await asyncio.to_thread(host.apply, {"turn.detector": "silence"})
    assert len(builds) == 1
    result = await asyncio.to_thread(host.apply, {"turn.detector": "smart_turn"})
    assert result["reloaded"] and len(builds) == 2
    result = await asyncio.to_thread(host.apply, {"tts.fillers": True})
    assert result["reloaded"] and len(builds) == 3


async def test_blocking_host_call_on_the_loop_fails_instead_of_deadlocking():
    host, _ = make_host()
    await host.open_session(object())
    with pytest.raises(RuntimeError, match="session loop"):
        host.reset_conversation()


async def test_handler_paces_frames_converts_pcm_and_routes_mic():
    from voiceagent.transport.fastrtc_handler import VoiceHandler

    host, _ = make_host()
    tapped: list[int] = []
    handler = VoiceHandler(host, 24000, on_mic=lambda rate, s: tapped.append(rate))
    await handler.start_up()
    await handler.receive((48000, np.zeros((1, 960), dtype=np.int16)))
    session = host.session_for(handler)
    assert session.fed == [(48000, 960)] and tapped == [48000]
    session.frames = [np.full(480, 0.5, dtype=np.float32) for _ in range(3)]
    rate, pcm = await handler.emit()
    assert rate == 24000 and pcm.dtype == np.int16 and pcm.shape == (480,) and pcm[0] == 16383
    loop = asyncio.get_running_loop()
    start = loop.time()
    assert await handler.emit() is not None
    assert await handler.emit() is not None
    assert loop.time() - start >= 0.02  # 40 ms of frames, less one early timer tick on Windows
    assert await handler.emit() is None
    assert handler.copy() is not handler


async def test_handler_reports_blocked_microphone():
    from fastrtc import WebRTCError

    from voiceagent.transport.fastrtc_handler import MIC_BLOCKED_MESSAGE, VoiceHandler

    host, _ = make_host()
    handler = VoiceHandler(host, 24000, mic_timeout_s=0.05)
    await handler.start_up()
    await asyncio.sleep(0.08)
    with pytest.raises(WebRTCError):
        await handler.emit()
    assert "Allow microphone access" in MIC_BLOCKED_MESSAGE


def test_app_parse_args_defaults():
    from voiceagent.ui.app import parse_args

    args = parse_args(["--smoke", "--port", "7861"])
    assert args.smoke and args.port == 7861 and args.host is None
