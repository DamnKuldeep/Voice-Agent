import json

from voiceagent.config import load_settings
from voiceagent.core.events import Flag, Mark
from voiceagent.core.metrics import TurnTrace
from voiceagent.pipeline.trace_log import TraceWriter, config_hash


def make_trace() -> TurnTrace:
    trace = TurnTrace(7)
    trace.mark(Mark.SPEECH_END, 10.0)
    trace.mark(Mark.COMMIT, 10.2)
    trace.mark(Mark.AUDIO_OUT_START, 10.7)
    trace.flag(Flag.LLM_MODEL, "fake/model")
    return trace


def test_one_json_line_per_turn(tmp_path):
    settings = load_settings()
    writer = TraceWriter(tmp_path / "traces", settings, vram_peak_mib=lambda: 1234.56)
    writer.write(make_trace(), "hello", "hi there")
    writer.write(make_trace(), "again", "yes")
    files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(files) == 1
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    row = rows[0]
    assert row["turn_id"] == 7 and row["transcript"] == "hello" and row["reply"] == "hi there"
    assert row["marks_ms"] == {"speech_end": 0.0, "commit": 200.0, "audio_out_start": 700.0}
    assert row["metrics"]["v2v_ms"] == 700.0 and row["flags"] == {"llm_model": "fake/model"}
    assert row["config_hash"] == config_hash(settings) and row["vram_peak_mib"] == 1234.6


def test_config_hash_changes_with_the_config():
    assert config_hash(load_settings()) != config_hash(load_settings(overrides={"tts": {"stream": False}}))


def test_unwritable_directory_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    TraceWriter(blocker / "sub", load_settings()).write(make_trace(), "a", "b")
