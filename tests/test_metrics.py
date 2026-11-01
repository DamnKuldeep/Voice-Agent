import pytest

from voiceagent.core.events import Mark
from voiceagent.core.metrics import TurnTrace, percentile, summarize


def make_trace(speech_end: float, first_audio: float) -> TurnTrace:
    t = TurnTrace(turn_id=1)
    t.mark(Mark.SPEECH_END, speech_end)
    t.mark(Mark.COMMIT, speech_end + 0.2)
    t.mark(Mark.AUDIO_OUT_START, first_audio)
    return t


def test_first_mark_wins():
    t = TurnTrace(turn_id=1)
    t.mark(Mark.COMMIT, 1.0)
    t.mark(Mark.COMMIT, 2.0)
    assert t.marks[Mark.COMMIT] == 1.0


def test_derived_spans_in_milliseconds():
    d = make_trace(10.0, 10.9).derived()
    assert d["endpoint_ms"] == pytest.approx(200)
    assert d["v2v_ms"] == pytest.approx(900)
    assert "llm_ttft_ms" not in d  # marks missing, so metric is absent rather than invented


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 90) == 5


def test_summarize_across_turns():
    s = summarize([make_trace(0, 1.0), make_trace(0, 1.2), make_trace(0, 2.0)])
    assert s["v2v_ms"]["n"] == 3
    assert s["v2v_ms"]["p50"] == pytest.approx(1200)
    assert s["v2v_ms"]["max"] == pytest.approx(2000)
