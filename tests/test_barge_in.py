from voiceagent.config import load_settings
from voiceagent.pipeline.barge_in import BargeEvent, BargeInDetector, is_backchannel

CFG = load_settings().barge_in
FRAME = 32
END_SILENCE = 160
SPEECH, QUIET = 0.9, 0.1


def run(detector: BargeInDetector, prob: float, ms: int) -> list[BargeEvent]:
    events: list[BargeEvent] = []
    for _ in range(ms // FRAME):
        events.extend(detector.process(prob, FRAME))
    return events


def test_blip_shorter_than_min_speech_is_dismissed():
    detector = BargeInDetector(CFG, END_SILENCE)
    events = run(detector, SPEECH, 96) + run(detector, QUIET, 320)
    assert events == [BargeEvent.ONSET, BargeEvent.DISMISSED]
    assert not detector.active


def test_probability_below_the_stricter_threshold_is_ignored():
    detector = BargeInDetector(CFG, END_SILENCE)
    assert run(detector, CFG.vad_threshold - 0.05, 2000) == []


def test_long_speech_confirms_without_waiting_for_a_transcript():
    detector = BargeInDetector(CFG, END_SILENCE)
    events = run(detector, SPEECH, 704)
    assert events == [BargeEvent.ONSET, BargeEvent.DUCK, BargeEvent.CONFIRMED]


def test_short_phrase_is_checked_and_a_backchannel_is_dismissed():
    detector = BargeInDetector(CFG, END_SILENCE)
    events = run(detector, SPEECH, 416) + run(detector, QUIET, 320)
    assert events == [BargeEvent.ONSET, BargeEvent.DUCK, BargeEvent.CHECK]
    assert run(detector, SPEECH, 640) == []  # ignored while the check is pending
    assert detector.resolve("uh-huh") is BargeEvent.DISMISSED
    assert not detector.active


def test_short_phrase_with_content_is_confirmed():
    detector = BargeInDetector(CFG, END_SILENCE)
    run(detector, SPEECH, 416)
    assert run(detector, QUIET, 320) == [BargeEvent.CHECK]
    assert detector.resolve("wait, stop") is BargeEvent.CONFIRMED


def test_empty_transcript_is_noise():
    detector = BargeInDetector(CFG, END_SILENCE)
    run(detector, SPEECH, 416)
    run(detector, QUIET, 320)
    assert detector.resolve("  ") is BargeEvent.DISMISSED


def test_with_backchannels_not_ignored_min_speech_confirms_at_once():
    detector = BargeInDetector(CFG.model_copy(update={"ignore_backchannels": False}), END_SILENCE)
    just_enough = -(-CFG.min_speech_ms // FRAME) * FRAME  # min_speech_ms rounded up to whole frames
    assert run(detector, SPEECH, just_enough) == [BargeEvent.ONSET, BargeEvent.CONFIRMED]


def test_short_gap_inside_speech_does_not_end_the_candidate():
    detector = BargeInDetector(CFG, END_SILENCE)
    events = run(detector, SPEECH, 192) + run(detector, QUIET, 64) + run(detector, SPEECH, 512)
    assert events == [BargeEvent.ONSET, BargeEvent.DUCK, BargeEvent.CONFIRMED]


def test_backchannel_words():
    for text in ["yeah", "Uh-huh.", "okay, right", "mm-hmm", "I see"]:
        assert is_backchannel(text), text
    for text in ["wait", "no that is wrong", "yeah but what about", ""]:
        assert not is_backchannel(text), text


def test_content_words_confirm_as_soon_as_min_speech_is_heard():
    detector = BargeInDetector(CFG, END_SILENCE)
    run(detector, SPEECH, CFG.min_speech_ms + FRAME)
    assert detector.on_partial("Actually what") is BargeEvent.CONFIRMED
    assert not detector.active


def test_backchannel_words_do_not_confirm_early():
    detector = BargeInDetector(CFG, END_SILENCE)
    run(detector, SPEECH, CFG.min_speech_ms + FRAME)
    assert detector.on_partial("Yeah") is None
    assert detector.on_partial("") is None
    assert detector.active


def test_partial_before_min_speech_or_without_a_candidate_is_ignored():
    detector = BargeInDetector(CFG, END_SILENCE)
    assert detector.on_partial("Actually what") is None
    run(detector, SPEECH, 96)
    assert detector.on_partial("Actually what") is None


def test_duck_comes_once_at_min_speech_by_vad_alone():
    detector = BargeInDetector(CFG, END_SILENCE)
    before = run(detector, SPEECH, CFG.min_speech_ms - FRAME)
    assert before == [BargeEvent.ONSET] and not detector.ducked
    assert run(detector, SPEECH, 2 * FRAME) == [BargeEvent.DUCK]
    assert detector.ducked and run(detector, SPEECH, FRAME) == []


def test_blip_below_min_speech_never_ducks():
    detector = BargeInDetector(CFG, END_SILENCE)
    events = run(detector, SPEECH, 128) + run(detector, QUIET, 320)
    assert BargeEvent.DUCK not in events
