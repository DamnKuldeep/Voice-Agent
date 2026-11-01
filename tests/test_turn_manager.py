import pytest

from voiceagent.config import load_settings
from voiceagent.pipeline.turn_manager import commit_after_ms, ends_with_hold_word, should_commit

CFG = load_settings().turn
SILENCE = CFG.model_copy(update={"detector": "silence"})
MID = (CFG.min_silence_ms + CFG.max_silence_ms) // 2


@pytest.mark.parametrize(("prob", "text", "expected"), [
    (0.9, "what is the capital of france", CFG.min_silence_ms),
    (0.7, "what is the capital of france", CFG.min_silence_ms),
    (0.1, "what is the capital of france", CFG.max_silence_ms),
    (0.3, "what is the capital of france", CFG.max_silence_ms),
    (0.5, "what is the capital of france", MID),
    (0.95, "i would like a pizza and", CFG.max_silence_ms),       # a hold word beats a confident model
    (0.95, "I would like to know why, um.", CFG.max_silence_ms),
    (None, "what is the capital of france", CFG.max_silence_ms),  # no probability yet: wait the full time
    (0.9, "", CFG.min_silence_ms),
])
def test_commit_delay_table(prob, text, expected):
    assert commit_after_ms(CFG, eot_prob=prob, transcript=text) == expected


def test_silence_detector_is_a_fixed_delay_whatever_the_model_says():
    for prob, text in [(0.99, "hello"), (0.01, "hello and"), (None, "")]:
        assert commit_after_ms(SILENCE, eot_prob=prob, transcript=text) == CFG.silence_only_ms


def test_never_commits_before_min_silence_and_always_by_max():
    assert not should_commit(CFG.min_silence_ms - 1, CFG, eot_prob=0.99, transcript="done")
    assert should_commit(CFG.min_silence_ms, CFG, eot_prob=0.99, transcript="done")
    assert not should_commit(CFG.max_silence_ms - 1, CFG, eot_prob=0.0, transcript="so")
    assert should_commit(CFG.max_silence_ms, CFG, eot_prob=0.0, transcript="so")


def test_hold_word_detection():
    words = CFG.hold_trailing_words
    assert ends_with_hold_word("I want that, and", words)
    assert ends_with_hold_word("Well UM", words)
    assert not ends_with_hold_word("a band", words)  # whole words only
    assert not ends_with_hold_word("", words)
