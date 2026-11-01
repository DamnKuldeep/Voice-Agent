from voiceagent.config import load_settings
from voiceagent.pipeline.speculation import Speculator, normalize

SETTINGS = load_settings()


class Handle:
    def __init__(self, tokens: int = 5) -> None:
        self.tokens_used = tokens
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def speculator(**changes) -> Speculator:
    cfg = SETTINGS.speculation.model_copy(update=changes)
    return Speculator(cfg, SETTINGS.turn.hold_trailing_words)


def stable(spec: Speculator, text: str, times: int) -> None:
    for _ in range(times):
        spec.observe(text)


def test_starts_only_after_stable_chunks_and_enough_silence():
    spec = speculator(stable_chunks=2)
    stable(spec, "what is the capital", 1)
    assert not spec.should_start(300, "what is the capital")      # seen once
    stable(spec, "what is the capital", 1)
    assert not spec.should_start(100, "what is the capital")      # not silent long enough
    assert spec.should_start(200, "what is the capital")


def test_a_changing_partial_resets_stability():
    spec = speculator(stable_chunks=2)
    stable(spec, "what is the", 3)
    spec.observe("what is the capital")
    assert not spec.should_start(500, "what is the capital")


def test_does_not_start_on_a_hold_word_or_when_disabled_or_empty():
    spec = speculator()
    stable(spec, "i want a pizza and", 3)
    assert not spec.should_start(500, "i want a pizza and")
    assert not speculator(enabled=False).should_start(500, "x")
    empty = speculator()
    stable(empty, "", 3)
    assert not empty.should_start(500, "")


def test_hit_ignores_case_and_punctuation():
    spec = speculator()
    handle = Handle()
    spec.start(handle, "What is the capital of France")
    assert spec.resolve("what's... is the capital of france?") is None  # different words: a miss
    assert handle.cancelled and spec.wasted == 1

    handle = Handle()
    spec.start(handle, "What is the capital of France")
    assert spec.resolve("what is the capital, of France?") is handle
    assert not handle.cancelled and spec.hits == 1 and not spec.running


def test_cancel_counts_waste_and_tokens_once():
    spec = speculator()
    handle = Handle(tokens=12)
    spec.start(handle, "hello there")
    assert spec.cancel() is True and spec.cancel() is False
    assert handle.cancelled and spec.wasted == 1 and spec.wasted_tokens == 12


def test_reset_clears_turn_state_but_keeps_counters():
    spec = speculator()
    spec.start(Handle(), "hello")
    spec.resolve("hello")
    stable(spec, "hello", 3)
    spec.reset()
    assert not spec.should_start(500, "hello") and spec.hits == 1


def test_normalize():
    assert normalize("  Hello, WORLD!! It's me. ") == "hello world it's me"
