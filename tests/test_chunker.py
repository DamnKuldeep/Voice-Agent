import pytest

from voiceagent.config import Chunker
from voiceagent.llm.chunker import SpeechChunker


@pytest.fixture
def chunker() -> SpeechChunker:
    return SpeechChunker(
        Chunker(first_chunk_min_words=4, first_chunk_max_words=8, chunk_max_words=25)
    )


def stream(chunker: SpeechChunker, text: str, step: int = 3) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i : i + step])
    return out


def test_first_chunk_at_clause_after_min_words(chunker: SpeechChunker) -> None:
    out = stream(chunker, "Well, that is a good question, and here is why. More text. ")
    assert out[0] == "Well, that is a good question,"
    assert out[1] == "and here is why."


def test_early_comma_below_min_words_is_skipped(chunker: SpeechChunker) -> None:
    out = chunker.feed("Well, that is great and fine. ")
    assert out == ["Well, that is great and fine."]


def test_force_break_at_max_words_on_word_boundary(chunker: SpeechChunker) -> None:
    out = stream(chunker, "one two three four five six seven eight nine ten eleven twelve ")
    assert out[0] == "one two three four five six seven eight"


def test_no_half_words_before_whitespace(chunker: SpeechChunker) -> None:
    assert chunker.feed("one two three four five six seven eight") == []
    assert chunker.feed("th") == []


def test_never_splits_inside_number(chunker: SpeechChunker) -> None:
    out = stream(chunker, "The value of pi is 3.14 and the cost is 12,500 rupees right now. ")
    assert "three point one four" in " ".join(out)


def test_later_chunks_end_at_sentences(chunker: SpeechChunker) -> None:
    out = stream(
        chunker, "Sure thing, I can help with that. First idea, which is long. Second idea. "
    )
    assert out == [
        "Sure thing, I can help with that.",
        "First idea, which is long.",
        "Second idea.",
    ]


def test_later_chunk_bounded_by_max_words() -> None:
    c = SpeechChunker(Chunker(first_chunk_min_words=2, first_chunk_max_words=3, chunk_max_words=6))
    out = stream(c, "a b c d e f g h i j k l m n o p q r. ")
    assert all(len(x.split()) <= 6 for x in out)
    assert "".join(out).replace(" ", "") == "abcdefghijklmnopqr."


def test_abbreviation_is_not_a_boundary(chunker: SpeechChunker) -> None:
    out = chunker.feed("Please ask Dr. Smith about the plan today. ")
    assert out == ["Please ask Doctor Smith about the plan today."]


def test_flush_emits_rest_and_resets(chunker: SpeechChunker) -> None:
    assert chunker.feed("Hello there") == []
    assert chunker.flush() == ["Hello there"]
    assert chunker.flush() == []


def test_flush_truncated_drops_partial_sentence(chunker: SpeechChunker) -> None:
    out = chunker.feed("This is the first full sentence. And this one is cut o")
    out += chunker.flush(truncated=True)
    assert out == ["This is the first full sentence."]


def test_flush_truncated_speaks_unspoken_complete_sentences() -> None:
    c = SpeechChunker(Chunker(first_chunk_min_words=4, first_chunk_max_words=8, chunk_max_words=25))
    c.feed("Hi. Bye. and then half")
    assert c.flush(truncated=True) == ["Hi. Bye."]


def test_flush_truncated_speaks_text_if_nothing_complete_exists(chunker: SpeechChunker) -> None:
    chunker.feed("only a few")
    assert chunker.flush(truncated=True) == ["only a few"]


def test_flush_truncated_prefers_last_complete_sentence_in_rest(chunker: SpeechChunker) -> None:
    chunker.feed("Alpha beta gamma delta epsilon zeta eta theta iota")
    out = chunker.feed(" kappa. Lambda mu nu xi omicron pi rho sigma tau")
    out += chunker.flush(truncated=True)
    assert " ".join(out).endswith("kappa.")


def test_never_emits_empty_chunks(chunker: SpeechChunker) -> None:
    out = stream(chunker, "** \n\n ... \n 😀 \n Hello there my friend, welcome. ")
    out += chunker.flush()
    assert out and all(c.strip() and any(ch.isalnum() for ch in c) for c in out)


def test_chunks_are_normalised(chunker: SpeechChunker) -> None:
    out = chunker.feed("You owe Rs 500 and 20% extra, ok. ")
    assert "five hundred rupees" in out[0]


def test_newline_ends_a_sentence_after_first(chunker: SpeechChunker) -> None:
    out = chunker.feed("Okay here is the plan for you\nsecond line has words\nthird")
    assert out[1] == "second line has words"


def test_long_stage_direction_is_never_split(chunker: SpeechChunker) -> None:
    text = "*leans back in the big chair and smiles warmly at you* Sure, I can help with that. "
    out = stream(chunker, text) + chunker.flush()
    assert out == ["Sure, I can help with that."]


def test_direction_only_reply_yields_nothing(chunker: SpeechChunker) -> None:
    assert stream(chunker, "*remains silent*") + chunker.flush() == []


def test_unclosed_asterisk_does_not_hold_text_forever(chunker: SpeechChunker) -> None:
    words = " ".join(f"w{i}" for i in range(60)) + " "
    out = stream(chunker, "*" + words)
    assert out and sum(len(c.split()) for c in out) >= 49
