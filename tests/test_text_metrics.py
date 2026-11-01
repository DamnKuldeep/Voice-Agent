from voiceagent.core.text_metrics import normalize_words, wer, word_errors


def test_normalisation_ignores_case_and_punctuation():
    assert normalize_words("Hello, World! It's twenty-one.") == ["hello", "world", "it's", "twenty", "one"]


def test_identical_text_has_no_errors():
    assert word_errors("Set a reminder.", "set a reminder") == (0, 3)


def test_substitution_insertion_deletion():
    assert word_errors("a b c", "a x c") == (1, 3)
    assert word_errors("a b c", "a b c d") == (1, 3)
    assert word_errors("a b c", "a c") == (1, 3)


def test_corpus_wer_pools_errors_over_all_reference_words():
    pairs = [("one two", "one two"), ("three four five six", "three four")]
    assert wer(pairs) == 2 / 6


def test_empty_reference_gives_zero():
    assert wer([]) == 0.0
