"""Word error rate, with case and punctuation ignored and hyphenated words split."""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"[^a-z0-9' ]+")


def normalize_words(text: str) -> list[str]:
    return _NON_WORD.sub(" ", text.lower().replace("-", " ")).split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (edit distance in words, reference word count)."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r_word in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h_word in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r_word != h_word))
    return row[len(hyp)], len(ref)


def wer(pairs: list[tuple[str, str]]) -> float:
    """Corpus WER over (reference, hypothesis) pairs: total errors over total reference words."""
    errors = words = 0
    for reference, hypothesis in pairs:
        e, n = word_errors(reference, hypothesis)
        errors, words = errors + e, words + n
    return errors / words if words else 0.0
