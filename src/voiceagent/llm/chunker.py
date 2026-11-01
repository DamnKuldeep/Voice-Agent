"""Cut a streaming LLM reply into chunks the TTS can start on early."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

from voiceagent.llm.directions import inside_direction
from voiceagent.llm.normalize import normalize_for_speech

if TYPE_CHECKING:
    from voiceagent.config import Chunker

_SENTENCE_END = ".!?"
_CLAUSE_END = ";:,"
_CLOSERS = "\"')]”’"
# A period after these is not the end of a sentence.
_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "jr", "sr", "vs", "e.g", "i.e", "approx"}
)
_WORD_BEFORE_SPACE = re.compile(r"\S+(?=\s)")
_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]”’]*$")


def _is_false_stop(buf: str, i: int) -> bool:
    """True when the period at ``buf[i]`` belongs to an abbreviation, an initial or a list number."""
    if buf[i] != ".":
        return False
    start = i
    while start > 0 and not buf[start - 1].isspace():
        start -= 1
    token = buf[start:i].lstrip("\"'([“‘")
    if token.lower() in _ABBREVIATIONS or (len(token) == 1 and token.isupper()):
        return True
    if token.isdigit():
        before = buf[:start].rstrip(" \t")
        return not before or before.endswith("\n")
    return False


def _boundaries(buf: str) -> Iterator[tuple[int, str]]:
    """Yield (cut position, kind) for each place the buffer could be cut, in order.

    A mark only counts once the character after it is known, so "3.5" and "12,500" never split.
    """
    n = len(buf)
    for i, ch in enumerate(buf):
        if ch == "\n":
            if buf[:i].strip():
                yield i, "sentence"
        elif ch in _SENTENCE_END or ch in _CLAUSE_END:
            j = i + 1
            while j < n and buf[j] in _CLOSERS:
                j += 1
            if j < n and buf[j].isspace() and not _is_false_stop(buf, i):
                yield j, "sentence" if ch in _SENTENCE_END else "clause"
        elif ch in "–—":
            if i > 0 and i + 1 < n:
                yield i + 1, "clause"
        elif ch == "-" and 0 < i < n - 1 and buf[i - 1].isspace() and buf[i + 1].isspace():
            yield i + 1, "clause"


def _word_end(buf: str, count: int) -> int | None:
    """Position just after the ``count``-th word that is already followed by whitespace."""
    for k, m in enumerate(_WORD_BEFORE_SPACE.finditer(buf), start=1):
        if k == count:
            return m.end()
    return None


def _words(text: str) -> int:
    return len(text.split())


def _speakable(text: str) -> str:
    spoken = normalize_for_speech(text)
    return spoken if any(c.isalnum() for c in spoken) else ""


class SpeechChunker:
    def __init__(self, cfg: Chunker) -> None:
        self._cfg = cfg
        self._buf = ""
        self._first_done = False
        self._emitted = False

    def reset(self) -> None:
        self._buf = ""
        self._first_done = False
        self._emitted = False

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        chunks: list[str] = []
        while (cut := self._next_cut()) is not None:
            piece, self._buf = self._buf[:cut], self._buf[cut:].lstrip()
            spoken = _speakable(piece)
            if spoken:
                chunks.append(spoken)
                self._first_done = self._emitted = True
        return chunks

    def flush(self, truncated: bool = False) -> list[str]:
        rest = self._buf
        if truncated:
            rest = self._complete_part(rest)
        spoken = _speakable(rest)
        self.reset()
        return [spoken] if spoken else []

    def _complete_part(self, rest: str) -> str:
        """Keep only whole sentences; if nothing was spoken yet, keep everything."""
        stripped = rest.rstrip()
        if _ENDS_SENTENCE.search(stripped) and not _is_false_stop(stripped, len(stripped) - 1):
            return stripped
        ends = [end for end, kind in _boundaries(rest) if kind == "sentence"]
        if ends:
            return rest[: ends[-1]]
        return "" if self._emitted else rest

    def _next_cut(self) -> int | None:
        cut = self._plain_cut()
        if cut is None or not inside_direction(self._buf[:cut]):
            return cut
        # Never split a "*stage direction*": cut just after it closes, but not without bound.
        for k, m in enumerate(_WORD_BEFORE_SPACE.finditer(self._buf), start=1):
            if k >= 2 * self._cfg.chunk_max_words:
                return m.end()
            if m.end() > cut and not inside_direction(self._buf[: m.end()]):
                return m.end()
        return None

    def _plain_cut(self) -> int | None:
        cfg = self._cfg
        buf = self._buf
        if not self._first_done:
            limit = _word_end(buf, cfg.first_chunk_max_words)
            for end, _ in _boundaries(buf):
                if limit is not None and end > limit:
                    break
                if _words(buf[:end]) >= cfg.first_chunk_min_words:
                    return end
            return limit
        limit = _word_end(buf, cfg.chunk_max_words)
        last_clause = None
        for end, kind in _boundaries(buf):
            if limit is not None and end > limit:
                break
            if kind == "sentence":
                return end
            last_clause = end
        if limit is None:
            return None
        if last_clause is not None and _words(buf[:last_clause]) >= cfg.first_chunk_min_words:
            return last_clause
        return limit
