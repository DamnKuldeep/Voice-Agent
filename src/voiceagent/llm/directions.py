"""Remove stage directions ("*smiles*", "[laughs]", "(pauses)") that chat models write but nobody says.

Rules, in order:
- ``[anything]`` is dropped (a markdown link ``[text](url)`` is left for the markdown stripper).
- ``*text*`` (single asterisks) is dropped when it uses an action word or stands alone as its own
  sentence; inline emphasis such as "that is *really* good" is kept.
- ``(text)`` is dropped only when it is short and uses an action word; "(about five kilometres)"
  is real speech and stays.
- A text that is nothing but a bare direction ("pauses", "remains silent") becomes empty.
"""

from __future__ import annotations

import re

_ACTION = re.compile(
    r"\b(?:paus(?:e|es|ed|ing)|smil(?:e|es|ed|ing)|grin(?:s|ning)?|laugh(?:s|ing|ter)?"
    r"|chuckl(?:e|es|ing)|giggl(?:e|es|ing)|sigh(?:s|ing)?|nod(?:s|ding)?|shrug(?:s|ging)?"
    r"|wink(?:s|ing)?|clear(?:s|ing)? (?:her|his|my) throat|lean(?:s|ing)|silen(?:t|ce|tly)"
    r"|quietly|says nothing|no (?:response|reply)|waits|waiting|whisper(?:s|ing)|gasp(?:s|ing)?"
    r"|cough(?:s|ing)?|exhal(?:es|ing)|inhal(?:es|ing)|deep breath|a beat|hesitat(?:es|ing)"
    r"|blink(?:s|ing)|tilts|looks (?:up|down|away|around)|softly|warmly|cheerfully"
    r"|excitedly|thoughtfully)\b",
    re.IGNORECASE,
)
_BRACKETS = re.compile(r"\[[^\]\n]*\](?!\()")
_SINGLE_STARS = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)")
_PARENS = re.compile(r"\(([^()\n]*)\)")
_PAREN_MAX_WORDS = 6
_BARE = re.compile(
    r"^\W*(?:(?:she|he|i|mira)\s+)?(?:paus(?:e|es|ing)|silence|says nothing|no response"
    r"|(?:remains|stays|staying|keeps|goes|remain|stay)\s+(?:quiet|silent)"
    r"|laughs|smiles|sighs|nods)\W*$",
    re.IGNORECASE,
)
_OPEN_STAR = re.compile(r"(?<!\*)\*(?!\*)")


def _stands_alone(text: str, start: int, end: int) -> bool:
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    return (not before or before[-1] in ".!?") and (not after or not after[0].islower())


def _star(m: re.Match[str]) -> str:
    inner = m.group(1)
    if _ACTION.search(inner) or _stands_alone(m.string, m.start(), m.end()):
        return " "
    return m.group(0)


def _paren(m: re.Match[str]) -> str:
    inner = m.group(1)
    if not inner.strip() or (len(inner.split()) <= _PAREN_MAX_WORDS and _ACTION.search(inner)):
        return " "
    return m.group(0)


def strip_directions(text: str) -> str:
    text = _BRACKETS.sub(" ", text)
    text = _SINGLE_STARS.sub(_star, text)
    text = _PARENS.sub(_paren, text)
    return "" if _BARE.match(text) else text


def inside_direction(text: str) -> bool:
    """True when ``text`` ends inside an unclosed ``*...``, ``[...`` or ``(...``.

    The chunker uses this so it never cuts a direction in two, which would hide it from
    ``strip_directions``.
    """
    return (
        len(_OPEN_STAR.findall(text)) % 2 == 1
        or text.count("[") > text.count("]")
        or text.count("(") > text.count(")")
    )
