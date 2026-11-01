"""Software echo guard: is a barge-in candidate just the agent's own voice coming back through the mic?

The browser's echo cancellation removes most of it; what leaks through is recognised as the words the
agent has just said. Those words are compared, not the audio, so this costs nothing per frame.
"""

from __future__ import annotations

import re
from enum import Enum

ECHO_WINDOW_MS = 3000   # agent speech this recent can still be in the room or the speaker path
ECHO_MATCH = 0.7        # share of the candidate's words that must appear in that speech
MIN_WORDS = 2           # one matching word could be a real "wait" or "what"; wait for a second word

_WORD = re.compile(r"[a-z0-9']+")


class EchoVerdict(Enum):
    CLEAR = "clear"          # not echo: judge the words as an interruption
    ECHO = "echo"
    UNDECIDED = "undecided"  # too few words to tell yet


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def echo_verdict(candidate: str, agent_recent: str, *, final: bool = False) -> EchoVerdict:
    """``final``: the candidate has ended, so a single matching word is judged rather than waited on."""
    heard = _words(candidate)
    spoken = set(_words(agent_recent))
    if not heard or not spoken:
        return EchoVerdict.CLEAR
    matched = sum(word in spoken for word in heard) / len(heard)
    if matched < ECHO_MATCH:
        return EchoVerdict.CLEAR
    if len(heard) < MIN_WORDS and not final:
        return EchoVerdict.UNDECIDED
    return EchoVerdict.ECHO
