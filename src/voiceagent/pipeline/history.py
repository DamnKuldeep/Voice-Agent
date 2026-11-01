"""Conversation history sent to the LLM: the last few turns, with interrupted replies cut to what was heard."""

from __future__ import annotations

from dataclasses import dataclass

INTERRUPTED_MARK = "[interrupted by the user]"


@dataclass
class _Item:
    role: str
    content: str
    interrupted: bool = False


class History:
    def __init__(self, turns: int) -> None:
        self._turns = turns
        self._items: list[_Item] = []

    def add_user(self, text: str) -> None:
        self._items.append(_Item("user", text))

    def add_assistant(self, text: str) -> None:
        self._items.append(_Item("assistant", text))

    def drop_last_user(self) -> None:
        """Forget a user message whose turn was abandoned because the user kept talking."""
        if self._items and self._items[-1].role == "user":
            self._items.pop()

    def mark_interrupted(self, spoken_text: str) -> None:
        """Cut the last assistant message to the words the user actually heard and flag it."""
        for index in range(len(self._items) - 1, -1, -1):
            item = self._items[index]
            if item.role != "assistant":
                continue
            if spoken_text.strip():
                item.content, item.interrupted = spoken_text.strip(), True
            else:
                del self._items[index]
            return

    def clear(self) -> None:
        self._items.clear()

    def messages(self) -> list[dict[str, str]]:
        recent = self._items[-self._turns * 2:] if self._turns > 0 else []
        while recent and recent[0].role != "user":
            recent = recent[1:]
        return [
            {"role": i.role, "content": f"{i.content} {INTERRUPTED_MARK}" if i.interrupted else i.content}
            for i in recent
        ]
