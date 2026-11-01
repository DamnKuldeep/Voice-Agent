"""What one streamed chat call used and reported, carried by the stream it belongs to."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class CallResult:
    """Model, provider, time to first token, usage and finish reason; filled in while the stream runs."""

    model: str = ""
    provider: str | None = None
    ttft_s: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    def adopt(self, other: CallResult) -> None:
        """Take the winning attempt's values (the attempt keeps filling its own as it streams)."""
        self.model, self.provider, self.ttft_s = other.model, other.provider, other.ttft_s
        self.usage, self.finish_reason = other.usage, other.finish_reason


class ChatStream:
    """Async iterator of text deltas that carries its own CallResult.

    Each call gets its own result, so overlapping calls (speculation next to a committed turn)
    cannot overwrite each other's model, provider or usage.
    """

    def __init__(self, deltas: AsyncIterator[str], result: CallResult) -> None:
        self._deltas = deltas
        self.result = result

    def __aiter__(self) -> ChatStream:
        return self

    async def __anext__(self) -> str:
        return await self._deltas.__anext__()

    async def aclose(self) -> None:
        await self._deltas.aclose()  # type: ignore[attr-defined]
