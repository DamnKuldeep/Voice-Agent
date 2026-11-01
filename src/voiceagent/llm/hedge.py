"""Hedged start for streamed replies: try the next model before the current one gives up.

Waiting out each model's first-token timeout in turn cost 7.5 s in live testing when one network
stall hit all three models. Here the next model starts after ``hedge_after_s`` of silence (or at once when an
attempt fails), the first stream to produce a delta wins, and the others are closed.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Sequence


class _Attempt:
    def __init__(self, index: int, stream: AsyncIterator[str]) -> None:
        self.index = index
        self.stream = stream
        self.first: asyncio.Task[str] = asyncio.ensure_future(anext(stream))

    async def close(self) -> None:
        self.first.cancel()
        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
            await self.first
        with contextlib.suppress(Exception):
            await self.stream.aclose()  # type: ignore[attr-defined]


async def race(
    starters: Sequence[Callable[[], AsyncIterator[str]]],
    hedge_after_s: float,
    cancelled: asyncio.Future[object],
    failure: type[Exception],
    on_failure: Callable[[int, Exception], None],
) -> tuple[int, AsyncIterator[str], str] | None:
    """Return (index, stream, first delta) of the first stream to speak, or None.

    None means ``cancelled`` completed or every stream failed or ended without a delta. Exceptions of
    type ``failure`` are reported through ``on_failure``; anything else propagates.
    """
    pending: list[_Attempt] = []
    launched = 0

    def launch() -> None:
        nonlocal launched
        pending.append(_Attempt(launched, starters[launched]()))
        launched += 1

    launch()
    try:
        while pending:
            more = launched < len(starters)
            done, _ = await asyncio.wait(
                {a.first for a in pending} | {cancelled},
                timeout=hedge_after_s if more else None,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancelled in done:
                return None
            if not done:
                launch()
                continue
            winners = []
            for attempt in [a for a in pending if a.first in done]:
                pending.remove(attempt)
                exc = attempt.first.exception()
                if exc is None:
                    winners.append(attempt)
                elif isinstance(exc, failure):
                    on_failure(attempt.index, exc)
                elif not isinstance(exc, StopAsyncIteration):
                    raise exc
            if winners:
                winners.sort(key=lambda a: a.index)
                for loser in winners[1:]:
                    await loser.close()
                best = winners[0]
                return best.index, best.stream, best.first.result()
            if not pending and launched < len(starters):
                launch()
        return None
    finally:
        for attempt in pending:
            await attempt.close()
