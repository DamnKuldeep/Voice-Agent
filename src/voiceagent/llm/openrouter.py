"""Streaming OpenRouter chat client: provider routing, hedged fallback models, SSE parsing.

A model that fails or stays silent before its first token is replaced by the next fallback model
(started early by the hedge in ``hedge.py``); once a token has arrived, the stream is never switched.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from .call_result import CallResult, ChatStream
from .hedge import race

if TYPE_CHECKING:
    from voiceagent.config import Llm

log = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "cost")
# These models reject reasoning {"enabled": false}; the lowest effort is the closest we can get.
_REASONING_ALWAYS_ON = ("openai/gpt-oss",)


class LlmError(RuntimeError):
    """Every configured model failed before producing a token."""


class _Retry(Exception):
    """This model failed before its first token; try the next one."""


class OpenRouterChat:
    """``stream()`` returns a ChatStream; its ``result`` says which model and provider answered."""

    def __init__(
        self, settings: Llm, api_key: str, client: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                http2=importlib.util.find_spec("h2") is not None,
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
        return self._client

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def prewarm(self) -> None:
        """Open the TLS connection with a free metadata call so the first turn skips the handshake."""
        if not self._settings.prewarm_connection:
            return
        try:
            await self._http().get(f"{BASE_URL}/key", headers=self._headers())
        except httpx.HTTPError as exc:
            log.warning("LLM prewarm failed: %s", type(exc).__name__)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _body(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
        s = self._settings
        reasoning: dict[str, Any] = {"enabled": False}
        if model.startswith(_REASONING_ALWAYS_ON):
            reasoning = {"effort": "low", "exclude": True}
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": s.temperature,
            "max_tokens": max_tokens,
            "reasoning": reasoning,
            "usage": {"include": True},
        }
        provider: dict[str, Any] = {}
        if s.provider_order:
            provider = {"order": list(s.provider_order), "allow_fallbacks": True}
        if s.routing_sort != "default":
            provider["sort"] = s.routing_sort
        if provider:
            body["provider"] = provider
        return body

    def stream(
        self, messages: list[dict[str, str]], *, max_tokens: int, cancel: asyncio.Event
    ) -> ChatStream:
        result = CallResult()
        return ChatStream(self._stream(messages, max_tokens, cancel, result), result)

    async def _stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        cancel: asyncio.Event,
        result: CallResult,
    ) -> AsyncIterator[str]:
        started = time.perf_counter()
        cancel_waiter = asyncio.ensure_future(cancel.wait())
        models = [self._settings.model, *self._settings.fallback_models]
        results = [CallResult() for _ in models]  # one each, so a losing attempt cannot overwrite
        starters = [
            (lambda i=i: self._run_model(models[i], messages, max_tokens, cancel_waiter, started,
                                         results[i]))
            for i in range(len(models))
        ]

        def failed(index: int, exc: Exception) -> None:
            log.warning("LLM model %s failed before first token: %s", models[index], exc)

        won = None
        try:
            won = await race(starters, self._settings.hedge_after_s, cancel_waiter, _Retry, failed)
            if won is None:
                if cancel.is_set():
                    return
                raise LlmError("all models failed before producing a token")
            index, deltas, first = won
            result.adopt(results[index])
            # Closing must also cover a consumer that stops right after the first delta.
            async with contextlib.aclosing(deltas):  # type: ignore[type-var]
                yield first
                async for delta in deltas:
                    result.adopt(results[index])
                    yield delta
        finally:
            if won is not None:
                result.adopt(results[won[0]])
            cancel_waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_waiter

    async def _run_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        cancel_waiter: asyncio.Future[Any],
        started: float,
        result: CallResult,
    ) -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)
        pump = asyncio.ensure_future(self._pump(model, messages, max_tokens, queue))
        deadline = time.perf_counter() + self._settings.ttft_timeout_s
        got_token = False
        provider: str | None = None
        try:
            while True:
                getter = asyncio.ensure_future(queue.get())
                timeout = None if got_token else max(0.0, deadline - time.perf_counter())
                done, _ = await asyncio.wait(
                    {getter, cancel_waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if getter not in done:
                    getter.cancel()
                    if cancel_waiter in done:
                        return
                    raise _Retry("no first token before timeout")
                kind, payload = getter.result()
                if kind == "provider":
                    provider = provider or payload
                    if got_token:
                        result.provider = provider
                elif kind == "delta":
                    if not got_token:
                        got_token = True
                        result.model = model
                        result.provider = provider
                        result.ttft_s = time.perf_counter() - started
                    yield payload
                elif kind == "usage":
                    result.usage = payload
                elif kind == "finish":
                    result.finish_reason = payload
                elif kind == "done":
                    if not got_token:
                        raise _Retry("empty response")
                    return
                else:
                    if not got_token:
                        raise _Retry(payload)
                    log.warning("LLM stream broke after first token: %s", payload)
                    result.finish_reason = "error"
                    return
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    async def _pump(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        queue: asyncio.Queue[tuple[str, Any]],
    ) -> None:
        try:
            async with self._http().stream(
                "POST",
                f"{BASE_URL}/chat/completions",
                json=self._body(model, messages, max_tokens),
                headers=self._headers(),
            ) as response:
                if response.status_code >= 400:
                    await queue.put(("error", f"HTTP {response.status_code}"))
                    return
                async for line in response.aiter_lines():
                    if line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    await _emit_event(data, queue)
            await queue.put(("done", None))
        except httpx.HTTPError as exc:
            await queue.put(("error", type(exc).__name__))


async def _emit_event(data: str, queue: asyncio.Queue[tuple[str, Any]]) -> None:
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return
    if "error" in event:
        await queue.put(("error", "provider error"))
        return
    if provider := event.get("provider"):
        await queue.put(("provider", str(provider)))
    for choice in event.get("choices") or []:
        # Only "content" is speech; "reasoning" deltas from thinking models are never spoken.
        if text := (choice.get("delta") or {}).get("content"):
            await queue.put(("delta", text))
        if reason := choice.get("finish_reason"):
            await queue.put(("finish", reason))
    if usage := event.get("usage"):
        kept = {k: usage[k] for k in _USAGE_KEYS if k in usage}
        details = usage.get("completion_tokens_details") or {}
        if details.get("reasoning_tokens"):
            kept["reasoning_tokens"] = details["reasoning_tokens"]
        await queue.put(("usage", kept))
