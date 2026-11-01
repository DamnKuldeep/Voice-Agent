import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from voiceagent.config import Llm
from voiceagent.llm.call_result import CallResult
from voiceagent.llm.openrouter import LlmError, OpenRouterChat


def make_settings(**kw: Any) -> Llm:
    base: dict[str, Any] = {
        "model": "m/main",
        "fallback_models": ["m/fb1", "m/fb2"],
        "routing_sort": "latency",
        "provider_order": [],
        "stream": True,
        "ttft_timeout_s": 0.3,
        "hedge_after_s": 10.0,  # sequential unless a test hedges
        "temperature": 0.5,
        "history_turns": 6,
        "prewarm_connection": True,
        "persona_file": "x",
    }
    return Llm(**{**base, **kw})


def sse(*events: object) -> bytes:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    return "".join(lines).encode()


def delta(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


GOOD = sse(
    {"provider": "Groq", "choices": [{"delta": {"role": "assistant", "content": ""}}]},
    delta("Hi "),
    delta("there"),
    {"choices": [{"delta": {}, "finish_reason": "length"}]},
    {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 1e-6, "x": 1}},
    "[DONE]",
)
MSGS = [{"role": "user", "content": "hi"}]
Handler = Callable[[httpx.Request], httpx.Response]


def chat_with(handler: Handler, **kw: Any) -> OpenRouterChat:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterChat(make_settings(**kw), "test-key", client)


async def collect(chat: OpenRouterChat, cancel: asyncio.Event | None = None) -> list[str]:
    return (await collect_with_result(chat, cancel))[0]


async def collect_with_result(
    chat: OpenRouterChat, cancel: asyncio.Event | None = None
) -> tuple[list[str], CallResult]:
    stream = chat.stream(MSGS, max_tokens=20, cancel=cancel or asyncio.Event())
    return [d async for d in stream], stream.result


def model_of(request: httpx.Request) -> str:
    return str(json.loads(request.content)["model"])


async def test_sse_parsing_and_stats() -> None:
    chat = chat_with(lambda r: httpx.Response(200, content=GOOD))
    deltas, result = await collect_with_result(chat)
    assert deltas == ["Hi ", "there"]
    assert result.model == "m/main"
    assert result.ttft_s is not None and result.ttft_s < 0.3
    assert result.finish_reason == "length"
    assert result.usage == {"prompt_tokens": 5, "completion_tokens": 2, "cost": 1e-6}
    assert result.provider == "Groq"


async def test_request_body_and_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=GOOD)

    await collect(chat_with(handler))
    body = json.loads(seen[0].content)
    assert body["stream"] is True and body["max_tokens"] == 20 and body["temperature"] == 0.5
    assert body["provider"] == {"sort": "latency"}
    assert body["reasoning"] == {"enabled": False}
    assert body["usage"] == {"include": True}
    assert body["model"] == "m/main" and body["messages"] == MSGS
    assert seen[0].headers["authorization"] == "Bearer test-key"


async def test_default_routing_sends_no_provider() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=GOOD)

    await collect(chat_with(handler, routing_sort="default"))
    assert "provider" not in seen[0]


async def test_fallback_on_429_and_5xx() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = model_of(request)
        models.append(model)
        if model == "m/main":
            return httpx.Response(429)
        if model == "m/fb1":
            return httpx.Response(503)
        return httpx.Response(200, content=GOOD)

    deltas, result = await collect_with_result(chat_with(handler))
    assert deltas == ["Hi ", "there"]
    assert models == ["m/main", "m/fb1", "m/fb2"]
    assert result.model == "m/fb2"


async def test_fallback_on_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if model_of(request) == "m/main":
            raise httpx.ConnectError("boom")
        return httpx.Response(200, content=GOOD)

    deltas, result = await collect_with_result(chat_with(handler))
    assert deltas == ["Hi ", "there"]
    assert result.model == "m/fb1"


class SlowStream(httpx.AsyncByteStream):
    """Sends ``first`` and then nothing, to model a model that stalls."""

    def __init__(self, first: bytes = b"") -> None:
        self.first = first
        self.gate = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.first:
            yield self.first
        await self.gate.wait()

    async def aclose(self) -> None:
        self.closed = True


async def test_fallback_on_ttft_timeout() -> None:
    slow = SlowStream()

    def handler(request: httpx.Request) -> httpx.Response:
        if model_of(request) == "m/main":
            return httpx.Response(200, stream=slow)
        return httpx.Response(200, content=GOOD)

    deltas, result = await collect_with_result(chat_with(handler))
    assert deltas == ["Hi ", "there"]
    assert result.model == "m/fb1"
    assert result.ttft_s is not None and result.ttft_s >= 0.3
    assert slow.closed


async def test_no_switch_after_first_token() -> None:
    models: list[str] = []
    stream = SlowStream(first=sse(delta("Partial")))

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(model_of(request))
        return httpx.Response(200, stream=stream)

    chat = chat_with(handler, ttft_timeout_s=0.1)
    cancel = asyncio.Event()
    out: list[str] = []
    async for d in chat.stream(MSGS, max_tokens=5, cancel=cancel):
        out.append(d)
        asyncio.get_running_loop().call_later(0.3, cancel.set)
    assert out == ["Partial"] and models == ["m/main"]


async def test_cancel_mid_stream_stops_promptly_and_closes_response() -> None:
    stream = SlowStream(first=sse(delta("one")))
    chat = chat_with(lambda r: httpx.Response(200, stream=stream), ttft_timeout_s=5)
    cancel = asyncio.Event()
    out: list[str] = []

    async def run() -> None:
        async for d in chat.stream(MSGS, max_tokens=20, cancel=cancel):
            out.append(d)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    cancel.set()
    await asyncio.wait_for(task, 0.5)
    assert out == ["one"] and stream.closed


async def test_cancel_before_first_token() -> None:
    stream = SlowStream()
    chat = chat_with(lambda r: httpx.Response(200, stream=stream), ttft_timeout_s=5)
    cancel = asyncio.Event()
    task = asyncio.create_task(collect(chat, cancel))
    await asyncio.sleep(0.05)
    cancel.set()
    assert await asyncio.wait_for(task, 0.5) == []


async def test_consumer_break_closes_response() -> None:
    stream = SlowStream(first=sse(delta("a"), delta("b")))
    chat = chat_with(lambda r: httpx.Response(200, stream=stream), ttft_timeout_s=5)
    gen = chat.stream(MSGS, max_tokens=5, cancel=asyncio.Event())
    assert await anext(gen) == "a"
    await gen.aclose()
    assert stream.closed


async def test_all_models_fail_raises() -> None:
    chat = chat_with(lambda r: httpx.Response(500))
    with pytest.raises(LlmError):
        await collect(chat)


async def test_midstream_error_ends_stream_with_error_reason() -> None:
    body = sse(delta("ok"), {"error": {"message": "upstream"}})
    deltas, result = await collect_with_result(chat_with(lambda r: httpx.Response(200, content=body)))
    assert deltas == ["ok"]
    assert result.finish_reason == "error"


async def test_comments_and_garbage_lines_ignored() -> None:
    body = b": OPENROUTER PROCESSING\n\n" + sse("not json", delta("x"), "[DONE]")
    chat = chat_with(lambda r: httpx.Response(200, content=body))
    assert await collect(chat) == ["x"]


async def test_prewarm_respects_setting_and_swallows_errors() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        raise httpx.ConnectError("offline")

    await chat_with(handler, prewarm_connection=False).prewarm()
    assert calls == []
    await chat_with(handler).prewarm()
    assert calls == ["/api/v1/key"]


async def test_constructor_does_no_io_and_aclose_keeps_injected_client() -> None:
    chat = OpenRouterChat(make_settings(), "k")
    assert chat._client is None
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    injected = OpenRouterChat(make_settings(), "k", client)
    await injected.aclose()
    assert not client.is_closed
    await client.aclose()


def bodies_of(**kw: Any) -> tuple[OpenRouterChat, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=GOOD)

    return chat_with(handler, **kw), seen


async def test_provider_order_sent_with_fallbacks_and_sort() -> None:
    chat, seen = bodies_of(provider_order=["groq", "cerebras"])
    await collect(chat)
    assert seen[0]["provider"] == {
        "order": ["groq", "cerebras"], "allow_fallbacks": True, "sort": "latency"
    }


async def test_provider_order_without_sort() -> None:
    chat, seen = bodies_of(provider_order=["groq"], routing_sort="default")
    await collect(chat)
    assert seen[0]["provider"] == {"order": ["groq"], "allow_fallbacks": True}


async def test_gpt_oss_gets_lowest_reasoning_effort() -> None:
    chat, seen = bodies_of(model="openai/gpt-oss-120b")
    await collect(chat)
    assert seen[0]["reasoning"] == {"effort": "low", "exclude": True}


async def test_reasoning_deltas_are_not_content() -> None:
    body = sse(
        {"choices": [{"delta": {"reasoning": "Let me think", "content": None}}]},
        {"choices": [{"delta": {"reasoning": " more", "content": ""}}]},
        delta("Answer."),
        "[DONE]",
    )
    chat = chat_with(lambda r: httpx.Response(200, content=body))
    assert await collect(chat) == ["Answer."]


async def test_reasoning_only_reply_counts_as_empty_and_falls_back() -> None:
    thinking = sse({"choices": [{"delta": {"reasoning": "hmm"}}]}, "[DONE]")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=thinking if model_of(request) == "m/main" else GOOD)

    deltas, result = await collect_with_result(chat_with(handler))
    assert deltas == ["Hi ", "there"]
    assert result.model == "m/fb1"


async def test_each_call_has_its_own_result_when_calls_overlap() -> None:
    slow = SlowStream(first=sse({"provider": "SlowCo", **delta("slow")}))

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["messages"][0]["content"] == "slow":
            return httpx.Response(200, stream=slow)
        return httpx.Response(200, content=GOOD)

    chat = chat_with(handler, ttft_timeout_s=5)
    cancel = asyncio.Event()
    first = chat.stream([{"role": "user", "content": "slow"}], max_tokens=5, cancel=cancel)
    assert await anext(first) == "slow"
    second = chat.stream(MSGS, max_tokens=5, cancel=asyncio.Event())
    assert [d async for d in second] == ["Hi ", "there"]
    assert isinstance(first.result, CallResult)
    assert first.result.provider == "SlowCo" and first.result.usage is None
    assert second.result.provider == "Groq" and second.result.finish_reason == "length"
    cancel.set()
    await first.aclose()


async def test_provider_reported_after_first_token_is_recorded() -> None:
    body = sse(delta("a"), {"provider": "Late", "choices": []}, "[DONE]")
    chat = chat_with(lambda r: httpx.Response(200, content=body))
    stream = chat.stream(MSGS, max_tokens=5, cancel=asyncio.Event())
    assert [d async for d in stream] == ["a"]
    assert stream.result.provider == "Late"


async def test_hedge_starts_the_next_model_while_the_first_is_slow() -> None:
    slow = SlowStream()
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(model_of(request))
        if model_of(request) == "m/main":
            return httpx.Response(200, stream=slow)
        return httpx.Response(200, content=GOOD)

    chat = chat_with(handler, hedge_after_s=0.05, ttft_timeout_s=5)
    started = asyncio.get_running_loop().time()
    deltas, result = await collect_with_result(chat)
    assert deltas == ["Hi ", "there"]
    # the fallback answered long before the 5 s first-token timeout of the stalled model
    assert asyncio.get_running_loop().time() - started < 1.0
    assert models == ["m/main", "m/fb1"] and result.model == "m/fb1"
    assert result.provider == "Groq" and result.finish_reason == "length"
    assert slow.closed


async def test_hedge_does_not_start_another_model_when_the_first_is_quick() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(model_of(request))
        return httpx.Response(200, content=GOOD)

    chat = chat_with(handler, hedge_after_s=0.5)
    assert await collect(chat) == ["Hi ", "there"] and models == ["m/main"]


async def test_hedge_moves_on_at_once_when_an_attempt_fails() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(model_of(request))
        if model_of(request) == "m/main":
            return httpx.Response(503)
        return httpx.Response(200, content=GOOD)

    chat = chat_with(handler, hedge_after_s=10.0)
    started = asyncio.get_running_loop().time()
    assert await collect(chat) == ["Hi ", "there"]
    assert asyncio.get_running_loop().time() - started < 1.0 and models == ["m/main", "m/fb1"]
