"""Voice-agent LLM benchmark: realistic requests through the real OpenRouterChat client.

Each request is the persona plus a six-exchange conversation and a new user line, streamed with
the brief token budget. Conditions (model + provider routing) are interleaved in a shuffled order
every round so they share network conditions. Reports TTFT, time to the first eight words (what
the speech chunker needs for its first chunk), total time, provider actually used and cost.

    python scripts/bench_llm.py --n 20
    python scripts/bench_llm.py --spotcheck        # replies verbatim, for a quality read
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import httpx
from dotenv import load_dotenv

from voiceagent.config import ROOT, load_settings
from voiceagent.core.metrics import percentile
from voiceagent.llm.length_policy import choose_mode, max_tokens_for
from voiceagent.llm.openrouter import LlmError, OpenRouterChat
from voiceagent.llm.persona import build_messages, load_persona

# label -> (model, provider_order, routing_sort); a pinned provider still allows fallbacks, as in
# production, and the provider column shows who actually served.
CONDITIONS: dict[str, tuple[str, list[str], str]] = {
    "llama-3.1-8b (sort=latency)": ("meta-llama/llama-3.1-8b-instruct", [], "latency"),
    "llama-3.1-8b @groq": ("meta-llama/llama-3.1-8b-instruct", ["groq"], "latency"),
    "llama-3.3-70b @coreweave": ("meta-llama/llama-3.3-70b-instruct", ["coreweave"], "latency"),
    "llama-3.3-70b @groq": ("meta-llama/llama-3.3-70b-instruct", ["groq"], "latency"),
    "llama-3.3-70b @groq>coreweave": (
        "meta-llama/llama-3.3-70b-instruct", ["groq", "coreweave"], "latency"
    ),
    "gpt-oss-120b @groq": ("openai/gpt-oss-120b", ["groq"], "latency"),
    "gpt-oss-120b @cerebras": ("openai/gpt-oss-120b", ["cerebras"], "latency"),
    "gpt-oss-20b @coreweave": ("openai/gpt-oss-20b", ["coreweave"], "latency"),
    "deepseek-v4.1-flash (sort=latency)": ("deepseek/deepseek-v4.1-flash", [], "latency"),
}

CONVERSATIONS: list[tuple[list[str], str]] = [
    (["Hi, who am I talking to?", "I'm Mira. What can I do for you?",
      "Tell me a fact about history.", "The Great Fire of London in 1666 burned for four days.",
      "Wow, how big was it?", "It destroyed over thirteen thousand houses and most of the city.",
      "Did many people die?", "Officially only a handful, though the real number is unknown.",
      "Why so few?", "Most people had time to flee, since the fire spread over days.",
      "Interesting.", "It also led London to rebuild in brick and stone."],
     "And how did it actually start?"),
    (["I'm planning a trip to Japan.", "Lovely! When are you going?",
      "In April.", "That's cherry blossom season, a beautiful time to visit.",
      "Is it crowded then?", "Yes, popular parks get very busy, especially on weekends.",
      "Where should I stay in Tokyo?", "Shinjuku or Shibuya are lively and well connected.",
      "And for a quieter place?", "Try Asakusa or Yanaka for an older, calmer feel.",
      "How long should I stay in Tokyo?", "Four or five days covers the highlights."],
     "What about Kyoto, is two days enough?"),
    (["Can you help me study for a physics exam?", "Sure. Which topic?",
      "Newton's laws.", "Good choice. Which law is giving you trouble?",
      "The second one.", "Force equals mass times acceleration.",
      "So heavier things need more force?", "Yes, to get the same acceleration.",
      "What units is force in?", "Newtons. One newton accelerates one kilogram by one meter per second squared.",
      "Got it.", "Want to try a quick practice question?"],
     "Yes, give me one."),
    (["I've been feeling a bit tired lately.", "Sorry to hear that. Are you sleeping well?",
      "Not really, maybe six hours.", "Most adults need seven to nine hours.",
      "I drink a lot of coffee too.", "Caffeine late in the day can make sleep worse.",
      "I usually have one at 5 pm.", "Try moving that to before two in the afternoon.",
      "Okay, anything else?", "A regular bedtime and less screen time at night help too.",
      "I'll try that.", "Great. Small changes add up."],
     "How long before I notice a difference?"),
]

SPOTCHECK: list[tuple[str, list[str], str]] = [
    ("follow-up needs history", CONVERSATIONS[0][0], "And how did it actually start?"),
    ("please stay silent", ["Tell me a joke.", "Why did the scarecrow win an award? He was outstanding in his field."],
     "Okay, now please just stay silent."),
    ("cut-off user line", [], "Tell me about the history of plas"),
    ("garbled word", ["Tell me about quantum mechanics.", "It's the physics of very small things, like atoms."],
     "I meant one term mechan"),
    ("request for detail", [], "Can you explain in detail how vaccines train the immune system?"),
    ("small talk", [], "Hey, how's it going?"),
    ("number-heavy", [], "If I invest 12,500 rupees at 8% a year, how much will I have after 3 years?"),
    ("follow-up pronoun", CONVERSATIONS[1][0], "Is it expensive there?"),
]


def history_messages(lines: list[str]) -> list[dict[str, str]]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(lines)]


def make_chat(base, model: str, order: list[str], sort: str, client: httpx.AsyncClient,
              key: str) -> OpenRouterChat:
    llm = base.llm.model_copy(update={"model": model, "fallback_models": [], "provider_order": order,
                                      "routing_sort": sort, "ttft_timeout_s": 10.0})
    return OpenRouterChat(llm, key, client)


async def one_call(chat: OpenRouterChat, messages: list[dict[str, str]], max_tokens: int) -> dict:
    start = time.perf_counter()
    text, t8 = "", None
    stream = chat.stream(messages, max_tokens=max_tokens, cancel=asyncio.Event())
    try:
        async for delta in stream:
            text += delta
            if t8 is None and len(text.split()) >= 8:
                t8 = time.perf_counter() - start
    except (LlmError, httpx.HTTPError) as exc:
        return {"error": type(exc).__name__}
    total = time.perf_counter() - start
    r = stream.result
    return {"error": None, "ttft": r.ttft_s, "t8": t8 if t8 is not None else total, "total": total,
            "words": len(text.split()), "provider": r.provider, "finish": r.finish_reason,
            "cost": (r.usage or {}).get("cost"),
            "reasoning_tokens": (r.usage or {}).get("reasoning_tokens", 0), "text": text}


def p(values: list[float], q: float) -> str:
    return f"{percentile(values, q):.2f}" if values else "-"


def report(results: dict[str, list[dict]]) -> None:
    print("| Condition | n ok | TTFT p50 | TTFT p90 | TTFT max | 8 words p50 | 8 words p90 | "
          "Total p50 | Words p50 | Cut at limit | Reasoning tok p50 | Errors | Provider used | "
          "$/1k turns |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, rows in results.items():
        ok = [r for r in rows if not r["error"]]
        ttft = [r["ttft"] for r in ok]
        costs = [r["cost"] for r in ok if r["cost"] is not None]
        provs = ", ".join(f"{k} {v}" for k, v in Counter(r["provider"] for r in ok).most_common())
        cost = f"{1000 * sum(costs) / len(costs):.3f}" if costs else "-"
        print(f"| {label} | {len(ok)} | {p(ttft, 50)} | {p(ttft, 90)} | "
              f"{max(ttft) if ttft else 0:.2f} | {p([r['t8'] for r in ok], 50)} | "
              f"{p([r['t8'] for r in ok], 90)} | {p([r['total'] for r in ok], 50)} | "
              f"{p([float(r['words']) for r in ok], 50)} | "
              f"{sum(r['finish'] == 'length' for r in ok)} | "
              f"{p([float(r['reasoning_tokens']) for r in ok], 50)} | "
              f"{len(rows) - len(ok)} | {provs} | {cost} |")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="timed requests per condition")
    parser.add_argument("--only", nargs="*", help="substrings of condition labels to run")
    parser.add_argument("--spotcheck", action="store_true")
    parser.add_argument("--json", help="also write raw rows to this file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.environ["OPENROUTER_API_KEY"]
    settings = load_settings()
    persona = load_persona(settings.llm.persona_file)
    conditions = {k: v for k, v in CONDITIONS.items()
                  if not args.only or any(s in k for s in args.only)}
    rng = random.Random(args.seed)
    async with httpx.AsyncClient(http2=True, timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        chats = {k: make_chat(settings, *v, client, key) for k, v in conditions.items()}
        warm = history_messages([]) + [{"role": "user", "content": "Hi."}]
        for chat in chats.values():  # first call per route pays DNS, TLS and provider cold start
            await one_call(chat, [{"role": "system", "content": persona}, *warm], 8)
        if args.spotcheck:
            await spotcheck(chats, persona, settings)
            return
        results: dict[str, list[dict]] = {k: [] for k in chats}
        for i in range(args.n):
            history, line = CONVERSATIONS[i % len(CONVERSATIONS)]
            messages = build_messages(persona, history_messages(history), line, "brief", True)
            order = list(chats)
            rng.shuffle(order)
            for label in order:
                results[label].append(await one_call(chats[label], messages, 90))
    print(f"n = {args.n} requests per condition, seconds, brief budget (90 tokens), "
          "interleaved, warm-up excluded (Measured, this machine)\n")
    report(results)
    if args.json:
        await asyncio.to_thread(
            Path(args.json).write_text, json.dumps(results, indent=1), encoding="utf-8"
        )


async def spotcheck(chats: dict[str, OpenRouterChat], persona: str, settings) -> None:
    for case, history, line in SPOTCHECK:
        mode = choose_mode(line, settings.length_policy)
        messages = build_messages(persona, history_messages(history), line, mode, True)
        print(f"\n### {case}: \"{line}\" ({mode})\n")
        for label, chat in chats.items():
            r = await one_call(chat, messages, max_tokens_for(mode, settings.length_policy))
            reply = r.get("text", "").strip().replace("\n", " ") or f"[{r['error'] or 'empty'}]"
            print(f"- **{label}**: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
