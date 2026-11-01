"""Draw the README and docs charts from docs/results.json, each in a light and a dark version.

    python scripts/plot_results.py        # writes docs/assets/<chart>-light.png and -dark.png
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "docs" / "results.json").read_text(encoding="utf-8"))
OUT = ROOT / "docs" / "assets"

# Validated reference palette (categorical slots in fixed order) and surfaces, per theme.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "base": "#b5b4ad", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33332f",
             "base": "#6b6a64", "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]},
}


def style(ax, t: dict, *, xgrid: bool = False) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    if xgrid:
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    else:
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t: dict, w: float, h: float, cols: int = 1):
    fig, axes = plt.subplots(1, cols, figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def title(fig, t: dict, text: str, sub: str) -> None:
    top = 1 - 0.08 / fig.get_figheight()  # fixed gaps in inches, whatever the figure height
    fig.text(0.02, top, text, color=t["text"], fontsize=12.5, weight="bold", va="top")
    fig.text(0.02, top - 0.27 / fig.get_figheight(), sub, color=t["muted"], fontsize=9, va="top")


def latency(t: dict):
    d = DATA["latency_stages_p50_ms"]
    stages = {k: v for k, v in d.items() if not k.startswith("_")}
    v2v = DATA["voice_to_voice_ms"]["after"]
    fig, ax = figure(t, 9, 2.6)
    left = 0
    for i, (name, ms) in enumerate(stages.items()):
        ax.barh(0, ms - 2, left=left, height=0.5, color=t["series"][i], label=f"{name}  {ms} ms")
        if ms >= 120:
            ax.text(left + ms / 2, 0, f"{ms}", ha="center", va="center", color="#ffffff", fontsize=9, weight="bold")
        left += ms
    style(ax, t, xgrid=True)
    ax.set_yticks([])
    ax.set_xlim(0, 1100)
    ax.set_xlabel("milliseconds after you stop speaking (stage medians)", color=t["muted"], fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False, fontsize=8.5,
              labelcolor=t["text"], handlelength=1, columnspacing=1.6)
    title(fig, t, f"Where the time goes: voice-to-voice p50 {v2v['p50']:,} ms, p90 {v2v['p90']:,} ms",
          f"Median of each stage, n={d['_n']} spoken questions; stage medians do not sum exactly to the voice-to-voice median")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.74, bottom=0.42)
    return fig


def before_after(t: dict):
    lat, err = DATA["voice_to_voice_ms"], DATA["listening_errors_pct"]
    fig, (a1, a2) = figure(t, 9.5, 3.4, cols=2)
    colors = [t["base"], t["series"][0]]
    for ax, groups, unit in ((a1, ["p50", "p90"], "ms"), (a2, list(err["after"]), "%")):
        source = lat if unit == "ms" else err
        for j, when in enumerate(("before", "after")):
            xs = [i + (j - 0.5) * 0.36 for i in range(len(groups))]
            vals = [source[when][g] for g in groups]
            ax.bar(xs, vals, width=0.34, color=colors[j], label=when.capitalize())
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:,}{'' if unit == 'ms' else '%'}", ha="center", va="bottom", fontsize=8.5,
                        color=t["text"])
        style(ax, t)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["Voice-to-voice p50", "Voice-to-voice p90"] if unit == "ms"
                           else ["Last word\nmissing/wrong", "Sentence split\n(early commit)", "Word error\nrate"],
                           color=t["text"], fontsize=8.5)
        ax.set_ylabel("ms (lower is better)" if unit == "ms" else "% (lower is better)", color=t["muted"], fontsize=9)
    a1.legend(frameon=False, fontsize=9, labelcolor=t["text"], loc="upper left")
    title(fig, t, "First version vs released defaults",
          f"Same 30 public clips for both, replayed at real time through the full pipeline (n={lat['_n']})")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.2, wspace=0.28)
    return fig


def contention(t: dict):
    d = DATA["asr_rtf_contention"]
    fig, ax = figure(t, 8.5, 3.3)
    for j, key in enumerate(("160 ms chunks", "560 ms chunks")):
        xs = [i + (j - 0.5) * 0.36 for i in range(len(d["conditions"]))]
        ax.bar(xs, d[key], width=0.34, color=t["series"][j], label=key)
        for x, v in zip(xs, d[key]):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=t["text"])
    ax.axhline(1.0, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
    ax.text(-0.45, 1.02, "real time", color=t["muted"], fontsize=8.5, ha="left", va="bottom")
    style(ax, t)
    ax.set_xticks(range(len(d["conditions"])))
    ax.set_xticklabels(d["conditions"], color=t["text"], fontsize=9)
    ax.set_ylabel("real-time factor", color=t["muted"], fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=t["text"], loc="upper left")
    title(fig, t, "Why the recogniser uses 560 ms chunks",
          f"Streaming ASR compute per second of audio on the shared 4 GB GPU, n={d['_n']} clips per bar")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.12)
    return fig


def accent(t: dict):
    d = DATA["asr_wer_by_source_pct"]
    fig, ax = figure(t, 8.5, 3.3)
    for j, key in enumerate(("160 ms chunks", "560 ms chunks")):
        xs = [i + (j - 0.5) * 0.36 for i in range(len(d["sources"]))]
        ax.bar(xs, d[key], width=0.34, color=t["series"][j], label=key)
        for x, v in zip(xs, d[key]):
            ax.text(x, v, f"{v}%", ha="center", va="bottom", fontsize=8.5, color=t["text"])
    style(ax, t)
    ax.set_xticks(range(len(d["sources"])))
    ax.set_xticklabels(d["sources"], color=t["text"], fontsize=9)
    ax.set_ylabel("word error rate, %", color=t["muted"], fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=t["text"], loc="upper left")
    title(fig, t, "Accuracy by accent: the remaining gap is the recogniser, not the pipeline",
          f"Nemotron Speech Streaming EN 0.6B, fp16, {d['_n']}")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.12)
    return fig


def interruption(t: dict):
    d = DATA["interruption_ms"]
    fig, ax = figure(t, 9, 2.9)
    rows = [("Agent yields (voice ducks)", None, d["after"]["yield_p50"], None, d["after"]["yield_p90"]),
            ("Agent stops", d["before"]["stop_p50"], d["after"]["stop_p50"], d["before"]["stop_p90"], d["after"]["stop_p90"])]
    for i, (name, b50, a50, b90, a90) in enumerate(rows):
        y = len(rows) - 1 - i
        if b50 is not None:
            ax.barh(y + 0.19, b50, height=0.34, color=t["base"], label="Before" if i == 1 else None)
            ax.text(b50 + 15, y + 0.19, f"{b50} ms (p90 {b90:,})", va="center", fontsize=8.5, color=t["text"])
        else:
            ax.text(15, y + 0.19, "did not yield until it stopped", va="center", fontsize=8.5, color=t["muted"])
        ax.barh(y - 0.19, a50, height=0.34, color=t["series"][0], label="After" if i == 1 else None)
        ax.text(a50 + 15, y - 0.19, f"{a50} ms (p90 {a90})", va="center", fontsize=8.5, color=t["text"])
    style(ax, t, xgrid=True)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=t["text"], fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("ms after voice activity is detected in the interrupting speech (p50)", color=t["muted"], fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=t["text"], loc="lower right")
    title(fig, t, "Interruptions: the agent yields in ~130 ms, then the words decide",
          "The full stop still waits for a real word or 700 ms of speech, so 'yeah' and coughs do not stop it (0/3 each)")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.74, bottom=0.18)
    return fig


def llm(t: dict):
    rows = DATA["llm_ttft_s"]["rows"]
    fig, ax = figure(t, 10, 3.4)
    for i, r in enumerate(reversed(rows)):
        color = t["series"][0] if r.get("chosen") else t["base"]
        ax.barh(i, r["p50"], height=0.55, color=color)
        ax.plot([r["p50"], r["p90"]], [i, i], color=t["muted"], linewidth=1)
        ax.plot([r["p90"]], [i], marker="|", markersize=9, color=t["muted"])
        ax.text(max(r["p90"], r["p50"]) + 0.06, i, f"{r['p50']:.2f} / {r['p90']:.2f} s   ${r['usd_per_1k_turns']} per 1k turns",
                va="center", fontsize=8, color=t["text"])
    style(ax, t, xgrid=True)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)], color=t["text"], fontsize=8.5)
    ax.set_xlim(0, 6.2)
    ax.set_xlabel("time to first token, seconds (bar p50, whisker to p90)", color=t["muted"], fontsize=9)
    title(fig, t, "Choosing the LLM: first-token latency from this network",
          f"OpenRouter, realistic requests with conversation history, {DATA['llm_ttft_s']['_n']}")
    fig.subplots_adjust(left=0.36, right=0.98, top=0.8, bottom=0.15)
    return fig


def vram(t: dict):
    d = DATA["vram_mib"]
    fig, ax = figure(t, 9, 2.6)
    other, agent = d["Windows and other apps"], d["Voice agent (CUDA context + ASR + TTS)"]
    ax.barh(0, other - 6, height=0.5, color=t["base"], label=f"Windows and other apps {other} MiB")
    ax.barh(0, agent - 6, left=other, height=0.5, color=t["series"][0], label=f"Voice agent (CUDA context + ASR + TTS) {agent:,} MiB")
    ax.axvline(d["budget"], color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
    ax.text(d["budget"] - 20, 0.42, f"budget {d['budget']:,} MiB", color=t["muted"], fontsize=8.5, ha="right")
    style(ax, t, xgrid=True)
    ax.set_yticks([])
    ax.set_xlim(0, d["total"])
    ax.set_xlabel(f"GPU memory, MiB (of {d['total']:,})", color=t["muted"], fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=t["text"], handlelength=1)
    title(fig, t, f"Peak GPU memory: {(other + agent) / 1024:.2f} GiB of 4 GiB",
          "Both speech models resident; " + d["_source"].replace("both models resident, ", ""))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.7, bottom=0.36)
    return fig


def lanes(ax, t: dict, names: list[str]) -> None:
    style(ax, t, xgrid=True)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), color=t["text"], fontsize=9)
    ax.set_ylim(-0.6, len(names) - 0.4)


def span(ax, t: dict, lane: int, start: float, end: float, color: str, label: str = "", *,
         alpha: float = 1.0, inside: bool = True) -> None:
    """A bar on a lane; ``inside`` labels sit on the bar (long bars), otherwise just to its right."""
    ax.barh(lane, end - start, left=start, height=0.56, color=color, alpha=alpha,
            edgecolor=t["surface"], linewidth=1.5)
    if label:
        ax.text((start + end) / 2 if inside else end + 25, lane, label, va="center",
                ha="center" if inside else "left", fontsize=8.5,
                color=("#ffffff" if alpha > 0.5 and color != t["base"] else t["text"]) if inside else t["text"],
                weight="bold" if inside else "normal")


def marker(ax, t: dict, x: float, text: str, top: float, ha: str = "center") -> None:
    ax.axvline(x, color=t["muted"], linestyle=(0, (3, 3)), linewidth=1)
    nudge = {"left": 12, "right": -12}.get(ha, 0)
    ax.text(x + nudge, top, text, ha=ha, va="bottom", fontsize=8.5, color=t["text"], weight="bold")


def turn_timeline(t: dict):
    """One turn on a shared time axis: what runs while you speak and in the second after you stop."""
    s = {k: v for k, v in DATA["latency_stages_p50_ms"].items() if not k.startswith("_")}
    v2v = DATA["voice_to_voice_ms"]["after"]["p50"]
    end_wait = s["Wait until sure you finished"]
    final = end_wait + s["Final transcript"]
    token = final + s["LLM first token"]
    chunk = token + s["First speakable chunk"]
    audio = chunk + s["TTS first audio"]
    names = ["You", "Recogniser (streaming)", "Turn decision", "Language model", "Speech synthesis",
             "Agent"]
    row = {n: len(names) - 1 - i for i, n in enumerate(names)}
    c = t["series"]
    fig, ax = figure(t, 10, 3.9)
    lanes(ax, t, names)
    span(ax, t, row["You"], -1800, 0, t["base"], "you are speaking")
    span(ax, t, row["Recogniser (streaming)"], -1700, 0, c[0], "words transcribed as you speak", alpha=0.55)
    span(ax, t, row["Recogniser (streaming)"], end_wait, final, c[0],
         f"final transcript {s['Final transcript']} ms", inside=False)
    span(ax, t, row["Turn decision"], 0, end_wait, c[1], f"are you done? {end_wait} ms", inside=False)
    span(ax, t, row["Language model"], final, token, c[2])
    span(ax, t, row["Language model"], token, token + 1200, c[2],
         f"first token after {s['LLM first token']} ms, then streams", alpha=0.3)
    span(ax, t, row["Speech synthesis"], token, chunk, c[3])
    span(ax, t, row["Speech synthesis"], chunk, audio, c[3],
         f"first chunk ready, first audio {s['TTS first audio']} ms later", inside=False)
    span(ax, t, row["Agent"], v2v, v2v + 1300, c[4], "you hear the answer")
    marker(ax, t, 0, "you stop", len(names) - 0.45)
    marker(ax, t, v2v, f"{v2v:,} ms", len(names) - 0.45)
    ax.set_xlim(-1900, 2350)
    ax.set_xlabel("milliseconds relative to the end of your speech (stage medians)", color=t["muted"],
                  fontsize=9)
    title(fig, t, "One turn, end to end: about a second from your last word to its first word",
          "The recogniser works while you talk; the language model and speech synthesis overlap, "
          "so the reply starts before it is written")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.76, bottom=0.14)
    return fig


def barge_timeline(t: dict):
    """The agent is talking; you start to talk over it at 0 ms."""
    d = DATA["interruption_ms"]["after"]
    duck, stop = d["yield_p50"], d["stop_p50"]
    names = ["Agent's voice", "You", "Barge-in controller"]
    row = {n: len(names) - 1 - i for i, n in enumerate(names)}
    c = t["series"]
    fig, ax = figure(t, 10, 3.0)
    lanes(ax, t, names)
    span(ax, t, row["Agent's voice"], -1500, duck, c[0], "full volume")
    span(ax, t, row["Agent's voice"], duck, stop, c[0], "ducked to 25%", alpha=0.4)
    span(ax, t, row["You"], 0, 2300, t["base"], "\"wait, actually, what about ...\"")
    span(ax, t, row["Barge-in controller"], 0, duck, c[1], "")
    span(ax, t, row["Barge-in controller"], duck, stop, c[1], "words decide", alpha=0.4)
    marker(ax, t, 0, "you start", len(names) - 0.45, ha="right")
    marker(ax, t, duck, f"yields {duck} ms", len(names) - 0.45, ha="left")
    marker(ax, t, stop, f"stops {stop} ms", len(names) - 0.45, ha="left")
    ax.text(stop + 30, row["Agent's voice"], "silent: reply cancelled, history trimmed",
            va="center", fontsize=8.5, color=t["muted"])
    ax.set_xlim(-1600, 2400)
    ax.set_xlabel("milliseconds after voice activity is detected in your speech (medians)",
                  color=t["muted"], fontsize=9)
    title(fig, t, "Anatomy of an interruption: yield at once, decide by the words",
          "A real word (or 700 ms of speech) stops it; \"yeah\", \"mm-hmm\" and coughs only dip the volume "
          "and the reply carries on")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.7, bottom=0.18)
    return fig


CHARTS: dict[str, Callable] = {
    "turn-timeline": turn_timeline, "barge-timeline": barge_timeline,
    "latency-breakdown": latency, "before-after": before_after, "asr-contention": contention,
    "accent-wer": accent, "interruption": interruption, "llm-ttft": llm, "vram": vram,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in CHARTS.items():
        for theme, t in THEMES.items():
            fig = draw(t)
            fig.savefig(OUT / f"{name}-{theme}.png", facecolor=t["surface"])
            plt.close(fig)
    print(f"wrote {len(CHARTS) * len(THEMES)} images to {OUT}")


if __name__ == "__main__":
    main()
