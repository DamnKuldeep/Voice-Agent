"""Before/after measurements on fixed inputs, through real sessions at real time (docs/benchmarks.md).

    python scripts/run_eval.py --set listening --n 40 --tag baseline
    python scripts/run_eval.py --set listening --n 40 --tag chunk_size \
        --compare asr.chunk_ms=160 --compare asr.chunk_ms=560
    python scripts/run_eval.py --set interruptions --n 4 --tag barge \
        --compare barge_in.min_speech_ms=150 --compare barge_in.min_speech_ms=250

listening: one sentence per fresh session from a public manifest (source-balanced subset of --n).
interruptions: the scenarios in evals/sets/interruptions.yaml, --n rounds of each.
Writes evals/results/<tag>.jsonl (one line per item per condition) and evals/results/<tag>.md.
Loads the real models: nothing else may hold the GPU while this runs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from voiceagent.config import DEFAULT_CONFIG, ROOT, Settings, load_settings
from voiceagent.eval.conditions import Condition, parse_conditions
from voiceagent.eval.dataset import Utterance, balanced_subset, load_audio, load_manifest
from voiceagent.eval.interruptions import Stimuli, run_trial
from voiceagent.eval.listening import run_listening
from voiceagent.eval.runner import ComponentSlot, run_conditions, write_report
from voiceagent.eval.stimuli import expand, load_scenarios

RESULTS = ROOT / "evals" / "results"
WARMUP_QUESTION = "What is the capital of France?"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--set", choices=["listening", "interruptions"], required=True)
    parser.add_argument("--tag", required=True, help="output name under evals/results/")
    parser.add_argument("--n", type=int, default=40,
                        help="listening: clips; interruptions: rounds of every scenario")
    parser.add_argument("--manifest", default="public_utterances.yaml", help="listening set manifest")
    parser.add_argument("--scenarios", default="interruptions.yaml", help="interruption scenarios")
    parser.add_argument("--compare", nargs="+", action="append", metavar="KEY=VALUE",
                        help="one condition (repeat the flag for more); values are parsed as YAML")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def build(settings: Settings):
    from voiceagent.pipeline.factory import build_components

    print("loading models ...", flush=True)
    started = time.perf_counter()
    components = build_components(settings, progress=lambda msg: print(f"  {msg}", flush=True))
    print(f"models ready in {time.perf_counter() - started:.0f} s", flush=True)
    return components


async def main_async(args: argparse.Namespace) -> None:
    conditions = parse_conditions(args.compare)

    def settings_for(cond: Condition) -> Settings:
        return load_settings(args.config, overrides=cond.overrides or None)

    stimuli = Stimuli()

    async def warmup(settings: Settings, components) -> None:
        audio = await stimuli.speak(components, WARMUP_QUESTION)
        await run_listening(Utterance("warmup", "", WARMUP_QUESTION, "synthetic"), audio, settings,
                            components)

    if args.set == "listening":
        items = balanced_subset(load_manifest(args.manifest), args.n)

        async def episode(item, settings, components):
            return await run_listening(item, load_audio(item), settings, components)

        what = f"{len(items)} clips from `{args.manifest}` (sources: "
        what += ", ".join(sorted({i.source for i in items})) + ")"
    else:
        spec = load_scenarios(args.scenarios)
        items = expand(spec, args.n)

        async def episode(item, settings, components):
            return await run_trial(item, spec, stimuli, settings, components)

        what = (f"{len(items)} trials from `{args.scenarios}` ({args.n} rounds of "
                + ", ".join(s.id for s in spec.scenarios) + f"); question: \"{spec.question}\"")

    slot = ComponentSlot(build, warmup)
    started = time.perf_counter()
    try:
        rows = await run_conditions(items, conditions, settings_for, episode, slot,
                                    RESULTS / f"{args.tag}.jsonl", lambda item: item.id)
    finally:
        slot.close()
    minutes = (time.perf_counter() - started) / 60
    notes = [
        f"- Set `{args.set}`: {what}.",
        "- Conditions (interleaved item by item): " + "; ".join(f"`{c.label}`" for c in conditions) + ".",
        ("- Measured on this machine with the real models, replayed at 1x; one fresh session per item; "
         "the first session after each model load is a warm-up and is excluded."),
        f"- Wall time {minutes:.1f} min, {slot.builds} model load(s). Raw rows: `{args.tag}.jsonl`.",
        ("- Rates count items where the check applies; `*_heuristic` checks are word-overlap guesses, "
         "not judgements."),
    ]
    table = write_report(rows, RESULTS / f"{args.tag}.md", title=f"Evaluation `{args.tag}`", notes=notes)
    print("\n" + table)
    print(f"\nwrote {RESULTS / (args.tag + '.md')} ({minutes:.1f} min)")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()
