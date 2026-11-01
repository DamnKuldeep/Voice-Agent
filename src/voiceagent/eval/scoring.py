"""Per-item scores (WER, clipped first and last words) and per-condition summaries as Markdown."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.metrics import percentile
from ..core.text_metrics import normalize_words, word_errors

# Words too common to show that a reply was about what the user said.
_STOPWORDS = frozenset((
    "the", "and", "but", "from", "with", "this", "that", "there", "their", "they", "them", "your",
    "were", "been", "then", "than", "what", "which", "when", "where", "about", "into", "over",
    "after", "before", "also", "very", "just", "tell", "please",
))  # only words over three letters matter: shorter ones are skipped anyway
WINDOW = 2  # a reference edge word counts as heard if it is among this many hypothesis words at that edge


def first_word_missing(reference: str, hypothesis: str) -> bool:
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    return bool(ref) and ref[0] not in hyp[:WINDOW]


def last_word_missing(reference: str, hypothesis: str) -> bool:
    """True when the reference's last word is not among the hypothesis's last words: a clipped ending
    ("history of plas") or a dropped one. A misrecognised last word also counts, so read it with WER."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    return bool(ref) and ref[-1] not in hyp[-WINDOW:]


def content_overlap(reference: str, reply: str) -> bool:
    """Heuristic only: does the reply share a content word with what the user said."""
    ref = {w for w in normalize_words(reference) if len(w) > 3 and w not in _STOPWORDS}
    return bool(ref & set(normalize_words(reply)))


def transcript_scores(reference: str, hypothesis: str) -> dict[str, object]:
    errors, words = word_errors(reference, hypothesis)
    return {
        "wer_errors": errors, "wer_words": words, "wer": errors / words if words else 0.0,
        "first_word_missing": first_word_missing(reference, hypothesis),
        "last_word_missing": last_word_missing(reference, hypothesis),
    }


def summarise(rows: Iterable[dict]) -> dict[str, dict]:
    """Per condition: item count, corpus WER, the rate of every check and p50/p90 of every metric.

    Rows carry ``condition``, optional ``wer_errors``/``wer_words``, ``checks`` (name -> bool, or None
    when the check does not apply to that item) and ``metrics`` (name -> ms). Unknown names are
    summarised like known ones, so metrics the pipeline adds later appear without changes here.
    """
    out: dict[str, dict] = {}
    for row in rows:
        cell = out.setdefault(row["condition"], {"n": 0, "errors": 0, "words": 0, "checks": {},
                                                 "metrics": {}})
        cell["n"] += 1
        if row.get("wer_words"):
            cell["errors"] += row["wer_errors"]
            cell["words"] += row["wer_words"]
        for name, value in row.get("checks", {}).items():
            if value is not None:
                cell["checks"].setdefault(name, []).append(bool(value))
        for name, value in row.get("metrics", {}).items():
            if value is not None:
                cell["metrics"].setdefault(name, []).append(float(value))
    for cell in out.values():
        words = cell.pop("words")
        errors = cell.pop("errors")
        cell["wer"] = (errors / words, words) if words else None
        cell["checks"] = {k: (sum(v), len(v)) for k, v in sorted(cell["checks"].items())}
        cell["metrics"] = {k: (percentile(v, 50), percentile(v, 90), len(v))
                           for k, v in sorted(cell["metrics"].items())}
    return out


def markdown_table(summary: dict[str, dict]) -> str:
    """Quantities down, conditions across, so conditions sit side by side for comparison."""
    conditions = list(summary)
    lines = ["| quantity | " + " | ".join(f"`{c}`" for c in conditions) + " |",
             "|---|" + "---|" * len(conditions)]

    def row(name: str, cells: Iterable[str]) -> None:
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    row("items", (str(summary[c]["n"]) for c in conditions))
    if any(summary[c]["wer"] for c in conditions):
        row("WER (reference words)", (_wer(summary[c]["wer"]) for c in conditions))
    for name in _union(summary, "checks"):
        row(f"{name} rate", (_rate(summary[c]["checks"].get(name)) for c in conditions))
    for name in _union(summary, "metrics"):
        row(f"{name} p50 / p90 (n)", (_spread(summary[c]["metrics"].get(name)) for c in conditions))
    return "\n".join(lines)


def _wer(value: tuple[float, int] | None) -> str:
    return f"{value[0]:.1%} ({value[1]})" if value else "-"


def _rate(value: tuple[int, int] | None) -> str:
    return f"{value[0] / value[1]:.0%} ({value[0]}/{value[1]})" if value else "-"


def _spread(value: tuple[float, float, int] | None) -> str:
    return f"{value[0]:.0f} / {value[1]:.0f} ({value[2]})" if value else "-"


def _union(summary: dict[str, dict], key: str) -> list[str]:
    return sorted({name for cell in summary.values() for name in cell[key]})
