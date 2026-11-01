import pytest

from voiceagent.config import LengthPolicy, TokenBudget
from voiceagent.llm.length_policy import (
    choose_mode,
    max_tokens_for,
    style_hint,
)

CFG = LengthPolicy(
    enabled=True,
    brief=TokenBudget(max_tokens=90),
    detail=TokenBudget(max_tokens=220),
    detail_cues=["explain", "in detail", "step by step", "how does", "story"],
)


@pytest.mark.parametrize(
    ("text", "mode"),
    [
        ("What time is it?", "brief"),
        ("Please EXPLAIN gravity", "detail"),
        ("tell me in   detail", "detail"),
        ("Walk me step by step", "detail"),
        ("How does a fridge work", "detail"),
        ("Tell me a story", "detail"),
        ("that is history", "brief"),
        ("unexplained noises", "brief"),
    ],
)
def test_choose_mode(text: str, mode: str) -> None:
    assert choose_mode(text, CFG) == mode


def test_no_cues_is_always_brief() -> None:
    assert choose_mode("explain", CFG.model_copy(update={"detail_cues": []})) == "brief"


def test_max_tokens() -> None:
    assert max_tokens_for("brief", CFG) == 90
    assert max_tokens_for("detail", CFG) == 220


def test_style_hint_is_one_line_per_mode() -> None:
    assert "\n" not in style_hint("brief") and "\n" not in style_hint("detail")
    assert style_hint("brief") != style_hint("detail")
