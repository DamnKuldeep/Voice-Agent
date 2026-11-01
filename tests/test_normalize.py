import pytest

from voiceagent.llm.normalize import normalize_for_speech as n

CASES = [
    ("Rs 12,500", "twelve thousand five hundred rupees"),
    ("Rs. 500 only", "five hundred rupees only"),
    ("₹1", "one rupee"),
    ("$5", "five dollars"),
    ("It costs $5.50 today", "It costs five dollars fifty cents today"),
    ("$2 million", "two million dollars"),
    ("Rs 5 lakh", "five lakh rupees"),
    ("20%", "twenty percent"),
    ("grew 3.5% last year", "grew three point five percent last year"),
    ("the 1st, 2nd, 3rd and 21st", "the first, second, third and twenty first"),
    ("the 12th and 30th", "the twelfth and thirtieth"),
    ("in 1999", "in nineteen ninety nine"),
    ("in 2005", "in two thousand five"),
    ("in 2024", "in twenty twenty four"),
    ("in 1905", "in nineteen oh five"),
    ("pi is 3.14", "pi is three point one four"),
    ("I have 42 apples", "I have forty two apples"),
    (
        "1,234,567 people",
        "one million two hundred thirty four thousand five hundred sixty seven people",
    ),
    ("call 9876543210", "call nine eight seven six five four three two one zero"),
    ("1, 2, 3", "one, two, three"),
    ("It is -5 degrees", "It is minus five degrees"),
    ("at 10:30", "at ten thirty"),
    ("at 9:05 pm", "at nine oh five p m"),
    ("at 5 pm.", "at five p m."),
    ("run 5 km", "run five kilometers"),
    ("run 1 km", "run one kilometer"),
    ("2.5 kg of rice", "two point five kilograms of rice"),
    ("Dr. Smith and Mr. Rao", "Doctor Smith and Mister Rao"),
    ("fruits, e.g. apples", "fruits, for example, apples"),
    ("apples, etc. Then more", "apples, et cetera. Then more"),
    ("tea & coffee", "tea and coffee"),
    ("2 + 2 = 4", "two plus two equals four"),
    ("**bold** and *italic* and `code`", "bold and italic and code"),
    ("## Heading\nBody text.", "Heading, Body text."),
    ("- apples\n- pears", "apples, pears"),
    ("1. first\n2. second", "first, second"),
    ("see [the docs](https://x.com/a) now", "see the docs now"),
    ("visit https://example.com today", "visit today"),
    ("Great job 😀👍!", "Great job!"),
    ("  lots   of \n\n space  ", "lots of, space"),
    ("```py\nprint(1)\n```Done", "Done"),
    ("snake_case word", "snake case word"),
    ("it’s “fine” — really", 'it\'s "fine", really'),
]


@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_normalize(raw: str, expected: str) -> None:
    assert n(raw) == expected


def test_empty_and_symbol_only() -> None:
    assert n("") == ""
    assert n("***") == ""
    assert n("😀") == ""


def test_huge_number_is_read_by_digit() -> None:
    assert n("1" + "0" * 20).startswith("one zero zero")


DIRECTION_CASES = [
    ("*smiles* Hi there.", "Hi there."),
    ("Sure. *pauses* Let me think.", "Sure. Let me think."),
    ("*Tilts head.* What do you mean?", "What do you mean?"),
    ("That is *really* good.", "That is really good."),
    ("[laughs] Okay.", "Okay."),
    ("Okay (pauses) fine.", "Okay fine."),
    ("Well (laughs softly), maybe.", "Well, maybe."),
    ("It is far (about 5 km) from here.", "It is far (about five kilometers) from here."),
    ("Paris (the capital) is lovely.", "Paris (the capital) is lovely."),
    ("pauses", ""),
    ("Pause.", ""),
    ("remains silent", ""),
    ("Stays silent.", ""),
    ("*remains silent*", ""),
    ("(silence)", ""),
    ("[pause]", ""),
    ("Okay.", "Okay."),
    ("I will stay quiet now.", "I will stay quiet now."),
]


@pytest.mark.parametrize(("raw", "expected"), DIRECTION_CASES)
def test_stage_directions_are_not_spoken(raw: str, expected: str) -> None:
    assert n(raw) == expected
