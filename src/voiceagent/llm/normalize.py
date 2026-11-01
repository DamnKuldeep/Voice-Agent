"""Turn LLM text into something a TTS engine reads aloud correctly."""

from __future__ import annotations

import re
import unicodedata

from voiceagent.llm.directions import strip_directions

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand"))
_ORDINAL_IRREGULAR = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SCALE_WORDS = {
    "k": "thousand",
    "thousand": "thousand",
    "lakh": "lakh",
    "lakhs": "lakh",
    "crore": "crore",
    "crores": "crore",
    "million": "million",
    "mn": "million",
    "billion": "billion",
    "bn": "billion",
    "trillion": "trillion",
}
_RUPEE = ("rupee", "rupees", "paisa", "paise")
_DOLLAR = ("dollar", "dollars", "cent", "cents")
# key -> (singular, plural, minor singular, minor plural)
_CURRENCIES = {
    "rs": _RUPEE,
    "inr": _RUPEE,
    "₹": _RUPEE,
    "$": _DOLLAR,
    "usd": _DOLLAR,
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
}
_UNITS = {
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "kmph": ("kilometer per hour", "kilometers per hour"),
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "kg": ("kilogram", "kilograms"),
    "mg": ("milligram", "milligrams"),
    "ml": ("milliliter", "milliliters"),
    "gb": ("gigabyte", "gigabytes"),
    "mb": ("megabyte", "megabytes"),
    "kb": ("kilobyte", "kilobytes"),
    "ms": ("millisecond", "milliseconds"),
    "°c": ("degree Celsius", "degrees Celsius"),
    "°f": ("degree Fahrenheit", "degrees Fahrenheit"),
}


def _alternation(words: object) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))  # type: ignore[call-overload]


_CURRENCY_RE = re.compile(
    rf"(?:(?<![A-Za-z])(?P<word>Rs|INR|USD)\.?\s?|(?P<sym>[$€£₹]))\s?(?P<num>{_NUM})"
    rf"(?:\s?(?P<scale>{_alternation(_SCALE_WORDS)})\b)?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(rf"({_NUM})\s?%")
_UNIT_RE = re.compile(
    rf"(?P<num>{_NUM})\s?(?P<unit>{_alternation(_UNITS)})(?![A-Za-z])", re.IGNORECASE
)
_AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?([ap])\.?m\b", re.IGNORECASE)
_CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<![\d.,])(1[1-9]\d\d|20\d\d)(?!\d|[.,]\d)")
_NUMBER_RE = re.compile(rf"(?<![\d.])({_NUM})")
_NEGATIVE_RE = re.compile(r"(?<![\w.)])-(?=\d)")

_ABBREVIATIONS = (
    (re.compile(r"\bDr\."), "Doctor"),
    (re.compile(r"\bMr\."), "Mister"),
    (re.compile(r"\bMrs\."), "Missus"),
    (re.compile(r"\bMs\."), "Miss"),
    (re.compile(r"\bProf\."), "Professor"),
    (re.compile(r"\bJr\."), "Junior"),
    (re.compile(r"\bSr\."), "Senior"),
    (re.compile(r"\bvs\.?", re.IGNORECASE), "versus"),
    (re.compile(r"\bapprox\.", re.IGNORECASE), "approximately"),
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example,"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is,"),
    (re.compile(r"\betc\.(?=\s+[A-Z]|\s*$)"), "et cetera."),
    (re.compile(r"\betc\."), "et cetera"),
)
_SYMBOL_WORDS = {
    "&": " and ",
    "+": " plus ",
    "=": " equals ",
    "@": " at ",
    "×": " times ",
    "%": " percent ",
    "°": " degrees ",
    "/": " ",
    "_": " ",
}
_QUOTE_MAP = str.maketrans(
    {"’": "'", "‘": "'", "“": '"', "”": '"', "–": ", ", "—": ", ", "…": "..."}
)
_KEPT_PUNCTUATION = set(".,!?;:'\"()-")

_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_IMAGE_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_LINE_MARKERS_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*|>[ \t]?|[-*+•][ \t]+|\d+[.)][ \t]+)", re.MULTILINE
)
_RULE_RE = re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$", re.MULTILINE)
_LINE_BREAK_RE = re.compile(r"[ \t]*\n+[ \t]*")


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {int_to_words(rest)}" if rest else "")
    for size, name in _SCALES:
        if n >= size:
            head, rest = divmod(n, size)
            return f"{int_to_words(head)} {name}" + (f" {int_to_words(rest)}" if rest else "")
    raise AssertionError("unreachable")


def _digits_to_words(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _number_to_words(text: str) -> str:
    text = text.replace(",", "")
    whole, _, frac = text.partition(".")
    # Long or zero-led digit strings are identifiers (phone numbers, codes), read digit by digit.
    if len(whole) > 15 or (len(whole) > 1 and whole[0] == "0") or (len(whole) >= 10 and not frac):
        words = _digits_to_words(whole)
    else:
        words = int_to_words(int(whole))
    return f"{words} point {_digits_to_words(frac)}" if frac else words


def _ordinal_words(n: int) -> str:
    *head, last = int_to_words(n).split()
    if last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return " ".join([*head, last])


def _year_words(year: int) -> str:
    if year % 1000 == 0:
        return int_to_words(year)
    if 2000 < year < 2010:
        return f"two thousand {_ONES[year - 2000]}"
    first, second = divmod(year, 100)
    if second == 0:
        return f"{int_to_words(first)} hundred"
    if second < 10:
        return f"{int_to_words(first)} oh {_ONES[second]}"
    return f"{int_to_words(first)} {int_to_words(second)}"


def _currency(m: re.Match[str]) -> str:
    key = (m.group("word") or m.group("sym")).lower()
    singular, plural, minor_one, minor_many = _CURRENCIES[key]
    num = m.group("num").replace(",", "")
    scale = _SCALE_WORDS.get((m.group("scale") or "").lower())
    if scale:
        return f"{_number_to_words(num)} {scale} {plural}"
    whole, _, frac = num.partition(".")
    words = f"{_number_to_words(whole)} {singular if int(whole) == 1 else plural}"
    minor = int((frac + "0")[:2]) if frac else 0
    if minor:
        words += f" {int_to_words(minor)} {minor_one if minor == 1 else minor_many}"
    return words


def _unit(m: re.Match[str]) -> str:
    one, many = _UNITS[m.group("unit").lower()]
    num = m.group("num")
    return f"{num} {one if float(num.replace(',', '')) == 1 else many}"


def _minutes(minute: int) -> str:
    return ("oh " if minute < 10 else "") + int_to_words(minute)


def _clock(m: re.Match[str]) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if minute == 0:
        return f"{int_to_words(hour)} o'clock"
    return f"{int_to_words(hour)} {_minutes(minute)}"


def _ampm(m: re.Match[str]) -> str:
    words = int_to_words(int(m.group(1)))
    if m.group(2) and int(m.group(2)):
        words += " " + _minutes(int(m.group(2)))
    return f"{words} {m.group(3).lower()} m"


def _line_break(m: re.Match[str]) -> str:
    return " " if m.start() > 0 and m.string[m.start() - 1] in ".!?;:," else ", "


def _strip_markdown(text: str) -> str:
    text = _FENCE_RE.sub(" ", text)
    text = _IMAGE_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _RULE_RE.sub("", text)
    text = _LINE_MARKERS_RE.sub("", text)
    text = text.replace("`", "").replace("*", "").replace("~", "").replace("|", " ")
    text = re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)
    # A line break without punctuation becomes a comma so list items do not run together.
    return _LINE_BREAK_RE.sub(_line_break, text)


def _clean_symbols(text: str) -> str:
    out = []
    for ch in text.translate(_QUOTE_MAP):
        if ch in _SYMBOL_WORDS:
            out.append(_SYMBOL_WORDS[ch])
        elif ch.isspace() or ch in _KEPT_PUNCTUATION or unicodedata.category(ch)[0] in "LMN":
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def normalize_for_speech(text: str) -> str:
    # Directions go first: the markdown pass removes the asterisks that mark them.
    text = _strip_markdown(strip_directions(text))
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = _CURRENCY_RE.sub(_currency, text)
    text = _PERCENT_RE.sub(lambda m: f"{_number_to_words(m.group(1))} percent", text)
    text = _UNIT_RE.sub(_unit, text)
    text = _AMPM_RE.sub(_ampm, text)
    text = _CLOCK_RE.sub(_clock, text)
    text = _ORDINAL_RE.sub(lambda m: _ordinal_words(int(m.group(1))), text)
    text = _YEAR_RE.sub(lambda m: _year_words(int(m.group(1))), text)
    text = _NEGATIVE_RE.sub("minus ", text)
    text = _NUMBER_RE.sub(lambda m: _number_to_words(m.group(1)), text)
    text = _clean_symbols(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    return text.lstrip(" ,").rstrip()
