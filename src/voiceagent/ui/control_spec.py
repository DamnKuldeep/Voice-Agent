"""One settings-drawer control: what it edits, how it is shown and how a page value is cast back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["toggle", "select", "slider", "number", "text", "list"]
GROUPS: tuple[tuple[str, str], ...] = (
    ("listening", "Listening"),
    ("answering", "Answering"),
    ("speaking", "Speaking"),
    ("interruptions", "Interruptions"),
    ("advanced", "Advanced"),
)
_TRUE_WORDS = frozenset({"true", "1", "on", "yes"})


@dataclass(frozen=True)
class Control:
    key: str
    label: str
    group: str
    help: str
    kind: Kind
    choices: tuple[Any, ...] = ()
    needs_reload: bool = False
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    unit: str = ""
    suggestions: tuple[str, ...] = ()

    def coerce(self, raw: Any, like: Any) -> Any:
        """Cast a value from the page (JSON, or a form string) to the type of the current setting."""
        if self.kind == "toggle":
            return raw.strip().lower() in _TRUE_WORDS if isinstance(raw, str) else bool(raw)
        if self.kind == "list":
            items = raw.split(",") if isinstance(raw, str) else list(raw)
            return [str(item).strip() for item in items if str(item).strip()]
        if self.kind == "text":
            text = str(raw).strip()
            if not text:
                raise ValueError(f"{self.label}: cannot be empty")
            return text
        if self.kind == "select":
            value = type(like)(raw)
            if value not in self.choices:
                raise ValueError(f"{self.label}: {raw!r} is not one of {list(self.choices)}")
            return value
        number = float(raw)
        if (self.minimum is not None and number < self.minimum) or (
                self.maximum is not None and number > self.maximum):
            raise ValueError(f"{self.label}: {raw} is outside {self.minimum} to {self.maximum}")
        return round(number) if isinstance(like, int) and not isinstance(like, bool) else number

    def describe(self, value: Any, default: Any) -> dict[str, Any]:
        """JSON-ready definition for the page, with the current and the config-file value."""
        return {
            "key": self.key, "label": self.label, "group": self.group, "help": self.help,
            "kind": self.kind, "choices": list(self.choices), "needs_reload": self.needs_reload,
            "min": self.minimum, "max": self.maximum, "step": self.step, "unit": self.unit,
            "suggestions": list(self.suggestions), "value": value, "default": default,
        }
