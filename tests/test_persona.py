from pathlib import Path

from voiceagent.llm.persona import build_messages, load_persona


def test_load_persona_relative_to_repo_root() -> None:
    assert "voice assistant" in load_persona("prompts/persona.md")


def test_load_persona_absolute(tmp_path: Path) -> None:
    file = tmp_path / "p.md"
    file.write_text("  Be nice.\n", encoding="utf-8")
    assert load_persona(file) == "Be nice."


def test_build_messages_with_length_policy() -> None:
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    msgs = build_messages("PERSONA", history, "what now", "brief", True)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"].startswith("PERSONA\n\n") and len(msgs[0]["content"]) > 10
    assert msgs[-1] == {"role": "user", "content": "what now"}


def test_build_messages_without_length_policy_has_plain_persona() -> None:
    msgs = build_messages("PERSONA", [], "x", "detail", False)
    assert msgs[0]["content"] == "PERSONA"
