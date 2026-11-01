import pytest
from pydantic import ValidationError

from voiceagent.config import deep_merge, load_settings, parse_dotted


def test_default_config_loads():
    s = load_settings()
    assert s.asr.chunk_ms == 560  # D-021
    assert s.devices.require_full_gpu_residency is True


def test_overrides_merge_nested_keys():
    s = load_settings(overrides={"tts": {"stream": False}, "llm": {"routing_sort": "default"}})
    assert s.tts.stream is False
    assert s.tts.voice == "af_heart"
    assert s.llm.routing_sort == "default"


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        load_settings(overrides={"tts": {"streaming": True}})


def test_dotted_overrides_parse_types():
    out = parse_dotted(["tts.stream=false", "asr.chunk_ms=560", "llm.model=a/b"])
    assert out == {"tts": {"stream": False}, "asr": {"chunk_ms": 560}, "llm": {"model": "a/b"}}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}} and base == {"a": {"b": 1, "c": 2}}
