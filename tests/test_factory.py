from typing import get_args

from pipeline_fakes import FakeAsrBackend, make_components
from voiceagent.config import Asr, load_settings
from voiceagent.pipeline import factory


def test_components_get_their_own_single_thread_pools_and_close_cleanly():
    components = make_components()
    assert components.fillers == [] and components.turn_detector is None
    assert components.asr_pool.submit(lambda: 1).result() == 1
    components.close()


def test_the_configured_backend_is_built_from_the_registry(monkeypatch):
    monkeypatch.setitem(factory.ASR_BACKENDS, "nemotron", ("pipeline_fakes", "FakeAsrBackend"))
    settings = load_settings()
    messages: list[str] = []
    asr = factory._build_asr(settings, messages.append)
    assert isinstance(asr, FakeAsrBackend) and asr.device == settings.devices.asr
    assert messages == ["Loading speech recognition (nemotron)"]


def test_every_configurable_backend_is_registered():
    choices = get_args(Asr.model_fields["backend"].annotation)
    assert set(choices) == set(factory.ASR_BACKENDS)
