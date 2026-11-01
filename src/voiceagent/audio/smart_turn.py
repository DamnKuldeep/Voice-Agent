"""Smart Turn v3 end-of-turn detector (ONNX, CPU)."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

REPO = "pipecat-ai/smart-turn-v3"
FILES = {"cpu": "smart-turn-v3.2-cpu.onnx", "cuda": "smart-turn-v3.2-gpu.onnx"}
WINDOW_S = 8
RATE = 16000


def _model_path(filename: str) -> str:
    try:
        return hf_hub_download(REPO, filename, local_files_only=True)
    except OSError:
        return hf_hub_download(REPO, filename)


class SmartTurnDetector:
    def __init__(self, device: str = "cpu") -> None:
        kind = "cuda" if device.startswith("cuda") else "cpu"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if kind == "cuda" else [
            "CPUExecutionProvider"
        ]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            _model_path(FILES[kind]), sess_options=options, providers=providers
        )
        self._features = WhisperFeatureExtractor(chunk_length=WINDOW_S)

    def end_of_turn_probability(self, audio: np.ndarray) -> float:
        # The model was trained with the turn's end at the right edge of an 8 s window, so
        # shorter turns are zero-padded at the front.
        clip = np.asarray(audio, dtype=np.float32)[-WINDOW_S * RATE :]
        clip = np.pad(clip, (WINDOW_S * RATE - len(clip), 0))
        feats = self._features(
            clip,
            sampling_rate=RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_S * RATE,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        out = self._session.run(None, {"input_features": feats})[0]
        return float(np.asarray(out).reshape(-1)[0])
