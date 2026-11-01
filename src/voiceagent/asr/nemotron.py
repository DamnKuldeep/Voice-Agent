"""Nemotron Speech Streaming through Transformers: cache-aware chunked streaming, resident on the GPU.

Transformers drives streaming from a ``generate`` call that pulls mel chunks from a generator. All
``generate`` calls run on one long-lived worker thread (cuDNN keeps its convolution plans per thread, so a
new thread per turn would rebuild them every time). ``push`` turns audio into exact-size mel chunks, hands
them over through a queue, and waits until the worker has decoded them, so the partial it returns is current.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor

from ..config import Asr

# chunk_ms -> right attention context in encoder frames; latency is (context + 1) * 80 ms.
LOOKAHEAD = {80: 0, 160: 1, 560: 6, 1120: 13}
RATE = 16000
WARMUP_SECONDS = 1.0


class _TokenCollector:
    """Receives the token ids ``generate`` emits, as a streamer."""

    def __init__(self) -> None:
        self.ids: list[int] = []

    def put(self, value: torch.Tensor) -> None:
        self.ids.extend(value.flatten().tolist())

    def end(self) -> None:
        pass


class _Turn:
    """State of one streamed turn: the chunk queue the worker reads and what it has done so far."""

    def __init__(self) -> None:
        self.chunks: queue.Queue[torch.Tensor | None] = queue.Queue()
        self.collector = _TokenCollector()
        self.sent = 0               # chunks queued
        self.requested = 0          # chunks the worker has asked for; idle when requested == sent + 1
        self.ended = False
        self.error: Exception | None = None


class NemotronStreamingASR:
    def __init__(self, settings: Asr, device: str) -> None:
        self._lookahead = LOOKAHEAD[settings.chunk_ms]
        self._flush_chunks = settings.flush_chunks
        self._processor = AutoProcessor.from_pretrained(settings.model)
        self._processor.set_num_lookahead_tokens(self._lookahead)
        dtype = getattr(torch, settings.dtype)
        self._model = AutoModelForRNNT.from_pretrained(settings.model, dtype=dtype).to(device).eval()
        extractor = self._processor.feature_extractor
        self._hop, self._n_fft = extractor.hop_length, extractor.n_fft
        self._cv = threading.Condition()
        self._turns: queue.Queue[_Turn | None] = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True, name="asr-decode")
        self._worker.start()
        self._turn = _Turn()
        self._started = False
        self._audio = np.zeros(0, dtype=np.float32)
        self._mel_idx = 0

    def torch_modules(self) -> list[torch.nn.Module]:
        return [self._model]

    def warmup(self) -> None:
        self.reset()
        self.push(np.zeros(int(WARMUP_SECONDS * RATE), dtype=np.float32))
        self.finish()
        self.reset()

    def close(self) -> None:
        self.reset()
        self._turns.put(None)
        self._worker.join(timeout=10)

    def reset(self) -> None:
        self._end_turn()
        self._turn = _Turn()
        self._started = False
        self._audio = np.zeros(0, dtype=np.float32)
        self._mel_idx = 0
        self._text = ""

    def push(self, audio: np.ndarray) -> str:
        self._audio = np.concatenate([self._audio, audio.astype(np.float32, copy=False)])
        self._enqueue_ready_chunks()
        self._wait_idle()
        return self._decode()

    def finish(self) -> str:
        # Silence pushes the last real chunk through and supplies the lookahead the final words need.
        pad = np.zeros(self._flush_chunks * self._processor.num_samples_per_audio_chunk, dtype=np.float32)
        self._audio = np.concatenate([self._audio, pad])
        self._enqueue_ready_chunks()
        self._wait_idle()
        text = self._decode()
        self._end_turn()
        return text

    def _enqueue_ready_chunks(self) -> None:
        proc, turn = self._processor, self._turn
        while True:
            first = turn.sent == 0
            if first:
                start, end = 0, proc.num_samples_first_audio_chunk
            else:
                start = self._mel_idx * self._hop - self._n_fft // 2
                end = start + proc.num_samples_per_audio_chunk
            if len(self._audio) < end:
                return
            # With no lookahead the second window starts before sample 0; mirror the centred first frame.
            segment = (
                np.pad(self._audio[:end], (-start, 0), mode="reflect")
                if start < 0 else self._audio[start:end]
            )
            inputs = proc(
                segment, sampling_rate=RATE, is_streaming=True,
                is_first_audio_chunk=first, return_tensors="pt",
            )
            frames = (
                proc.num_mel_frames_first_audio_chunk if first else proc.num_mel_frames_per_audio_chunk
            )
            self._mel_idx += frames
            if not self._started:
                self._started = True
                self._turns.put(turn)
            turn.sent += 1
            # centre padding on the first chunk yields one frame too many
            turn.chunks.put(inputs.input_features[:, :frames, :])

    def _work(self) -> None:
        while (turn := self._turns.get()) is not None:
            try:
                with torch.inference_mode():
                    self._model.generate(
                        input_features=self._feature_stream(turn),
                        num_lookahead_tokens=self._lookahead,
                        streamer=turn.collector,
                    )
            except Exception as exc:  # noqa: BLE001  stored, then re-raised on the caller's thread
                turn.error = exc
            finally:
                with self._cv:
                    turn.ended = True
                    self._cv.notify_all()

    def _feature_stream(self, turn: _Turn):
        while True:
            with self._cv:
                turn.requested += 1
                self._cv.notify_all()
            chunk = turn.chunks.get()
            if chunk is None:
                return
            yield chunk

    def _wait_idle(self) -> None:
        turn = self._turn
        if not self._started:
            return
        with self._cv:
            self._cv.wait_for(lambda: turn.requested == turn.sent + 1 or turn.ended)
        if turn.error is not None:
            raise RuntimeError("ASR decode failed") from turn.error

    def _end_turn(self) -> None:
        turn = self._turn
        if self._started and not turn.ended:
            turn.chunks.put(None)
            with self._cv:
                self._cv.wait_for(lambda: turn.ended, timeout=10)
        self._started = False

    def _decode(self) -> str:
        if self._turn.collector.ids:
            self._text = self._processor.decode(self._turn.collector.ids, skip_special_tokens=True).strip()
        return self._text
