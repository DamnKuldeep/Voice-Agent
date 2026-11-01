"""Silence bookkeeping for a turn: when to ask the turn model and when the commit policy says go."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor

import numpy as np

from ..config import Turn as TurnConfig
from ..core.interfaces import TurnDetector
from .turn_manager import should_commit
from .turn_state import Turn

log = logging.getLogger(__name__)


class Endpointer:
    def __init__(self, cfg: TurnConfig, model: TurnDetector | None, pool: Executor, frame_ms: int) -> None:
        self._cfg, self._model, self._pool, self._frame_ms = cfg, model, pool, frame_ms

    def tick(self, turn: Turn) -> bool:
        """Count one more silent frame. True when the turn should be committed now."""
        turn.silence_ms += self._frame_ms
        cfg = self._cfg
        if (self._model is not None and cfg.detector == "smart_turn" and turn.eot_task is None
                and turn.silence_ms >= cfg.min_silence_ms):
            audio = np.concatenate(turn.frames)
            turn.eot_task = asyncio.get_running_loop().run_in_executor(
                self._pool, self._model.end_of_turn_probability, audio)
        if turn.eot_task is not None and turn.eot_task.done() and turn.eot_prob is None:
            try:
                turn.eot_prob = turn.eot_task.result()
            except Exception as exc:  # noqa: BLE001  unknown probability waits the full max silence
                log.warning("turn model failed: %s", exc)
        return should_commit(turn.silence_ms, cfg, eot_prob=turn.eot_prob, transcript=turn.partial)

    @staticmethod
    def cancel(turn: Turn) -> None:
        if turn.eot_task is not None:
            turn.eot_task.cancel()
        turn.eot_task, turn.eot_prob = None, None
