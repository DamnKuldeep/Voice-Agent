"""Everything after the user's turn is committed: final transcript, speculation, the reply and its playback."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from typing import cast

import numpy as np

from ..config import Settings
from ..core.events import Flag, Mark
from ..core.interfaces import ChatModel, SpeechSynthesizer
from ..llm.length_policy import choose_mode, max_tokens_for
from ..llm.persona import build_messages
from .asr_stage import AsrStage
from .history import History
from .playout import Playout
from .responder import UNCAPPED_TOKENS, Responder
from .speculation import Speculator
from .turn_state import State, Turn

log = logging.getLogger(__name__)

ASR_FAILURE_REPLY = "Sorry, I didn't catch that. Could you say it again?"
TTS_FAILURE_REPLY = "Sorry, my voice stopped working for a moment."
DRAIN_POLL_S = 0.02
TAIL_GRACE_S = 0.3  # the mic stopped delivering frames: finish with what the recogniser has


def join_text(carry: str, text: str) -> str:
    return " ".join(part for part in (carry.strip(), text.strip()) if part)


class ReplyFlow:
    def __init__(
        self, settings: Settings, chat: ChatModel, tts: SpeechSynthesizer, tts_pool: Executor,
        playout: Playout, history: History, asr: AsrStage, persona: str,
        fillers: Sequence[np.ndarray], emit: Callable[[dict], None], set_state: Callable[[State], None],
    ) -> None:
        self._cfg, self._chat, self._tts, self._pool = settings, chat, tts, tts_pool
        self._playout, self._history, self._asr = playout, history, asr
        self._persona, self._fillers = persona, fillers
        self._emit, self._set_state = emit, set_state
        self.spec = Speculator(settings.speculation, settings.turn.hold_trailing_words)
        self.synth_gate = asyncio.Event()  # cleared by the barge-in controller while a candidate is open
        self.synth_gate.set()
        self._live: set[Responder] = set()

    def maybe_speculate(self, turn: Turn) -> None:
        if self.spec.should_start(turn.silence_ms, turn.partial):
            text = join_text(turn.carry, turn.partial)
            self.spec.start(self._make_responder(turn.trace.turn_id, text, hold=True), text)

    def cancel_speculation(self, turn: Turn) -> None:
        if self.spec.cancel():
            turn.trace.flag(Flag.SPECULATION_WASTED)

    async def run(self, turn: Turn) -> bool:
        """Answer a committed turn. False means nothing was said (noise, or the recogniser failed)."""
        trace = turn.trace
        await self._wait_tail(turn)
        try:
            final = await self._asr.finish()
        except Exception:  # noqa: BLE001  already logged by the ASR stage
            self.spec.cancel()
            await self.notice(ASR_FAILURE_REPLY, turn)
            return False
        trace.mark(Mark.ASR_FINAL)
        text = join_text(turn.carry, final)
        if not text:
            self.spec.cancel()
            return False
        turn.user_text = text
        self._emit({"type": "final", "text": text})
        responder = self._pick_responder(turn, text)
        self._history.add_user(text)
        self._playout.begin_reply()
        turn.responder = responder
        responder.adopt(trace, lambda reply: self._emit({"type": "reply", "text": reply, "final": False}))
        responder.release()
        await self._play(turn, responder)
        return True

    async def notice(self, text: str, turn: Turn) -> None:
        """Say a fixed sentence (an apology) without involving the LLM."""
        responder = Responder(self._cfg, self._chat, self._tts, self._playout, self._pool,
                              turn.trace.turn_id, synth_gate=self.synth_gate)
        turn.responder = responder
        responder.start_canned(text)
        self._track(responder)
        self._playout.begin_reply()
        self._set_state(State.SPEAKING)
        await responder.wait()
        await self._drain()

    def interrupt(self, turn: Turn) -> str:
        """Stop talking now. Returns the words the user heard; history keeps only those."""
        spoken = self._playout.spoken_text()
        self._playout.clear()
        turn.trace.mark(Mark.PLAYBACK_STOPPED)
        turn.trace.flag(Flag.INTERRUPTED)
        if turn.responder is not None:
            turn.responder.cancel()
            if turn.responder.text:
                self._history.add_assistant(turn.responder.text)
                self._history.mark_interrupted(spoken)
        if turn.task is not None:
            turn.task.cancel()
        turn.reply_text = spoken
        return spoken

    def abandon(self, turn: Turn) -> None:
        """The user went on talking after the commit: drop the reply and the question it answered."""
        self.spec.cancel()
        if turn.responder is not None:
            turn.responder.cancel()
        if turn.task is not None:
            turn.task.cancel()
        self._playout.clear()
        if turn.user_text:
            self._history.drop_last_user()

    def _pick_responder(self, turn: Turn, text: str) -> Responder:
        speculated = self.spec.running
        hit = self.spec.resolve(text)
        if hit is not None:
            turn.trace.flag(Flag.SPECULATION_USED)
            return cast(Responder, hit)
        if speculated:
            turn.trace.flag(Flag.SPECULATION_WASTED)
        return self._make_responder(turn.trace.turn_id, text, hold=False)

    def _make_responder(self, turn_id: int, text: str, *, hold: bool) -> Responder:
        policy = self._cfg.length_policy
        mode = choose_mode(text, policy)
        messages = build_messages(self._persona, self._history.messages(), text, mode, policy.enabled)
        responder = Responder(self._cfg, self._chat, self._tts, self._playout, self._pool, turn_id,
                              self._fillers, synth_gate=self.synth_gate)
        responder.start(messages, max_tokens_for(mode, policy) if policy.enabled else UNCAPPED_TOKENS,
                        hold=hold)
        return self._track(responder)

    def _track(self, responder: Responder) -> Responder:
        """Remember a started responder until it ends, so ``aclose`` can wait for it."""
        self._live.add(responder)
        responder.task.add_done_callback(lambda _task: self._live.discard(responder))
        return responder

    async def aclose(self) -> None:
        """Cancel every reply still running and wait until its tasks have ended."""
        self.spec.cancel()
        self.synth_gate.set()
        for responder in list(self._live):
            await responder.aclose()

    async def _wait_tail(self, turn: Turn) -> None:
        """Let asr.tail_ms of real post-commit audio reach the recogniser; speculation runs meanwhile."""
        timeout = self._cfg.asr.tail_ms / 1000 + TAIL_GRACE_S
        try:
            await asyncio.wait_for(turn.tail_done.wait(), timeout)
        except TimeoutError:
            log.info("mic went quiet during the ASR tail; finishing early")
        turn.trace.mark(Mark.ASR_TAIL_END)

    async def _play(self, turn: Turn, responder: Responder) -> None:
        watcher = asyncio.ensure_future(responder.first_audio.wait())
        try:
            await asyncio.wait({watcher, responder.task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            watcher.cancel()
            await asyncio.wait({watcher})
        if responder.first_audio.is_set():
            self._set_state(State.SPEAKING)
        await responder.wait()
        if responder.error is not None and not responder.first_audio.is_set():
            await self.notice(TTS_FAILURE_REPLY, turn)
        await self._drain()
        trace = turn.trace
        if Mark.AUDIO_OUT_START in trace.marks:
            trace.mark(Mark.AUDIO_OUT_END)
        turn.reply_text = responder.text
        self._history.add_assistant(responder.text)
        self._record_flags(trace, responder)
        self._emit({"type": "reply", "text": responder.text, "final": True})

    async def _drain(self) -> None:
        while not self._playout.idle:
            await asyncio.sleep(DRAIN_POLL_S)

    @staticmethod
    def _record_flags(trace, responder: Responder) -> None:
        trace.flag(Flag.REPLY_WORDS, len(responder.text.split()))
        if responder.model:
            trace.flag(Flag.LLM_MODEL, responder.model)
        usage = responder.usage or {}
        if "prompt_tokens" in usage:
            trace.flag(Flag.PROMPT_TOKENS, usage["prompt_tokens"])
        if "completion_tokens" in usage:
            trace.flag(Flag.COMPLETION_TOKENS, usage["completion_tokens"])
