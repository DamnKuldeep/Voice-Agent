"""Names of the timestamps recorded for every conversational turn."""

from __future__ import annotations

from enum import Enum


class Mark(str, Enum):
    SPEECH_START = "speech_start"            # user began speaking (first VAD-positive frame)
    SPEECH_END = "speech_end"                # last speech frame before the final silence
    COMMIT = "commit"                        # turn manager decided the user is done
    ASR_TAIL_END = "asr_tail_end"            # asr.tail_ms of post-commit audio reached the recogniser
    ASR_FINAL = "asr_final"                  # final transcript available
    LLM_REQUEST = "llm_request"              # request sent (may be speculative, before COMMIT)
    LLM_FIRST_TOKEN = "llm_first_token"
    FIRST_CHUNK_READY = "first_chunk_ready"  # first speakable text chunk handed to the TTS
    TTS_FIRST_AUDIO = "tts_first_audio"      # first audio samples produced
    AUDIO_OUT_START = "audio_out_start"      # first sample handed to the transport
    AUDIO_OUT_END = "audio_out_end"
    BARGE_IN_ONSET = "barge_in_onset"        # user started talking over the agent
    BARGE_IN_DUCKED = "barge_in_ducked"      # playback volume lowered while the candidate is checked
    PLAYBACK_STOPPED = "playback_stopped"    # agent audio actually stopped


class Flag(str, Enum):
    SPECULATION_USED = "speculation_used"    # speculative LLM output was reused at commit
    SPECULATION_WASTED = "speculation_wasted"
    INTERRUPTED = "interrupted"
    LLM_MODEL = "llm_model"                  # model that actually answered (after any fallback)
    REPLY_WORDS = "reply_words"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    ASR_MAX_LAG_MS = "asr_max_lag_ms"        # largest recogniser backlog (audio ms) while the turn was open
