"""Every control in the settings drawer, in display order, with plain-language help.

Reload is marked where the value is baked into a loaded model or client at construction: the ASR
chunk size and flush, everything the OpenRouter client reads (model, routing, providers,
temperature, timeout) and the Kokoro voice and speed. Everything else is read when a session starts,
so applying it only starts a new session.
"""

from __future__ import annotations

from .control_spec import Control

LLM_SUGGESTIONS = (
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-v4.1-flash",
    "openai/gpt-4.1-nano",
)
PROVIDER_SUGGESTIONS = ("Groq", "Cerebras", "CoreWeave", "BaseTen", "DeepInfra", "Together")
VOICE_SUGGESTIONS = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_fable",
)

_LISTENING = (
    Control("vad.threshold", "Speech sensitivity", "listening",
            "How sure the voice detector must be that a sound is speech. Lower it if quiet speech is "
            "missed; raise it if fans, typing or music start turns.",
            "slider", minimum=0.2, maximum=0.9, step=0.05),
    Control("vad.min_speech_ms", "Speech needed to start a turn", "listening",
            "Sounds shorter than this (coughs, clicks) are ignored. Lower it if short answers "
            "like \"yes\" are missed.",
            "slider", minimum=50, maximum=640, step=10, unit="ms"),
    Control("vad.end_hangover_ms", "Pause before end of speech", "listening",
            "Speech must stay quiet this long before your words count as finished. Raise it if the "
            "last soft word (\"plastic\" heard as \"plas\") gets cut; lower it for snappier replies.",
            "slider", minimum=0, maximum=480, step=32, unit="ms"),
    Control("vad.pre_roll_ms", "Audio kept before you start", "listening",
            "Audio from just before speech was detected is added back so the first word is not "
            "clipped. Raise it if first words go missing.",
            "slider", minimum=0, maximum=800, step=50, unit="ms"),
    Control("turn.detector", "End-of-turn detector", "listening",
            "Smart turn listens to how you speak to tell a thinking pause from the end of a "
            "sentence. Silence just waits a fixed time. Switching to smart turn after starting "
            "with silence loads the turn model.",
            "select", choices=("smart_turn", "silence")),
    Control("turn.min_silence_ms", "Fastest reply after you stop", "listening",
            "The earliest the agent answers when the turn model is sure you are done.",
            "slider", minimum=50, maximum=800, step=25, unit="ms"),
    Control("turn.max_silence_ms", "Longest wait for you to continue", "listening",
            "How long the agent waits when you sound unfinished (\"and...\"). Raise it if it "
            "jumps in while you think; lower it if replies feel slow.",
            "slider", minimum=300, maximum=2500, step=50, unit="ms"),
    Control("turn.silence_only_ms", "Fixed wait (silence detector)", "listening",
            "Used only when the detector is \"silence\" or no turn model loaded.",
            "slider", minimum=200, maximum=2000, step=50, unit="ms"),
    Control("asr.chunk_ms", "Recogniser step size", "listening",
            "How much audio the speech recogniser takes per step. 160 ms shows words sooner, but on "
            "this laptop it ran slower than real time while the agent spoke (0.43x alone, 1.28x "
            "beside the voice, measured n=15), so text fell behind. 560 ms shows words in larger "
            "steps but kept up (0.23x alone, 0.42x beside the voice) and was more accurate on "
            "accented English (Svarah WER 18.8% vs 21.5%, measured). 80 ms is slower still (0.79x "
            "even with nothing else running) and less accurate (Indian-accent WER 11.4%).",
            "select", choices=(80, 160, 560), needs_reload=True, unit="ms"),
    Control("asr.tail_ms", "Audio after you stop", "listening",
            "Real microphone audio still sent to the recogniser after the turn ends, so a soft "
            "last word is heard. Each step adds a little delay to the reply.",
            "slider", minimum=0, maximum=480, step=32, unit="ms"),
)

_ANSWERING = (
    Control("llm.model", "Language model", "answering",
            "The OpenRouter model that writes replies. Small models answer faster; larger ones "
            "follow the conversation better. Any OpenRouter model id works.",
            "text", needs_reload=True, suggestions=LLM_SUGGESTIONS),
    Control("llm.provider_order", "Preferred providers", "answering",
            "OpenRouter provider names tried first, in order, separated by commas (for example "
            "Groq). Leave empty to let OpenRouter pick by the routing rule below.",
            "list", needs_reload=True, suggestions=PROVIDER_SUGGESTIONS),
    Control("llm.routing_sort", "Routing rule", "answering",
            "How OpenRouter picks a provider when none is preferred: lowest latency, highest "
            "throughput, lowest price, or OpenRouter's default.",
            "select", choices=("latency", "throughput", "price", "default"), needs_reload=True),
    Control("llm.history_turns", "Turns remembered", "answering",
            "How many earlier exchanges the model sees, so follow-up questions make sense. More "
            "turns cost a little more time per reply.",
            "slider", minimum=0, maximum=20, step=1),
    Control("llm.temperature", "Creativity", "answering",
            "Lower is more predictable, higher is more varied.",
            "slider", minimum=0.0, maximum=1.2, step=0.1, needs_reload=True),
    Control("length_policy.enabled", "Keep replies short", "answering",
            "Brevity rules and a token cap, so spoken answers stay short unless you ask for detail.",
            "toggle"),
    Control("speculation.enabled", "Start thinking early", "answering",
            "Starts the model on your words before you have clearly finished, which saves time; "
            "the early start is thrown away if you keep talking.",
            "toggle"),
    Control("llm.stream", "Stream the reply text", "answering",
            "Speak the first words while the rest is still being written. Turn off only to "
            "measure what streaming saves.",
            "toggle"),
)

_SPEAKING = (
    Control("tts.voice", "Voice", "speaking",
            "The Kokoro voice. a = American, b = British, f = female, m = male.",
            "text", needs_reload=True, suggestions=VOICE_SUGGESTIONS),
    Control("tts.speed", "Speaking speed", "speaking",
            "1.0 is the voice's natural pace.",
            "slider", minimum=0.7, maximum=1.4, step=0.05, needs_reload=True, unit="x"),
    Control("tts.stream", "Stream the voice", "speaking",
            "Play each sentence as soon as it is ready instead of after the whole reply.",
            "toggle"),
    Control("chunker.first_chunk_min_words", "First spoken chunk: at least", "speaking",
            "The first piece of the reply is spoken once it has this many words and a natural "
            "break. Fewer words start sooner but can sound choppy.",
            "slider", minimum=1, maximum=20, step=1, unit="words"),
    Control("chunker.first_chunk_max_words", "First spoken chunk: at most", "speaking",
            "Force the first piece out at this many words even without a natural break.",
            "slider", minimum=2, maximum=30, step=1, unit="words"),
    Control("tts.fillers", "Say \"hmm\" when slow", "speaking",
            "Plays a short filler phrase if the model has not answered in time. Turning it on "
            "for the first time renders the fillers.",
            "toggle"),
    Control("tts.filler_after_ms", "Filler after", "speaking",
            "How long to wait for the model before the filler plays.",
            "slider", minimum=300, maximum=3000, step=100, unit="ms"),
)

_INTERRUPTIONS = (
    Control("barge_in.enabled", "Let me interrupt", "interruptions",
            "Talk over the agent and it stops to listen. Turn off to make it always finish.",
            "toggle"),
    Control("barge_in.min_speech_ms", "Speech needed to interrupt", "interruptions",
            "How long you must talk over the agent before it lowers its voice and checks. Lower "
            "is quicker; higher ignores coughs and short noises.",
            "slider", minimum=100, maximum=800, step=10, unit="ms"),
    Control("barge_in.vad_threshold", "Interruption sensitivity", "interruptions",
            "Stricter than normal listening so the agent's own voice from the speakers does not "
            "interrupt it. Raise it if it stops by itself; lower it if it ignores you.",
            "slider", minimum=0.3, maximum=0.95, step=0.05),
    Control("barge_in.duck_gain", "Volume while checking", "interruptions",
            "While a possible interruption is checked the agent keeps talking at this volume "
            "(0 = silent, 1 = no change).",
            "slider", minimum=0.0, maximum=1.0, step=0.05),
    Control("barge_in.ignore_backchannels", "Ignore \"uh-huh\" and \"yeah\"", "interruptions",
            "Short acknowledgements do not stop the agent.",
            "toggle"),
    Control("barge_in.backchannel_max_ms", "Longest acknowledgement", "interruptions",
            "Speech shorter than this that sounds like \"yeah\" counts as an acknowledgement; longer "
            "speech stops the agent even before its words are recognised. Measured: 700 ms stopped "
            "real interruptions in about 0.68 s and ignored every \"yeah\"/\"okay\"; 500 ms stopped in "
            "0.48 s but also stopped for 2 of 3 acknowledgements.",
            "slider", minimum=300, maximum=1500, step=50, unit="ms"),
    Control("barge_in.echo_guard", "Ignore my own voice", "interruptions",
            "For laptop speakers only: dismisses \"interruptions\" whose words match what the agent "
            "just said (its own voice coming back). Off by default: with headphones there is no echo, "
            "and it wrongly ignored real interruptions that repeated the agent's words.",
            "toggle"),
)

_ADVANCED = (
    Control("vad.gate_asr", "Send only speech to the recogniser", "advanced",
            "On: after 300 ms of pause the recogniser stops receiving audio until you speak again, "
            "which saves a little GPU time but lost soft last words the voice detector missed "
            "(\"who are [you]\"). Off (default): the whole turn is transcribed.",
            "toggle"),
    Control("asr.flush_chunks", "Final flush steps", "advanced",
            "Silent steps pushed at the end of a turn to finish the last word. Each costs about "
            "80 ms.",
            "slider", minimum=0, maximum=4, step=1, needs_reload=True),
    Control("llm.ttft_timeout_s", "Model timeout", "advanced",
            "No first word by then: try the next fallback model.",
            "slider", minimum=0.5, maximum=8.0, step=0.5, needs_reload=True, unit="s"),
    Control("speculation.start_silence_ms", "Early start after", "advanced",
            "Silence needed before the early start is tried.",
            "slider", minimum=50, maximum=800, step=25, unit="ms"),
    Control("speculation.stable_chunks", "Early start stability", "advanced",
            "Recogniser steps the transcript must stay unchanged before the early start.",
            "slider", minimum=1, maximum=6, step=1),
    Control("chunker.chunk_max_words", "Later chunks: at most", "advanced",
            "Later pieces of the reply break at sentence ends, bounded by this.",
            "slider", minimum=5, maximum=60, step=1, unit="words"),
    Control("devices.require_full_gpu_residency", "Require everything on the GPU", "advanced",
            "Refuse to start if any speech model weight sits on the CPU.",
            "toggle", needs_reload=True),
    Control("devices.cpu_threads", "CPU threads for torch", "advanced",
            "All cores made the recogniser 30% slower on this laptop (measured), so the default is 2.",
            "slider", minimum=1, maximum=16, step=1, needs_reload=True),
)

CONTROLS: tuple[Control, ...] = _LISTENING + _ANSWERING + _SPEAKING + _INTERRUPTIONS + _ADVANCED
