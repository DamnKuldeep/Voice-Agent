# Design decisions

Short records of the choices that shaped the system: what was decided, why, and the evidence. Numbers are measured on the development laptop unless marked otherwise; the full measurements are in [benchmarks.md](benchmarks.md). IDs are referenced from code comments.

## Hardware and models

| ID | Decision | Why and evidence |
|---|---|---|
| D-001 | ASR and TTS run locally on the 4 GB GPU; the LLM is remote. | Only the two speech models need the GPU then; both fit with room to spare (peak 2.30 GiB of 4 GiB with a worst-case turn). A local LLM would not fit beside them. |
| D-002 | ASR: NVIDIA Nemotron Speech Streaming EN 0.6B, fp16, cache-aware streaming. | The only option here that streams natively with published accuracy per chunk size; loads through Hugging Face Transformers, no NeMo needed. Chunk size: see D-021. |
| D-003 | TTS: Kokoro-82M, one short text chunk at a time. | 8-word chunk to first audio p50 153 ms on this GPU (n=30); short chunks keep memory bounded. |
| D-004, D-017 | LLM: Llama 3.3 70B through OpenRouter, providers Groq then CoreWeave; fallbacks Llama 3.1 8B, GPT-4.1 Nano. | First token p50 0.20 s, p90 0.24 s from this network with realistic history (n=30), as fast as the 8B (0.18 / 0.23) but without its stage directions and guessing. About $0.31 per 1,000 turns. |
| D-005 | Turn-taking: Silero VAD + Smart Turn v3.2 + speculative LLM start. | VAD alone cannot tell a pause from a finished sentence; Smart Turn runs on the CPU in 61 ms p50. |
| D-014 | Python 3.11+ (developed on 3.12). | `typing.Self`; uv-managed interpreter. |

## Making a 4 GB GPU keep up

| ID | Decision | Why and evidence |
|---|---|---|
| D-013 | One long-lived ASR decode thread. | PyTorch keeps cuDNN plans per thread; a thread per turn re-planned every time and cost about 1.7 s per turn (first chunks 1,227 and 524 ms, then 80-90 ms once the thread persisted). |
| D-015 | `devices.cpu_threads: 2`. | Torch's default (all 8 cores) oversubscribed the CPU next to the audio and decode threads: ASR real-time factor 0.52 vs 0.75-0.80 (interleaved, 4 repeats). |
| D-021 | ASR chunk 560 ms (not 160 ms). | At 160 ms the recogniser fell behind real time whenever TTS ran (RTF 1.28; in live use its queue grew to 53 s). At 560 ms: RTF 0.42 under the same load, and lower WER on accented speech (Svarah 18.8% vs 21.5%). The end-of-turn flush keeps the final transcript fast regardless. |
| - | GPU clocks locked (`nvidia-smi -lgc 1500,1500`). | The ASR's many small kernels never woke the laptop GPU above 250-700 MHz; locking cut RTF at 160 ms from 1.14 to 0.77. A software keep-alive load raised clocks but time-sliced with the ASR and made it worse (RTF 3.8). |

## Hearing the user to the end

| ID | Decision | Why and evidence |
|---|---|---|
| D-022 | Earliest commit after 300 ms of silence (was 150 ms). | Premature commits 27% -> 20% of clips and voice-to-voice p50 1,154 -> 962 ms: fewer false commits mean less redone work. |
| - | Re-decode the whole turn when the user resumes after a commit. | Restarting the recogniser on the new words alone lost words at the pause ("faculty and distinguished" -> "faculty distinguished"). |
| D-023 | Feed the recogniser through the whole endpointing wait (`vad.gate_asr: false`), flush two silent chunks, count the post-commit tail against the pause. | Soft last words the VAD missed while the turn model waited were lost ("Who are [you]"). Recognising 150 clips the way the pipeline feeds them: right last word 126/150 with one flush chunk, 130/150 with two, 131/150 ideal. Counting the tail against the pause brought the cost down from +166 ms to about +50 ms. |
| D-016 | Public speech for evaluation (Common Voice Indian-accent, Svarah, LibriSpeech). | Reproducible, licensed, accented; never reported as the author's own WER. |

## Interruptions

| ID | Decision | Why and evidence |
|---|---|---|
| D-020 | Duck, then confirm. | Waiting to be sure before reacting left the agent talking over the user for 0.5-3 s. Now the volume ramps down after a short burst of speech and the words decide: a real word stops it, "yeah"/noise restores the volume. |
| D-024 | Duck after 150 ms of speech; keep the 700 ms backchannel window. | Duck p50 135 ms vs 219 ms with no extra false stops (15 trials each). A 500 ms window stopped 200 ms sooner but also stopped for 2 of 3 backchannels. |
| D-023 | Echo guard off by default. | With headphones there is no echo, and the guard dismissed real interruptions that repeated the agent's words. Switch on for laptop speakers. |
| - | Words spoken during the backchannel check reach the new turn; TTS synthesis pauses while a candidate is open. | Both fixed lost words and GPU contention seen in live testing. |

## The language model ("brain")

| ID | Decision | Why and evidence |
|---|---|---|
| D-023 | Hedged fallback: after 0.8 s without a first token, the next model starts alongside; first to speak wins. | One network moment timed out all three models in turn (7.5 s). The second request is made only on slow turns. |
| - | Length policy in code as well as in the prompt; never speak a half-cut sentence. | Brief answers by default (90 tokens), detail when asked (220); trimmed at the last sentence end. |
| - | Strip stage directions ("(pauses)", "*smiles*") before speech. | The 8B model spoke them aloud. |

## Interface and access

| ID | Decision | Why and evidence |
|---|---|---|
| D-006, D-019 | Own browser page (FastAPI + FastRTC offer endpoint + our JavaScript), not a Gradio layout. | Clear start/stop, live transcript, an audio-reactive orb, explained settings. FastRTC is confined to one adapter module. |
| D-012 | Gradio pinned below 6. | FastRTC 0.0.34 requires Gradio 5; with Gradio 6 the resolver silently picks an empty FastRTC 0.0.1. |
| D-025, D-026 | `--share`: Cloudflare quick tunnel; through it, audio and events travel over WebSockets. | WebRTC audio cannot pass an HTTP tunnel and the free TURN credential service was down (expired certificate); the tunnel also buffered Server-Sent Events completely. Through the public link: agent audio back 408 ms after the question ended, largest playback gap 86 ms with the adaptive jitter buffer. |
| D-007 | Every technique is a config switch with a fallback. | So each can be measured on and off (see the settings drawer and `scripts/run_eval.py --compare`). |
| D-009 | Secrets only in `.env`. | Never logged, printed or committed. |
| D-011 | Package installs on a slow network: torch wheel by direct download, a PyPI mirror, the spaCy model from Hugging Face. | `pypi.org` timed out from the development network. Versions pinned in `constraints.txt`. |
