// Live loudness and spectrum of the microphone and of the agent's voice, from WebAudio analysers.
// Chrome only feeds remote WebRTC audio into WebAudio while the same stream also plays in an
// <audio> element, so the agent analyser is never connected to the speakers itself.

const BANDS = 24;
const FLOOR_DB = -58;
const RANGE_DB = 46;

function makeTap(ctx, stream) {
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0.55;
  source.connect(analyser);
  return {
    source, analyser,
    freq: new Uint8Array(analyser.frequencyBinCount),
    wave: new Float32Array(analyser.fftSize),
  };
}

function readTap(tap, out) {
  tap.analyser.getFloatTimeDomainData(tap.wave);
  let sum = 0;
  for (let i = 0; i < tap.wave.length; i++) sum += tap.wave[i] * tap.wave[i];
  const rms = Math.sqrt(sum / tap.wave.length);
  tap.analyser.getByteFrequencyData(tap.freq);
  // Speech lives below ~4 kHz: spread the lowest bins (about 94 Hz each at 48 kHz) over the bands.
  const usable = Math.min(tap.freq.length, BANDS * 2);
  for (let b = 0; b < BANDS; b++) {
    const i = Math.floor((b / BANDS) * usable);
    out.bands[b] = (tap.freq[i] + tap.freq[Math.min(i + 1, usable - 1)]) / 510;
  }
  out.level = rmsToLevel(rms);
  return out;
}

export function rmsToLevel(rms) {
  if (!(rms > 0)) return 0;
  return Math.max(0, Math.min(1, (20 * Math.log10(rms) - FLOOR_DB) / RANGE_DB));
}

export class Levels {
  constructor() {
    this.ctx = null;
    this.mic = null;
    this.agent = null;
    this.fallbackMic = 0;
    this._out = { level: 0, bands: new Float32Array(BANDS) };
  }

  // Must run inside the Start click: browsers only allow audio to start from a user gesture.
  async unlock() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    if (!this.ctx) this.ctx = new Ctx();
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  attachMic(stream) {
    if (this.ctx) this.mic = makeTap(this.ctx, stream);
  }

  attachAgent(stream) {
    if (!this.ctx) return;
    if (this.agent) this.agent.source.disconnect();
    this.agent = makeTap(this.ctx, stream);
  }

  detach() {
    for (const tap of [this.mic, this.agent]) if (tap) tap.source.disconnect();
    this.mic = this.agent = null;
    this.fallbackMic = 0;
  }

  read(which) {
    const tap = which === "mic" ? this.mic : this.agent;
    const out = this._out;
    if (tap) return readTap(tap, out);
    out.bands.fill(0);
    out.level = which === "mic" ? this.fallbackMic : 0;
    return out;
  }
}

export const BAND_COUNT = BANDS;
