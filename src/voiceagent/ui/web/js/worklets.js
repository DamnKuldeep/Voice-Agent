// Audio worklets for the WebSocket transport: microphone capture and agent playback.

const MIC_RATE = 16000;
const MIC_FRAME = 320; // 20 ms at 16 kHz per message
// After running dry, wait for this much audio before playing again. It grows after every underrun
// (the shared link's tunnel delivered frames with 200-250 ms gaps) and shrinks back while the
// network is steady; an interruption still empties the buffer at once.
const PREBUFFER_MIN_S = 0.06;
const PREBUFFER_MAX_S = 0.3;
const PREBUFFER_STEP_S = 0.04;
const STEADY_S = 5; // this long without an underrun lowers the target one step

// Microphone in: average the samples of each output period (a simple anti-alias filter that is
// enough for speech), convert to int16 and post 20 ms frames to the page.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / MIC_RATE;
    this.pos = 0;
    this.sum = 0;
    this.count = 0;
    this.out = new Int16Array(MIC_FRAME);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.sum += ch[i];
      this.count += 1;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.sum / this.count));
        this.sum = 0;
        this.count = 0;
        this.out[this.n++] = v * 32767;
        if (this.n === MIC_FRAME) {
          this.port.postMessage(this.out.buffer, [this.out.buffer]);
          this.out = new Int16Array(MIC_FRAME);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

// Agent out: int16 frames at the server's rate, resampled to the context rate on arrival into a
// queue; {type: "clear"} empties it at once when the user interrupts.
class AgentPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.step = options.processorOptions.inputRate / sampleRate;
    this.queue = [];
    this.head = null;
    this.headPos = 0;
    this.queued = 0;
    this.primed = false;
    this.target = PREBUFFER_MIN_S;
    this.steady = 0;
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === "clear") {
        this.queue = [];
        this.head = null;
        this.queued = 0;
        this.primed = false;
        return;
      }
      const chunk = this._resample(new Int16Array(e.data));
      this.queue.push(chunk);
      this.queued += chunk.length;
    };
  }

  _resample(pcm) {
    const outLen = Math.floor(pcm.length / this.step);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const x = i * this.step;
      const j = Math.floor(x);
      const a = pcm[j] / 32768;
      const b = (j + 1 < pcm.length ? pcm[j + 1] : pcm[j]) / 32768;
      out[i] = a + (b - a) * (x - j);
    }
    return out;
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!this.primed) {
      if (this.queued < this.target * sampleRate) {
        out.fill(0);
        return true;
      }
      this.primed = true;
    }
    for (let i = 0; i < out.length; i++) {
      if (!this.head || this.headPos >= this.head.length) {
        this.head = this.queue.shift() || null;
        this.headPos = 0;
      }
      if (!this.head) {
        out.fill(0, i);
        this.primed = false;
        this.target = Math.min(PREBUFFER_MAX_S, this.target + PREBUFFER_STEP_S); // an underrun mid-reply
        this.steady = 0;
        return true;
      }
      out[i] = this.head[this.headPos++];
      this.queued -= 1;
    }
    this.steady += out.length / sampleRate;
    if (this.steady > STEADY_S) {
      this.target = Math.max(PREBUFFER_MIN_S, this.target - PREBUFFER_STEP_S);
      this.steady = 0;
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
registerProcessor("agent-player", AgentPlayer);
