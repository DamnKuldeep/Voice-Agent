// The agent over a WebSocket: works through the shared HTTPS link from any network, because the
// audio travels inside the same connection as the page (WebRTC media would need a TURN relay).
// Same interface as RtcCall, plus flush() to drop queued agent audio on an interruption.

import { MIC_CONSTRAINTS, micErrorMessage } from "./rtc.js";

const AGENT_RATE = 24000;
const OPEN_TIMEOUT_MS = 10000;

export class WsCall {
  // handlers: onLocalStream(stream), onRemoteStream(stream), onServerError(text), onDrop()
  // context(): the page's unlocked AudioContext
  constructor(handlers, context) {
    this.h = handlers;
    this.context = context;
    this.ws = null;
    this.stream = null;
    this.nodes = [];
    this.player = null;
  }

  get connected() {
    return this.ws !== null;
  }

  async start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This page needs a secure (https) address for the microphone.");
    }
    const ctx = this.context();
    if (!ctx || !ctx.audioWorklet) throw new Error("This browser cannot run the audio worklets this page needs.");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
    } catch (err) {
      throw new Error(micErrorMessage(err));
    }
    this.stream = stream;
    this.h.onLocalStream(stream);
    await ctx.audioWorklet.addModule("/static/js/worklets.js");

    const mic = ctx.createMediaStreamSource(stream);
    const capture = new AudioWorkletNode(ctx, "mic-capture");
    const mute = ctx.createGain();
    mute.gain.value = 0; // keeps the capture node pulled by the graph without playing the mic back
    mic.connect(capture).connect(mute).connect(ctx.destination);
    const player = new AudioWorkletNode(ctx, "agent-player", {
      numberOfInputs: 0, outputChannelCount: [1], processorOptions: { inputRate: AGENT_RATE },
    });
    const out = ctx.createMediaStreamDestination();
    player.connect(out);
    this.player = player;
    this.nodes = [mic, capture, mute, player, out];
    this.h.onRemoteStream(out.stream); // played by the page's <audio> element and drawn by the orb

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${location.host}/ws/audio`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    capture.port.onmessage = (e) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
    };
    ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        this._onText(e.data);
      } else if (this.player) {
        this.player.port.postMessage(e.data, [e.data]);
      }
    };
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("The server did not answer the audio connection.")), OPEN_TIMEOUT_MS);
      ws.onopen = () => {
        clearTimeout(timer);
        resolve();
      };
      ws.onerror = () => {
        clearTimeout(timer);
        reject(new Error("The audio connection to the server failed."));
      };
    }).catch((err) => {
      this._teardown();
      throw err;
    });
    ws.onclose = () => {
      if (this.ws === ws) {
        this._teardown();
        this.h.onDrop();
      }
    };
  }

  flush() {
    if (this.player) this.player.port.postMessage({ type: "clear" });
  }

  _onText(data) {
    let msg;
    try {
      msg = JSON.parse(data);
    } catch {
      return;
    }
    if (msg && msg.type === "error") this.h.onServerError(msg.message || "Server error");
  }

  stop() {
    this._teardown();
  }

  // Closing the socket ends the session; the browser does that by itself when the page goes away.
  beacon() {}

  _teardown() {
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onclose = null;
      ws.close();
    }
    for (const node of this.nodes) node.disconnect();
    this.nodes = [];
    this.player = null;
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    this.stream = null;
  }
}
