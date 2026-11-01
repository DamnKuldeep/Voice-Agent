// Page wiring: server events in, button presses out. All conversation logic lives on the server.

import { Levels, rmsToLevel } from "./levels.js";
import { Orb } from "./orb.js";
import { PerfPanel } from "./perf.js";
import { RtcCall } from "./rtc.js";
import { WsCall } from "./ws.js";
import { SettingsDrawer } from "./settings.js";
import { Transcript } from "./transcript.js";

const $ = (id) => document.getElementById(id);
const INTERRUPTED_SHOW_MS = 1600;
const VRAM_POLL_MS = 5000;
const EVENTS_RETRY_MS = 1500;

function isLocalPage() {
  return ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
}

const TEXT = {
  loading: ["Loading models...", ""],
  error: ["Models did not load", ""],
  off: ["Press Start and talk", "Your words show up on the right as you speak."],
  connecting: ["Connecting...", "Allow the microphone if the browser asks."],
  idle: ["Listening...", "Go ahead, I'm listening."],
  listening: ["Listening...", ""],
  thinking: ["Thinking...", ""],
  speaking: ["Speaking", "Talk any time to interrupt."],
  ducked: ["Hearing you...", "Checking whether you are interrupting."],
  interrupted: ["Interrupted", "Go ahead, I'm listening."],
};

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {
    return null;
  }
  return value;
}

class App {
  constructor() {
    this.phase = "loading";       // server model phase: loading | ready | error
    this.call = "off";            // off | connecting | live
    this.state = "idle";          // session state from the server
    this.ducked = false;
    this.interruptedUntil = 0;
    this.progress = "";
    this.levels = new Levels();
    this.orb = new Orb($("orb"), this.levels);
    this.transcript = new Transcript($("messages"), $("chat-scroll"), $("chat-empty"));
    this.perf = new PerfPanel($("perf"));
    this.settings = new SettingsDrawer($("drawer"), { onMessage: (text, err) => this.notice(text, err ? "error" : "info") });
    this.transport = "auto";     // ui.transport from the server: auto | webrtc | websocket
    this.handlers = {
      onLocalStream: (s) => this.levels.attachMic(s),
      onRemoteStream: (s) => this._remote(s),
      onServerError: (text) => this.notice(text, "error"),
      onDrop: () => this._dropped(),
    };
    this.rtc = new RtcCall(this.handlers);
    this._bind();
    this._theme(store("va-theme") || "dark");
    this._showPerf(store("va-perf") === "1");
    this._connectEvents();
    this.render();
  }

  _bind() {
    $("talk-btn").addEventListener("click", () => this.toggleCall());
    $("settings-btn").addEventListener("click", () => (this.settings.isOpen ? this.settings.close() : this.settings.open()));
    $("perf-btn").addEventListener("click", () => this._showPerf($("perf").hidden));
    $("theme-btn").addEventListener("click", () => this._theme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
    $("reset-btn").addEventListener("click", () => this._post("/api/reset").then((r) => r && this.notice(r.message, "info")));
    $("save-open-btn").addEventListener("click", () => {
      const box = $("save-box");
      box.hidden = !box.hidden;
      if (!box.hidden) $("save-text").focus();
    });
    $("save-box").addEventListener("submit", (e) => {
      e.preventDefault();
      this._save();
    });
    $("notice-action").addEventListener("click", () => this._retryLoad());
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.settings.isOpen) this.settings.close();
    });
    window.addEventListener("pagehide", () => this.rtc.beacon());
  }

  _connectEvents() {
    const handle = (data) => {
      let event;
      try {
        event = JSON.parse(data);
      } catch {
        return;
      }
      if (event.type !== "keepalive") this.onEvent(event);
    };
    const lost = () => {
      this.progress = "Lost the server; reconnecting...";
      this.render();
    };
    if (isLocalPage()) {
      const es = new EventSource("/api/events");
      es.onmessage = (e) => handle(e.data);
      es.onerror = () => {
        if (es.readyState !== EventSource.OPEN) lost();
      };
      return;
    }
    // Through the shared link: Cloudflare's tunnel buffers event streams, so use a WebSocket and
    // reconnect by hand (EventSource would have done that itself).
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${location.host}/ws/events`);
    ws.onmessage = (e) => handle(e.data);
    ws.onclose = () => {
      lost();
      setTimeout(() => this._connectEvents(), EVENTS_RETRY_MS);
    };
  }

  onEvent(e) {
    switch (e.type) {
      case "status":
        this.phase = e.phase;
        this.progress = e.phase === "error" ? e.message : "";
        if (e.phase !== "error" && !$("notice-action").hidden) this.hideNotice();
        if (e.phase === "ready" && this.call !== "live") this.state = "idle";
        if (e.config) {
          $("model-chip").textContent = e.config.llm_model;
          this.transport = e.config.transport || "auto";
        }
        this.perf.status(e);
        if (e.last_final) $("save-text").value = e.last_final;
        break;
      case "progress":
        this.progress = e.message;
        break;
      case "state":
        this.state = e.state;
        if (e.state !== "speaking") this.ducked = false;
        this.transcript.state(e.state);
        break;
      case "partial":
        this.transcript.partial(e.text, Boolean(e.barge));
        break;
      case "final":
        this.transcript.final(e.text);
        $("save-text").value = e.text;
        break;
      case "reply":
        this.transcript.reply(e.text, Boolean(e.final));
        break;
      case "barge":
        this._barge(e.phase);
        break;
      case "turn":
        this.transcript.turn(e);
        this.perf.turn(e);
        break;
      case "summary":
        this.perf.summary(e.summary);
        break;
      case "asr_lag":
        this.perf.lag(e.ms);
        break;
      case "level":
        this.levels.fallbackMic = rmsToLevel(e.rms);
        return;
      case "reset":
        this.transcript.reset();
        break;
      default:
        return;
    }
    this.render();
  }

  _barge(phase) {
    this.transcript.bargePhase(phase);
    if (phase === "duck") {
      this.ducked = true;
      this.orb.dip();
    } else if (phase === "resume") {
      this.ducked = false;
    } else if (phase === "stop") {
      this.ducked = false;
      if (this.rtc.flush) this.rtc.flush(); // drop agent audio already queued in the page
      this.orb.dip();
      this.interruptedUntil = performance.now() + INTERRUPTED_SHOW_MS;
      setTimeout(() => this.render(), INTERRUPTED_SHOW_MS + 20);
    }
  }

  async toggleCall() {
    if (this.call !== "off") {
      this.rtc.stop();
      this.levels.detach();
      $("agent-audio").srcObject = null;
      this.call = "off";
      this.state = "idle";
      this.render();
      return;
    }
    this.call = "connecting";
    this.hideNotice();
    this.render();
    try {
      await this.levels.unlock();
      this.rtc = this._newCall();
      await this.rtc.start();
      this.call = "live";
      this.state = "idle";
    } catch (err) {
      this.call = "off";
      this.levels.detach();
      this.notice(err.message, "error");
    }
    this.render();
  }

  // WebRTC on this machine (lowest latency); a WebSocket through the shared link, where WebRTC audio
  // would need a TURN relay to reach a phone on another network.
  _newCall() {
    const websocket = this.transport === "websocket" || (this.transport === "auto" && !isLocalPage());
    return websocket ? new WsCall(this.handlers, () => this.levels.ctx) : new RtcCall(this.handlers);
  }

  _remote(stream) {
    const audio = $("agent-audio");
    audio.srcObject = stream;
    audio.play().catch(() => this.notice("The browser blocked playback. Click anywhere on the page to hear the agent.", "info"));
    this.levels.attachAgent(stream);
  }

  _dropped() {
    if (this.call === "off") return;
    this.call = "off";
    this.levels.detach();
    this.notice("The connection dropped. Press Start to talk again.", "error");
    this.render();
  }

  mode() {
    if (this.phase === "loading") return "loading";
    if (this.phase === "error") return "error";
    if (this.call === "off") return "off";
    if (this.call === "connecting") return "connecting";
    if (this.ducked) return "ducked";
    if (performance.now() < this.interruptedUntil && this.state !== "speaking") return "interrupted";
    return this.state;
  }

  render() {
    const mode = this.mode();
    const orbMode = { connecting: "loading", idle: "ready", interrupted: "ducked" }[mode] || mode;
    this.orb.setMode(orbMode);
    const [title, sub] = TEXT[mode] || TEXT.idle;
    $("status").textContent = title;
    $("substatus").textContent = this.progress || sub;
    const btn = $("talk-btn");
    const live = this.call !== "off";
    btn.disabled = this.phase !== "ready" && !live;
    btn.classList.toggle("live", this.call === "live");
    btn.classList.toggle("busy", this.call === "connecting" || (this.phase === "loading" && !live));
    btn.setAttribute("aria-label", live ? "Stop talking" : "Start talking");
    $("talk-label").textContent = this.call === "connecting" ? "Connecting" : live ? "Stop" : this.phase === "loading" ? "Loading" : "Start";
    if (this.phase === "error") this.notice(this.progress || "Model loading failed.", "error", "Retry");
  }

  notice(text, kind = "error", action = "") {
    if (!text) return;
    $("notice").hidden = false;
    $("notice").className = `notice ${kind === "info" ? "info" : ""}`.trim();
    $("notice-text").textContent = text;
    $("notice-action").hidden = !action;
    $("notice-action").textContent = action;
  }

  hideNotice() {
    $("notice").hidden = true;
  }

  async _retryLoad() {
    this.hideNotice();
    await this._post("/api/load");
  }

  async _post(url, body = {}) {
    try {
      const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await res.json();
      if (!res.ok) {
        this.notice(data.detail || res.statusText, "error");
        return null;
      }
      return data;
    } catch (err) {
      this.notice(`The server did not answer (${err.message}).`, "error");
      return null;
    }
  }

  async _save() {
    const status = $("save-status");
    const result = await this._post("/api/save", { text: $("save-text").value });
    status.textContent = result ? `Saved as ${result.id}` : "Not saved.";
  }

  _theme(theme) {
    document.documentElement.dataset.theme = theme;
    store("va-theme", theme);
    this.orb.setLight(theme === "light");
    $("theme-btn").querySelector("use").setAttribute("href", theme === "dark" ? "#i-sun" : "#i-moon");
  }

  _showPerf(show) {
    $("perf").hidden = !show;
    $("perf-btn").setAttribute("aria-expanded", String(show));
    store("va-perf", show ? "1" : "0");
    clearInterval(this.vramTimer);
    if (show) {
      this._pollStatus();
      this.vramTimer = setInterval(() => this._pollStatus(), VRAM_POLL_MS);
    }
  }

  async _pollStatus() {
    try {
      const res = await fetch("/api/status");
      const status = await res.json();
      this.perf.setVram(status.vram);
    } catch {
      // the event stream reports a lost server; nothing to add here
    }
  }
}

new App();
