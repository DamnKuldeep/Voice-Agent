// The animated orb: layered, audio-reactive blobs drawn on a canvas, one look per conversation state.

import { BAND_COUNT } from "./levels.js";

const TAU = Math.PI * 2;
const POINTS = 120;

// colours (r, g, b) for the three layers, wobble amount, swirl speed, which audio drives it
const MODES = {
  off:       { c: [[86, 98, 124], [62, 72, 96], [120, 132, 160]], wobble: 0.02, swirl: 0.12, source: null, breathe: 1 },
  loading:   { c: [[86, 98, 124], [70, 120, 150], [120, 132, 160]], wobble: 0.03, swirl: 0.5, source: null, breathe: 1 },
  ready:     { c: [[40, 190, 184], [32, 120, 200], [120, 230, 214]], wobble: 0.025, swirl: 0.18, source: "mic", breathe: 1 },
  listening: { c: [[56, 214, 170], [30, 150, 210], [150, 255, 210]], wobble: 0.04, swirl: 0.3, source: "mic", breathe: 0 },
  thinking:  { c: [[150, 110, 240], [80, 120, 255], [230, 140, 255]], wobble: 0.07, swirl: 1.6, source: null, breathe: 0 },
  speaking:  { c: [[70, 140, 255], [40, 200, 230], [170, 120, 255]], wobble: 0.04, swirl: 0.45, source: "agent", breathe: 0 },
  ducked:    { c: [[242, 165, 65], [240, 106, 106], [255, 210, 120]], wobble: 0.05, swirl: 0.7, source: "mic", breathe: 0 },
  error:     { c: [[240, 106, 106], [170, 60, 80], [255, 150, 140]], wobble: 0.015, swirl: 0.1, source: null, breathe: 1 },
};

// Back to front: colour index, fade-to colour index, size, swirl direction, wave harmonics, phase.
const LAYERS = [
  { color: 1, fade: 2, alpha: 0.55, r: 1.04, speed: 1.0, harm: [3, 5], offset: 0 },
  { color: 2, fade: 0, alpha: 0.45, r: 0.98, speed: -1.3, harm: [2, 4], offset: 2.1 },
  { color: 0, fade: 1, alpha: 0.85, r: 0.92, speed: 0.7, harm: [4, 6], offset: 4.2 },
];

function approach(current, target, rate, dt) {
  return current + (target - current) * (1 - Math.exp(-rate * dt));
}

function rgba(c, a) {
  return `rgba(${c[0] | 0}, ${c[1] | 0}, ${c[2] | 0}, ${a})`;
}

export class Orb {
  constructor(canvas, levels) {
    this.canvas = canvas;
    this.g = canvas.getContext("2d");
    this.levels = levels;
    this.mode = "off";
    this.colors = MODES.off.c.map((c) => c.slice());
    this.wobble = MODES.off.wobble;
    this.swirl = MODES.off.swirl;
    this.phase = 0;
    this.energy = 0;
    this.bands = new Float32Array(BAND_COUNT);
    this.dipAt = -10;
    this.last = performance.now();
    this.light = false;
    this.reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    new ResizeObserver(() => this._resize()).observe(canvas);
    this._resize();
    this._frame = this._frame.bind(this);
    requestAnimationFrame(this._frame);
  }

  setMode(mode) {
    if (MODES[mode]) this.mode = mode;
  }

  setLight(light) {
    this.light = light;
  }

  // A quick shrink-and-recover, shown when the agent yields to an interruption.
  dip() {
    this.dipAt = performance.now() / 1000;
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(rect.width * dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * dpr));
  }

  _frame(nowMs) {
    const dt = Math.min(0.1, (nowMs - this.last) / 1000);
    this.last = nowMs;
    this._update(dt, nowMs / 1000);
    this._draw(nowMs / 1000);
    requestAnimationFrame(this._frame);
  }

  _update(dt, t) {
    const spec = MODES[this.mode];
    for (let i = 0; i < 3; i++) {
      for (let k = 0; k < 3; k++) this.colors[i][k] = approach(this.colors[i][k], spec.c[i][k], 5, dt);
    }
    this.wobble = approach(this.wobble, spec.wobble, 3, dt);
    this.swirl = approach(this.swirl, spec.swirl, 2, dt);
    this.phase += this.swirl * dt;
    const audio = spec.source ? this.levels.read(spec.source) : null;
    const target = audio ? audio.level : 0;
    // Fast attack, slower release: reads as "music-like" without flicker.
    this.energy = approach(this.energy, target, target > this.energy ? 18 : 5, dt);
    for (let b = 0; b < BAND_COUNT; b++) {
      const v = audio ? audio.bands[b] : 0;
      this.bands[b] = approach(this.bands[b], v, v > this.bands[b] ? 20 : 6, dt);
    }
    this.breathe = spec.breathe ? Math.sin(t * 1.4) * 0.03 : 0;
  }

  _band(angle) {
    // Mirror the spectrum around the vertical axis so the shape stays balanced.
    const x = Math.abs(((angle / TAU) % 1) * 2 - 1);
    const pos = x * (BAND_COUNT - 1);
    const i = Math.floor(pos);
    const f = pos - i;
    return this.bands[i] * (1 - f) + this.bands[Math.min(i + 1, BAND_COUNT - 1)] * f;
  }

  _draw(t) {
    const g = this.g;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const reduced = this.reduced.matches;
    g.clearRect(0, 0, w, h);

    const since = t - this.dipAt;
    const dip = since >= 0 && since < 0.55 ? Math.sin((Math.PI * since) / 0.55) * 0.16 : 0;
    const scale = (1 + this.breathe + this.energy * (reduced ? 0.06 : 0.2)) * (1 - dip);
    const base = Math.min(w, h) * 0.29 * scale;
    const c1 = this.colors[0];

    const glow = g.createRadialGradient(cx, cy, base * 0.4, cx, cy, base * 1.75);
    glow.addColorStop(0, rgba(c1, this.light ? 0.3 : 0.42));
    glow.addColorStop(1, rgba(c1, 0));
    g.fillStyle = glow;
    g.fillRect(0, 0, w, h);

    g.globalCompositeOperation = this.light ? "source-over" : "lighter";
    for (const layer of LAYERS) this._blob(cx, cy, base * layer.r, layer, reduced);
    g.globalCompositeOperation = "source-over";

    const shine = g.createRadialGradient(cx - base * 0.35, cy - base * 0.4, 0, cx - base * 0.35, cy - base * 0.4, base * 0.9);
    shine.addColorStop(0, `rgba(255, 255, 255, ${this.light ? 0.45 : 0.28})`);
    shine.addColorStop(1, "rgba(255, 255, 255, 0)");
    g.fillStyle = shine;
    g.beginPath();
    g.arc(cx, cy, base * 0.9, 0, TAU);
    g.fill();

    if (this.mode === "thinking" && !reduced) this._sparks(cx, cy, base, t);
  }

  _blob(cx, cy, radius, layer, reduced) {
    const g = this.g;
    const wob = reduced ? 0 : this.wobble;
    const react = reduced ? 0 : 0.32 * (0.35 + this.energy);
    const rot = reduced ? 0 : this.phase * layer.speed;
    const pts = [];
    for (let i = 0; i < POINTS; i++) {
      const a = (i / POINTS) * TAU;
      const [h1, h2] = layer.harm;
      const noise = Math.sin(h1 * a + rot * 2 + layer.offset) * wob
        + Math.sin(h2 * a - rot * 3 + layer.offset * 1.7) * wob * 0.6;
      const r = radius * (1 + noise + this._band(a + rot * 0.5) * react);
      pts.push([Math.cos(a) * r, Math.sin(a) * r]);
    }
    g.beginPath();
    for (let i = 0; i <= POINTS; i++) {
      const p = pts[i % POINTS];
      const q = pts[(i + 1) % POINTS];
      const mx = cx + (p[0] + q[0]) / 2;
      const my = cy + (p[1] + q[1]) / 2;
      if (i === 0) g.moveTo(mx, my);
      else g.quadraticCurveTo(cx + p[0], cy + p[1], mx, my);
    }
    g.closePath();
    const ang = rot + layer.offset;
    const grad = g.createLinearGradient(
      cx + Math.cos(ang) * radius, cy + Math.sin(ang) * radius,
      cx - Math.cos(ang) * radius, cy - Math.sin(ang) * radius);
    grad.addColorStop(0, rgba(this.colors[layer.color], layer.alpha));
    grad.addColorStop(1, rgba(this.colors[layer.fade], layer.alpha * 0.35));
    g.fillStyle = grad;
    g.fill();
  }

  _sparks(cx, cy, base, t) {
    const g = this.g;
    for (let i = 0; i < 3; i++) {
      const a = t * 2.4 + (i * TAU) / 3;
      const r = base * (1.22 + 0.05 * Math.sin(t * 3 + i));
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      const s = base * 0.07;
      const dot = g.createRadialGradient(x, y, 0, x, y, s);
      dot.addColorStop(0, rgba(this.colors[2], 0.95));
      dot.addColorStop(1, rgba(this.colors[2], 0));
      g.fillStyle = dot;
      g.beginPath();
      g.arc(x, y, s, 0, TAU);
      g.fill();
    }
  }
}

