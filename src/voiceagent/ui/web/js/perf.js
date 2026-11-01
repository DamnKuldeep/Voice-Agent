// The Performance panel: last-turn stage bar, facts, rolling p50/p90 and the recogniser lag.

const STAGES = [
  ["endpoint_ms", "End-of-speech wait", "--s1"],
  ["asr_final_ms", "Transcript", "--s2"],
  ["llm_ttft_ms", "LLM first token", "--s3"],
  ["first_chunk_ms", "First chunk", "--s4"],
  ["tts_ttfa_ms", "First audio", "--s5"],
];
const SUMMARY = [
  ...STAGES.map(([key, label]) => [key, label]),
  ["v2v_ms", "Voice to voice"],
  ["asr_flush_ms", "Transcript flush"],
  ["reply_audio_ms", "Reply length"],
  ["barge_in_duck_ms", "Interruption: duck"],
  ["barge_in_stop_ms", "Interruption: stop"],
];
const LAG_SHOW_MS = 300;
const LAG_FAR_MS = 1500;

function ms(value) {
  if (value === undefined || value === null) return "-";
  return value >= 10000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
}

export class PerfPanel {
  constructor(root) {
    this.root = root;
    this.bar = root.querySelector("#stagebar");
    this.legend = root.querySelector("#stage-legend");
    this.facts = root.querySelector("#facts");
    this.body = root.querySelector("#summary-body");
    this.count = root.querySelector("#summary-n");
    this.lagEl = root.querySelector("#asr-lag");
    this.config = {};
    this.vram = null;
    this.last = null;
  }

  status(status) {
    this.config = status.config || {};
    if ("vram" in status) this.vram = status.vram;
    if (status.last_turn) this.turn(status.last_turn);
    else this._facts();
    this.summary(status.summary || {});
    this.lag(status.asr_lag_ms || 0);
  }

  turn(event) {
    this.last = { metrics: event.metrics || {}, flags: event.flags || {} };
    const metrics = this.last.metrics;
    const present = STAGES.filter(([key]) => metrics[key] !== undefined);
    this.bar.textContent = "";
    this.legend.textContent = "";
    if (!present.length) {
      this.bar.innerHTML = "<em>No completed turn yet.</em>";
    }
    const total = present.reduce((sum, [key]) => sum + Math.max(metrics[key], 0), 0) || 1;
    for (const [key, label, colour] of present) {
      const seg = document.createElement("div");
      seg.className = "seg";
      seg.style.width = `${(Math.max(metrics[key], 0) / total) * 100}%`;
      seg.style.background = `var(${colour})`;
      seg.title = `${label}: ${ms(metrics[key])}`;
      this.bar.appendChild(seg);
      const item = document.createElement("span");
      item.innerHTML = `<i style="background: var(${colour})"></i>${label} <b>${ms(metrics[key])}</b>`;
      this.legend.appendChild(item);
    }
    this._facts();
  }

  _facts() {
    const m = (this.last && this.last.metrics) || {};
    const f = (this.last && this.last.flags) || {};
    const c = this.config;
    const provider = f.llm_provider || (c.provider_order && c.provider_order.length ? c.provider_order.join(" > ") : `auto (${c.routing_sort || "-"})`);
    const rows = [
      ["Voice to voice", ms(m.v2v_ms)],
      ["Model that answered", f.llm_model || c.llm_model || "-"],
      ["Provider", provider],
      ["Recogniser step", c.asr_chunk_ms ? `${c.asr_chunk_ms} ms` : "-"],
      ["Most recogniser lag", f.asr_max_lag_ms !== undefined ? ms(f.asr_max_lag_ms) : "-"],
      ["GPU memory", this.vram ? `${(this.vram.used_mib / 1024).toFixed(2)} / ${(this.vram.total_mib / 1024).toFixed(2)} GiB` : "-"],
    ];
    if (m.barge_in_duck_ms !== undefined) rows.push(["Interruption: duck", ms(m.barge_in_duck_ms)]);
    if (m.barge_in_stop_ms !== undefined) rows.push(["Interruption: stop", ms(m.barge_in_stop_ms)]);
    if (f.speculation_used) rows.push(["Early start", "used"]);
    this.facts.textContent = "";
    for (const [label, value] of rows) {
      const div = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = value;
      dd.title = value;
      div.append(dt, dd);
      this.facts.appendChild(div);
    }
  }

  summary(summary) {
    this.body.textContent = "";
    let n = 0;
    for (const [key, label] of SUMMARY) {
      const stats = summary[key];
      if (!stats) continue;
      n = Math.max(n, stats.n);
      const tr = document.createElement("tr");
      for (const text of [label, String(Math.round(stats.n)), ms(stats.p50), ms(stats.p90)]) {
        const td = document.createElement("td");
        td.textContent = text;
        tr.appendChild(td);
      }
      this.body.appendChild(tr);
    }
    this.count.textContent = n ? `over ${n} turn${n > 1 ? "s" : ""}` + (n < 30 ? " (under 30: percentiles are rough)" : "") : "";
  }

  setVram(vram) {
    this.vram = vram;
    this._facts();
  }

  lag(lagMs) {
    const el = this.lagEl;
    el.classList.toggle("behind", lagMs >= LAG_SHOW_MS && lagMs < LAG_FAR_MS);
    el.classList.toggle("far", lagMs >= LAG_FAR_MS);
    el.textContent = lagMs >= LAG_SHOW_MS ? `Recogniser ${ms(lagMs)} behind` : "Recogniser keeping up";
  }
}
