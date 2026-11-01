// The settings drawer, built from GET /api/controls; Apply sends only the controls that changed.

function el(tag, attrs = {}, text = "") {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  if (text) node.textContent = text;
  return node;
}

function shown(control, value) {
  if (control.kind === "list") return (value || []).join(", ");
  if (control.kind === "slider" || control.kind === "number") {
    const digits = control.step && control.step < 1 ? String(control.step).split(".")[1].length : 0;
    return Number(value).toFixed(digits) + (control.unit ? ` ${control.unit}` : "");
  }
  return String(value);
}

function same(control, a, b) {
  if (control.kind === "slider" || control.kind === "number") return Math.abs(Number(a) - Number(b)) < 1e-9;
  if (control.kind === "list") return shown(control, a) === shown(control, b);
  return String(a) === String(b);
}

export class SettingsDrawer {
  constructor(root, { onMessage }) {
    this.root = root;
    this.form = root.querySelector("#settings-form");
    this.tabs = root.querySelector("#settings-tabs");
    this.applyBtn = root.querySelector("#apply-btn");
    this.status = root.querySelector("#apply-status");
    this.scrim = document.getElementById("scrim");
    this.onMessage = onMessage;
    this.controls = [];
    this.inputs = new Map();
    this.group = "listening";
    this.busy = false;
    this.applyBtn.addEventListener("click", () => this.apply());
    root.querySelector("#defaults-btn").addEventListener("click", () => this.fillDefaults());
    root.querySelector("#drawer-close").addEventListener("click", () => this.close());
    this.scrim.addEventListener("click", () => this.close());
    this.form.addEventListener("submit", (e) => e.preventDefault());
  }

  get isOpen() {
    return this.root.classList.contains("open");
  }

  open() {
    this.root.classList.add("open");
    this.root.setAttribute("aria-hidden", "false");
    this.scrim.hidden = false;
    document.getElementById("settings-btn").setAttribute("aria-expanded", "true");
    if (!this.controls.length) this.load();
  }

  close() {
    this.root.classList.remove("open");
    this.root.setAttribute("aria-hidden", "true");
    this.scrim.hidden = true;
    document.getElementById("settings-btn").setAttribute("aria-expanded", "false");
  }

  async load() {
    try {
      const res = await fetch("/api/controls");
      this.render(await res.json());
    } catch {
      this._say("Could not load the settings from the server.", true);
    }
  }

  render(defs) {
    this.controls = defs.controls;
    this.groups = defs.groups;
    this.inputs.clear();
    this.form.textContent = "";
    this.tabs.textContent = "";
    for (const group of defs.groups) {
      const tab = el("button", { type: "button", class: "tab", role: "tab", "data-group": group.id }, group.label);
      tab.addEventListener("click", () => this._show(group.id));
      this.tabs.appendChild(tab);
      const section = el("section", { class: "group", "data-group": group.id, role: "tabpanel" });
      for (const control of this.controls.filter((c) => c.group === group.id)) {
        section.appendChild(this._control(control));
      }
      this.form.appendChild(section);
    }
    this._show(this.group);
    this._refresh();
  }

  _show(groupId) {
    this.group = groupId;
    for (const tab of this.tabs.children) tab.setAttribute("aria-selected", String(tab.dataset.group === groupId));
    for (const sec of this.form.children) sec.hidden = sec.dataset.group !== groupId;
  }

  _control(c) {
    const id = `ctl-${c.key.replace(/\./g, "-")}`;
    const box = el("div", { class: "ctl", "data-key": c.key });
    const top = el("div", { class: "ctl-top" });
    const label = el("label", { class: "ctl-label", for: id }, c.label);
    if (c.needs_reload) label.appendChild(el("span", { class: "badge", title: "Applying this reloads the speech models" }, "reloads models"));
    top.appendChild(label);
    let input;
    let readout = null;
    if (c.kind === "toggle") {
      const sw = el("span", { class: "switch" });
      input = el("input", { type: "checkbox", id, role: "switch" });
      input.checked = Boolean(c.value);
      sw.append(input, el("span"));
      top.appendChild(sw);
    } else if (c.kind === "slider") {
      readout = el("span", { class: "ctl-value" }, shown(c, c.value));
      top.appendChild(readout);
      input = el("input", { type: "range", id, min: c.min, max: c.max, step: c.step });
      input.value = c.value;
    } else if (c.kind === "select") {
      input = el("select", { id });
      for (const choice of c.choices) input.appendChild(el("option", { value: choice }, shown({ ...c, kind: "text" }, choice) + (c.unit ? ` ${c.unit}` : "")));
      input.value = String(c.value);
    } else if (c.kind === "number") {
      input = el("input", { type: "number", id, min: c.min, max: c.max, step: c.step });
      input.value = c.value;
    } else {
      input = el("input", { type: "text", id, spellcheck: "false", autocomplete: "off" });
      input.value = shown(c, c.value);
      if (c.suggestions.length) {
        const list = el("datalist", { id: `${id}-list` });
        for (const s of c.suggestions) list.appendChild(el("option", { value: s }));
        input.setAttribute("list", list.id);
        box.appendChild(list);
      }
    }
    box.appendChild(top);
    box.appendChild(el("p", { class: "ctl-help" }, c.help));
    if (c.kind !== "toggle") box.insertBefore(input, box.lastChild);
    input.addEventListener("input", () => {
      if (readout) readout.textContent = shown(c, input.value);
      this._refresh();
    });
    input.addEventListener("change", () => this._refresh());
    this.inputs.set(c.key, { control: c, input, readout, box });
    return box;
  }

  _read({ control, input }) {
    if (control.kind === "toggle") return input.checked;
    if (control.kind === "slider" || control.kind === "number") return Number(input.value);
    return input.value;
  }

  _write(entry, value) {
    const { control, input, readout } = entry;
    if (control.kind === "toggle") input.checked = Boolean(value);
    else if (control.kind === "list") input.value = shown(control, value);
    else input.value = String(value);
    if (readout) readout.textContent = shown(control, value);
  }

  changes() {
    const out = {};
    for (const [key, entry] of this.inputs) {
      const value = this._read(entry);
      if (!same(entry.control, value, entry.control.value)) out[key] = value;
    }
    return out;
  }

  _refresh() {
    const changed = this.changes();
    const groups = new Set();
    for (const [key, entry] of this.inputs) {
      const on = key in changed;
      entry.box.classList.toggle("changed", on);
      if (on) groups.add(entry.control.group);
    }
    for (const tab of this.tabs.children) {
      const dot = tab.querySelector(".dot");
      if (groups.has(tab.dataset.group) && !dot) tab.appendChild(el("span", { class: "dot" }));
      if (!groups.has(tab.dataset.group) && dot) dot.remove();
    }
    const count = Object.keys(changed).length;
    const reload = Object.keys(changed).some((k) => this.inputs.get(k).control.needs_reload);
    this.applyBtn.disabled = this.busy || count === 0;
    this.applyBtn.textContent = reload ? "Apply and reload" : "Apply";
    if (!this.busy) this._say(count ? `${count} unsaved change${count > 1 ? "s" : ""}` : "");
  }

  fillDefaults() {
    for (const entry of this.inputs.values()) this._write(entry, entry.control.default);
    this._refresh();
    if (Object.keys(this.changes()).length) this._say("Defaults filled in. Press Apply to use them.");
  }

  async apply() {
    const values = this.changes();
    if (!Object.keys(values).length || this.busy) return;
    const reload = Object.keys(values).some((k) => this.inputs.get(k).control.needs_reload);
    this.busy = true;
    this.applyBtn.disabled = true;
    this._say(reload ? "Applying and reloading the models; this takes a while..." : "Applying...");
    try {
      const res = await fetch("/api/controls", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || res.statusText);
      this.busy = false;
      this.render(body);
      this._say(body.message, !body.ok);
      this.onMessage(body.message, !body.ok);
    } catch (err) {
      this.busy = false;
      this._refresh();
      this._say(`Not applied: ${err.message}`, true);
    }
  }

  _say(text, error = false) {
    this.status.textContent = text;
    this.status.classList.toggle("error", error);
  }
}
