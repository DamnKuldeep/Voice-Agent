// The live conversation as chat bubbles, driven by session events.
//
// Event order from the pipeline: partial* -> final -> reply* (cumulative text) -> reply(final) -> turn.
// An interruption: partial(barge)* -> barge(duck) -> barge(stop) -> turn(interrupted) -> partial* -> ...
// A dismissed one (backchannel or echo): partial(barge)* -> barge(duck) -> barge(resume).

function joinWords(base, text) {
  if (!base) return text;
  if (!text) return base;
  const first = (s) => s.trim().split(/\s+/)[0].toLowerCase();
  // When the recogniser kept the earlier words, the partial already starts with them.
  return first(text) === first(base) ? text : `${base} ${text}`;
}

export class Transcript {
  constructor(list, scroller, empty) {
    this.list = list;
    this.scroller = scroller;
    this.empty = empty;
    this.reset();
  }

  reset() {
    this.list.textContent = "";
    this.empty.hidden = false;
    this.user = null;        // open user bubble {el, base}
    this.barge = null;       // user bubble for words heard while the agent talks
    this.lastUser = null;    // last finished user bubble, reopened if the user keeps talking
    this.agent = null;       // open agent bubble element
    this.lastAgent = null;
    this.agentSinceUser = false;
  }

  _bubble(role, extra = "") {
    const el = document.createElement("li");
    el.className = `msg ${role} ${extra}`.trim();
    el.appendChild(document.createElement("span"));
    this.list.appendChild(el);
    this.empty.hidden = true;
    return el;
  }

  _setText(el, text) {
    const nearBottom = this.scroller.scrollHeight - this.scroller.scrollTop - this.scroller.clientHeight < 80;
    el.firstChild.textContent = text;
    if (nearBottom) this.scroller.scrollTop = this.scroller.scrollHeight;
  }

  _tag(el, text) {
    if (el.querySelector(".tag")) return;
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = text;
    el.appendChild(tag);
  }

  partial(text, barge) {
    if (barge) {
      if (!this.barge) this.barge = { el: this._bubble("user", "live barge"), base: "" };
      this._setText(this.barge.el, text);
      return;
    }
    if (!this.user && this.barge) {
      // The interruption was confirmed: its bubble becomes the new turn's bubble.
      this.user = this.barge;
      this.user.base = this.barge.el.firstChild.textContent;
      this.user.el.classList.remove("barge");
      this.barge = null;
    }
    if (!this.user) {
      if (this.lastUser && !this.agentSinceUser) {
        this.user = { el: this.lastUser, base: this.lastUser.firstChild.textContent };
        this.user.el.classList.add("live");
      } else {
        this.user = { el: this._bubble("user", "live"), base: "" };
      }
    }
    this._setText(this.user.el, joinWords(this.user.base, text));
  }

  final(text) {
    const open = this.user || this.barge || { el: this._bubble("user") };
    open.el.classList.remove("live", "barge");
    this._setText(open.el, text);
    this.lastUser = open.el;
    this.user = this.barge = null;
    this.agentSinceUser = false;
  }

  reply(text, final) {
    if (!this.agent) {
      this.agent = this._bubble("agent", "live");
      this.agentSinceUser = true;
    }
    this._setText(this.agent, text);
    this.lastAgent = this.agent;
    if (final) {
      this.agent.classList.remove("live");
      this.agent = null;
    }
  }

  _markInterrupted() {
    const el = this.agent || this.lastAgent;
    if (!el) return;
    el.classList.remove("live");
    el.classList.add("interrupted");
    this._tag(el, "interrupted");
    this.agent = null;
  }

  bargePhase(phase) {
    if (phase === "stop") {
      this._markInterrupted();
    } else if (phase === "resume" && this.barge) {
      // Not an interruption after all (a backchannel or the agent's own echo).
      const el = this.barge.el;
      this.barge = null;
      el.classList.remove("live");
      el.classList.add("dismissed");
      setTimeout(() => el.remove(), 1500);
    }
  }

  turn(event) {
    const flags = event.flags || {};
    if (flags.interrupted) this._markInterrupted();
  }

  state(state) {
    if (state === "listening" && this.agent) {
      // The user kept talking before the reply played: the pipeline dropped it, so do we.
      this.agent.remove();
      this.agent = this.lastAgent = null;
      this.agentSinceUser = false;
    }
  }
}
