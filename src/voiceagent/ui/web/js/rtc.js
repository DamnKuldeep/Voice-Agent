// WebRTC to the FastRTC offer endpoint: our mic track goes up, the agent's voice comes back.
//
// Signalling follows fastrtc 0.0.34 (stream.py, webrtc_connection_mixin.py): POST /webrtc/offer with
// {sdp, type: "offer", webrtc_id}; the reply is the SDP answer or {status: "failed", meta: {error}}.
// Candidates found after the offer was sent go to the same endpoint as {type: "ice-candidate"}.
// Server errors (for example "no microphone audio") arrive on a data channel named "text".

const ICE_WAIT_MS = 1500;

export const MIC_CONSTRAINTS = {
  audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
};

function newId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function micErrorMessage(err) {
  const name = err && err.name;
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Microphone blocked: allow it in the address bar (the padlock or microphone icon), then press Start again.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No microphone found. Plug one in, or pick an input device in Windows Sound settings.";
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return "The microphone is busy or blocked by Windows. Close other apps using it and check Settings > Privacy > Microphone.";
  }
  return `Could not open the microphone (${name || err}).`;
}

function waitForIce(pc, ms) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", done);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", done);
    setTimeout(resolve, ms);
  });
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  return res.json();
}

const OFFER_ERRORS = {
  concurrency_limit_reached: "Too many open connections. Close other tabs with this page, wait a few seconds, then try again.",
  connection_already_exists: "This connection is already open.",
};

export class RtcCall {
  // handlers: onLocalStream(stream), onRemoteStream(stream), onServerError(text), onDrop()
  constructor(handlers) {
    this.h = handlers;
    this.pc = null;
    this.stream = null;
    this.id = null;
  }

  get connected() {
    return this.pc !== null;
  }

  async start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This page needs a secure address for the microphone: open it at 127.0.0.1 or localhost in the address bar.");
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
    } catch (err) {
      throw new Error(micErrorMessage(err));
    }
    this.stream = stream;
    this.h.onLocalStream(stream);
    // A TURN relay when the page is shared through a tunnel (the audio cannot use the tunnel).
    const ice = await fetch("/api/ice").then((r) => r.json()).catch(() => ({ iceServers: [] }));
    const pc = new RTCPeerConnection({ iceServers: ice.iceServers || [] });
    this.pc = pc;
    this.id = newId();
    const channel = pc.createDataChannel("text");
    channel.onmessage = (e) => this._onChannel(e.data);
    pc.ontrack = (e) => this.h.onRemoteStream(e.streams[0] || new MediaStream([e.track]));
    pc.onconnectionstatechange = () => {
      if (this.pc === pc && (pc.connectionState === "failed" || pc.connectionState === "closed")) {
        this._teardown();
        this.h.onDrop();
      }
    };
    for (const track of stream.getAudioTracks()) pc.addTrack(track, stream);

    let sent = false;
    let answered = false;
    const late = [];
    pc.onicecandidate = ({ candidate }) => {
      if (!candidate || !sent) return; // before the offer is sent, candidates ride in its SDP
      const msg = { candidate: candidate.toJSON(), webrtc_id: this.id, type: "ice-candidate" };
      if (answered) postJson("/webrtc/offer", msg).catch(() => {});
      else late.push(msg);
    };
    await pc.setLocalDescription(await pc.createOffer());
    await waitForIce(pc, ICE_WAIT_MS);
    sent = true;
    const answer = await postJson("/webrtc/offer", {
      sdp: pc.localDescription.sdp, type: pc.localDescription.type, webrtc_id: this.id,
    });
    if (answer.status === "failed") {
      const code = answer.meta && answer.meta.error;
      this.stop();
      throw new Error(OFFER_ERRORS[code] || `The server refused the connection (${code}).`);
    }
    await pc.setRemoteDescription(answer);
    answered = true;
    for (const msg of late) postJson("/webrtc/offer", msg).catch(() => {});
  }

  _onChannel(data) {
    let msg;
    try {
      msg = JSON.parse(data);
    } catch {
      return;
    }
    if (msg && msg.type === "error") this.h.onServerError(msg.message || String(msg.data || "Server error"));
  }

  // Tell the server first so the session ends at once, not when the peer connection times out.
  stop() {
    if (this.id) {
      postJson("/api/stop", { webrtc_id: this.id }).catch(() => {});
    }
    this._teardown();
  }

  // For page unload, where a normal fetch may be cancelled.
  beacon() {
    if (this.id && navigator.sendBeacon) {
      const blob = new Blob([JSON.stringify({ webrtc_id: this.id })], { type: "application/json" });
      navigator.sendBeacon("/api/stop", blob);
    }
  }

  _teardown() {
    const pc = this.pc;
    this.pc = null;
    if (pc) {
      pc.onconnectionstatechange = null;
      pc.close();
    }
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    this.stream = null;
    this.id = null;
  }
}
