/* A/B Derby - two synchronized GANs streamed as paired frames.
 *
 * Renders a left/right viewport plus a coverage-vs-coverage comparison chart,
 * and keeps a client-side ring buffer of frames so you can scrub back through
 * training (time travel) without any server state.
 */
(() => {
  "use strict";

  const COL = { real: "#4ea1ff", fake: "#ff8a3d", left: "#ff5d6c", right: "#46d19e", good: "#46d19e", bad: "#ff5d6c", muted: "#8a95ab" };
  const CAP = 600;
  const el = (id) => document.getElementById(id);

  const store = { ws: null, buffer: [], following: true, frames: 0 };

  /* ---- canvas helpers ---- */
  function fit(cv) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const r = cv.getBoundingClientRect();
    const w = Math.max(1, r.width | 0), h = Math.max(1, r.height | 0);
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  function mapper(extent, w, h, pad) {
    const [x0, x1, y0, y1] = extent, pw = w - pad * 2, ph = h - pad * 2;
    return { x: (x) => pad + (x - x0) / (x1 - x0) * pw, y: (y) => pad + ph - (y - y0) / (y1 - y0) * ph };
  }
  function dots(ctx, pts, m, color, r, alpha) {
    ctx.globalAlpha = alpha; ctx.fillStyle = color;
    for (let i = 0; i < pts.length; i++) { ctx.beginPath(); ctx.arc(m.x(pts[i][0]), m.y(pts[i][1]), r, 0, 6.2832); ctx.fill(); }
    ctx.globalAlpha = 1;
  }

  function drawViewport(cvId, real, side, extent, n) {
    const { ctx, w, h } = fit(el(cvId));
    ctx.clearRect(0, 0, w, h);
    const pad = 12, m = mapper(extent, w, h, pad), g = side.grid;
    const cw = (w - pad * 2) / (n - 1) + 1.3, ch = (h - pad * 2) / (n - 1) + 1.3;
    const gx = (i) => extent[0] + (extent[1] - extent[0]) * (i / (n - 1));
    const gy = (j) => extent[2] + (extent[3] - extent[2]) * (j / (n - 1));
    for (let i = 0; i < g.length; i++) {
      const col = i % n, row = (i / n) | 0, d = g[i] - 0.5, a = Math.min(0.4, Math.abs(d) * 0.8);
      ctx.fillStyle = d >= 0 ? "rgba(78,161,255," + a + ")" : "rgba(255,138,61," + a + ")";
      ctx.fillRect(m.x(gx(col)) - cw / 2, m.y(gy(row)) - ch / 2, cw, ch);
    }
    dots(ctx, real, m, COL.real, 2.5, 0.7);
    dots(ctx, side.fake, m, COL.fake, 2.7, 0.9);
  }

  function drawCompare(upTo) {
    const { ctx, w, h } = fit(el("compare"));
    ctx.clearRect(0, 0, w, h);
    const pad = 10, n = store.buffer.length;
    if (n < 2) return;
    // gridlines at coverage 0.5 and 1.0
    ctx.strokeStyle = "#1b2436"; ctx.lineWidth = 1;
    [0, 0.5, 1].forEach((v) => {
      const y = pad + (h - pad * 2) * (1 - v);
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    });
    const line = (key, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      const sx = (w - pad * 2) / (CAP - 1), start = CAP - n;
      for (let i = 0; i <= upTo && i < n; i++) {
        const x = pad + (start + i) * sx, y = pad + (h - pad * 2) * (1 - store.buffer[i][key]);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
    };
    line("lcov", COL.left);
    line("rcov", COL.right);
  }

  /* ---- render one paired frame ---- */
  function render(frame, idx) {
    drawViewport("leftView", frame.real, frame.left, frame.extent, frame.gridN);
    drawViewport("rightView", frame.real, frame.right, frame.extent, frame.gridN);
    badge("lcov", "leftPanel", frame.left);
    badge("rcov", "rightPanel", frame.right);
    drawCompare(idx);
    el("stepCount").textContent = frame.step;
    verdict(frame);
  }
  function badge(covId, panelId, side) {
    el(covId).textContent = "coverage " + Math.round(side.coverage * 100) + "%";
    el(covId).style.color = side.coverage > 0.7 ? COL.good : side.coverage < 0.45 ? COL.bad : COL.muted;
    el(panelId).classList.toggle("collapsed", !!side.collapsed);
  }
  function verdict(frame) {
    const l = frame.left.coverage, r = frame.right.coverage;
    const v = el("verdict");
    if (frame.step < 40) { v.textContent = "identical start…"; v.className = "verdict"; return; }
    const diff = Math.round((r - l) * 100);
    if (diff > 12) { v.textContent = "Wasserstein covers " + diff + " pts more of the distribution"; v.className = "verdict good"; }
    else if (diff < -12) { v.textContent = "Vanilla ahead by " + (-diff) + " pts"; v.className = "verdict bad"; }
    else { v.textContent = "neck and neck"; v.className = "verdict"; }
  }

  /* ---- ring buffer + time travel ---- */
  function onFrame(frame) {
    store.frames++;
    store.buffer.push({ full: frame, lcov: frame.left.coverage, rcov: frame.right.coverage });
    if (store.buffer.length > CAP) store.buffer.shift();
    const sc = el("scrub");
    sc.max = store.buffer.length - 1;
    if (store.following) { sc.value = sc.max; el("ttLabel").textContent = "live"; render(frame, store.buffer.length - 1); }
  }
  function scrubTo(idx) {
    store.following = false;
    el("liveBtn").classList.remove("active");
    const item = store.buffer[idx];
    if (item) { el("ttLabel").textContent = "step " + item.full.step; render(item.full, idx); }
  }
  function goLive() {
    store.following = true;
    el("liveBtn").classList.add("active");
    const sc = el("scrub"); sc.value = sc.max;
    const item = store.buffer[store.buffer.length - 1];
    if (item) { el("ttLabel").textContent = "live"; render(item.full, store.buffer.length - 1); }
  }

  /* ---- websocket ---- */
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(proto + "://" + location.host + "/ws/derby");
    store.ws = ws;
    ws.onopen = () => setConn(true);
    ws.onclose = () => { setConn(false); setTimeout(connect, 1200); };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "derby") onFrame(msg);
      else if (msg.type === "state") syncState(msg);
      else if (msg.type === "error") el("coach").textContent = msg.message;
    };
  }
  function send(o) { if (store.ws && store.ws.readyState === WebSocket.OPEN) store.ws.send(JSON.stringify(o)); }
  function setConn(ok) { const c = el("conn"); c.textContent = ok ? "live" : "reconnecting…"; c.className = "pill " + (ok ? "pill-on" : "pill-off"); }

  let running = false;
  function syncState(msg) {
    running = msg.running;
    const b = el("playBtn"); b.textContent = running ? "⏸ Pause" : "▶ Race"; b.classList.toggle("running", running);
    for (const s of el("datasetToggle").querySelectorAll(".seg")) s.classList.toggle("active", s.dataset.ds === msg.dataset);
    for (const s of el("speedToggle").querySelectorAll(".seg")) s.classList.toggle("active", s.dataset.speed === msg.speed);
  }

  /* ---- bindings ---- */
  el("playBtn").onclick = () => send({ action: "toggle" });
  el("resetBtn").onclick = () => { store.buffer = []; goLive(); send({ action: "reset" }); };
  for (const s of el("datasetToggle").querySelectorAll(".seg"))
    s.onclick = () => { store.buffer = []; goLive(); send({ action: "dataset", value: s.dataset.ds }); };
  for (const s of el("speedToggle").querySelectorAll(".seg"))
    s.onclick = () => send({ action: "speed", value: s.dataset.speed });
  el("scrub").oninput = (e) => scrubTo(+e.target.value);
  el("liveBtn").onclick = goLive;
  window.addEventListener("resize", () => {
    const item = store.buffer[store.following ? store.buffer.length - 1 : +el("scrub").value];
    if (item) render(item.full, store.following ? store.buffer.length - 1 : +el("scrub").value);
  });
  setInterval(() => { el("fps").textContent = store.frames; store.frames = 0; }, 1000);

  connect();
})();
