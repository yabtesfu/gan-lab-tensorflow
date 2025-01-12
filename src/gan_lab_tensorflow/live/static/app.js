/* GAN Observatory - live client.
 *
 * Receives telemetry frames over one WebSocket and renders four panels on
 * plain Canvas (no build step, no external libs), and sends steering messages
 * back. All drawing is in CSS pixels; canvases are scaled by devicePixelRatio
 * for crispness.
 */
(() => {
  "use strict";

  const COL = {
    real: "#4ea1ff", fake: "#ff8a3d", gen: "#ffb454", disc: "#7cc4ff",
    good: "#46d19e", bad: "#ff5d6c", grid: "#1b2436", muted: "#8a95ab",
  };
  const HIST = 400;

  const el = (id) => document.getElementById(id);
  const state = {
    ws: null, lastFrame: null, connected: false,
    hist: { gen: [], disc: [], mmd: [], cov: [], prec: [] },
    frames: 0, fps: 0,
  };

  /* ---------------- canvas helpers ---------------- */
  function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function mapper(extent, w, h, pad) {
    const [xmin, xmax, ymin, ymax] = extent;
    const pw = w - pad * 2, ph = h - pad * 2;
    const sx = pw / (xmax - xmin || 1);
    const sy = ph / (ymax - ymin || 1);
    return {
      x: (x) => pad + (x - xmin) * sx,
      y: (y) => pad + ph - (y - ymin) * sy,
    };
  }

  /* ---------------- distribution + boundary heatmap ---------------- */
  function drawDistribution(frame) {
    const { ctx, w, h } = fitCanvas(el("dist"));
    ctx.clearRect(0, 0, w, h);
    const pad = 14;
    const m = mapper(frame.extent, w, h, pad);

    // decision-boundary heatmap
    const g = frame.grid, n = g.n, vals = g.values;
    const cw = (w - pad * 2) / (n - 1) + 1.2;
    const ch = (h - pad * 2) / (n - 1) + 1.2;
    const gx = (i) => frame.extent[0] + (frame.extent[1] - frame.extent[0]) * (i / (n - 1));
    const gy = (j) => frame.extent[2] + (frame.extent[3] - frame.extent[2]) * (j / (n - 1));
    for (let idx = 0; idx < vals.length; idx++) {
      const col = idx % n, row = Math.floor(idx / n);
      const p = vals[idx];
      const d = p - 0.5;
      const a = Math.min(0.42, Math.abs(d) * 0.84);
      ctx.fillStyle = d >= 0
        ? `rgba(78,161,255,${a})`
        : `rgba(255,138,61,${a})`;
      ctx.fillRect(m.x(gx(col)) - cw / 2, m.y(gy(row)) - ch / 2, cw, ch);
    }

    // points
    plotPoints(ctx, frame.real, m, COL.real, 2.6);
    plotPoints(ctx, frame.fake, m, COL.fake, 2.6);
  }

  function plotPoints(ctx, pts, m, color, r) {
    ctx.fillStyle = color;
    for (let i = 0; i < pts.length; i++) {
      ctx.beginPath();
      ctx.arc(m.x(pts[i][0]), m.y(pts[i][1]), r, 0, 6.2832);
      ctx.fill();
    }
  }

  /* ---------------- discriminator feature plane ---------------- */
  function drawFeature(frame) {
    const { ctx, w, h } = fitCanvas(el("feature"));
    ctx.clearRect(0, 0, w, h);
    const all = frame.featReal.concat(frame.featFake);
    if (!all.length) return;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [x, y] of all) {
      if (x < xmin) xmin = x; if (x > xmax) xmax = x;
      if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    }
    const px = (xmax - xmin) * 0.1 + 1e-3, py = (ymax - ymin) * 0.1 + 1e-3;
    const m = mapper([xmin - px, xmax + px, ymin - py, ymax + py], w, h, 14);
    plotPoints(ctx, frame.featReal, m, COL.real, 2.4);
    plotPoints(ctx, frame.featFake, m, COL.fake, 2.4);
  }

  /* ---------------- loss chart ---------------- */
  function drawLoss() {
    const { ctx, w, h } = fitCanvas(el("loss"));
    ctx.clearRect(0, 0, w, h);
    const g = state.hist.gen, d = state.hist.disc;
    if (g.length < 2) return;
    let lo = Infinity, hi = -Infinity;
    for (const arr of [g, d]) for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
    const pad = 10;
    line(ctx, d, w, h, pad, lo, hi, COL.disc);
    line(ctx, g, w, h, pad, lo, hi, COL.gen);
  }

  function line(ctx, arr, w, h, pad, lo, hi, color) {
    const n = arr.length;
    const sx = (w - pad * 2) / (HIST - 1);
    const start = HIST - n;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad + (start + i) * sx;
      const y = pad + (h - pad * 2) * (1 - (arr[i] - lo) / (hi - lo));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }

  /* ---------------- sparklines ---------------- */
  function drawSpark(id, arr, color, invert) {
    const { ctx, w, h } = fitCanvas(el(id));
    ctx.clearRect(0, 0, w, h);
    if (arr.length < 2) return;
    let lo = Infinity, hi = -Infinity;
    for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-6) { hi += 0.01; lo -= 0.01; }
    const pad = 3, n = arr.length;
    const sx = (w - pad * 2) / (Math.max(n, HIST) - 1);
    const start = Math.max(n, HIST) - n;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad + (start + i) * sx;
      const y = pad + (h - pad * 2) * (1 - (arr[i] - lo) / (hi - lo));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.lineTo(pad + (start + n - 1) * sx, h - pad);
    ctx.lineTo(pad + start * sx, h - pad);
    ctx.closePath();
    ctx.fillStyle = color + "22"; ctx.fill();
  }

  /* ---------------- metrics + banner ---------------- */
  function updateMetrics(frame) {
    el("mmdVal").textContent = frame.mmd.toFixed(3);
    el("covVal").textContent = Math.round(frame.coverage * 100) + "%";
    el("precVal").textContent = Math.round(frame.precision * 100) + "%";
    el("mmdVal").style.color = frame.mmd < 0.15 ? COL.good : frame.mmd > 0.4 ? COL.bad : COL.muted;
    el("covVal").style.color = frame.coverage > 0.7 ? COL.good : frame.coverage < 0.4 ? COL.bad : COL.muted;
    el("precVal").style.color = frame.precision > 0.7 ? COL.good : frame.precision < 0.4 ? COL.bad : COL.muted;
    el("banner").classList.toggle("hidden", !frame.collapsed);
  }

  function pushHist(frame) {
    const H = state.hist;
    H.gen.push(frame.genLoss); H.disc.push(frame.discLoss);
    H.mmd.push(frame.mmd); H.cov.push(frame.coverage); H.prec.push(frame.precision);
    for (const k of Object.keys(H)) if (H[k].length > HIST) H[k].shift();
  }

  /* ---------------- render one frame ---------------- */
  function render(frame) {
    state.lastFrame = frame;
    pushHist(frame);
    el("stepCount").textContent = frame.step;
    drawDistribution(frame);
    drawFeature(frame);
    drawLoss();
    drawSpark("mmdSpark", state.hist.mmd, COL.good);
    drawSpark("covSpark", state.hist.cov, COL.disc);
    drawSpark("precSpark", state.hist.prec, COL.gen);
    updateMetrics(frame);
  }

  function rerender() { if (state.lastFrame) render(state.lastFrame); }

  /* ---------------- websocket ---------------- */
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    state.ws = ws;
    ws.onopen = () => setConn(true);
    ws.onclose = () => { setConn(false); setTimeout(connect, 1200); };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "frame") { state.frames++; render(msg); }
      else if (msg.type === "state") syncState(msg);
      else if (msg.type === "runs") renderRuns(msg.runs);
      else if (msg.type === "saved") onSaved(msg.id);
      else if (msg.type === "error") setCoach(msg.message);
    };
  }

  function send(obj) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
  }

  function setConn(ok) {
    state.connected = ok;
    const c = el("conn");
    c.textContent = ok ? "live" : "reconnecting…";
    c.className = "pill " + (ok ? "pill-on" : "pill-off");
  }
  function setCoach(text) { el("coach").innerHTML = text; }

  /* ---------------- control sync from server ---------------- */
  let running = false;
  function syncState(msg) {
    running = msg.running;
    const btn = el("playBtn");
    btn.textContent = running ? "⏸ Pause" : "▶ Train";
    btn.classList.toggle("running", running);
    const c = msg.config;
    el("dataset").value = c.dataset;
    setSeg("lossToggle", "loss", c.loss);
    setSeg("speedToggle", "speed", msg.speed);
    el("ttur").checked = c.ttur;
    el("inoise").checked = c.instance_noise;
    el("dSteps").value = c.d_steps; el("dStepsVal").textContent = c.d_steps;
    el("lr").value = Math.log10(c.learning_rate).toFixed(3);
    el("lrVal").textContent = fmtLr(c.learning_rate);
  }
  function setSeg(groupId, attr, value) {
    for (const b of el(groupId).querySelectorAll(".seg"))
      b.classList.toggle("active", b.dataset[attr] === value);
  }
  function fmtLr(v) { return v.toExponential(1).replace("e", "e"); }

  /* ---------------- model registry ---------------- */
  const DS_NAMES = { mixture: "3-mode mixture", ring: "8-Gaussian ring", quadratic: "quadratic", sine: "sine" };
  let activeRun = null;
  let lastRuns = [];

  function renderRuns(runs) {
    if (runs) lastRuns = runs;
    const box = el("runlist");
    if (!lastRuns.length) { box.innerHTML = '<p class="empty">No saved runs yet. Train a model, then hit <b>Save run</b>.</p>'; return; }
    box.innerHTML = "";
    lastRuns.forEach((r) => {
      const card = document.createElement("button");
      card.className = "runcard" + (r.id === activeRun ? " active" : "");
      const tag = r.collapsed ? '<span class="rc-tag badge-bad">collapsed</span>'
        : '<span class="rc-tag badge-ok">cov ' + Math.round(r.coverage * 100) + "%</span>";
      card.innerHTML =
        '<div class="rc-top"><span class="rc-id">#' + r.id + "</span>" + tag + "</div>" +
        '<span class="rc-name">' + (DS_NAMES[r.dataset] || r.dataset) + " &middot; " + r.loss + "</span>" +
        '<div class="rc-metrics"><span>MMD <b>' + r.mmd.toFixed(2) + "</b></span><span>step <b>" + r.step + "</b></span></div>";
      card.addEventListener("click", () => sampleRun(r));
      box.appendChild(card);
    });
  }

  async function sampleRun(r) {
    activeRun = r.id;
    renderRuns();
    try {
      const res = await fetch("/api/runs/" + r.id + "/sample?count=260");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      drawPreview(data.points, data.extent);
      el("prevMeta").textContent = "run #" + r.id + " · " + data.points.length + " pts · GET /api/runs/" + r.id + "/sample";
    } catch (e) {
      el("prevMeta").textContent = "sample failed: " + e.message;
    }
  }

  function drawPreview(pts, extent) {
    const { ctx, w, h } = fitCanvas(el("preview"));
    ctx.clearRect(0, 0, w, h);
    const m = mapper(extent, w, h, 12);
    ctx.globalAlpha = 0.85; ctx.fillStyle = COL.fake;
    for (let i = 0; i < pts.length; i++) { ctx.beginPath(); ctx.arc(m.x(pts[i][0]), m.y(pts[i][1]), 2.6, 0, 6.2832); ctx.fill(); }
    ctx.globalAlpha = 1;
  }

  function onSaved(id) {
    setCoach("Saved run <strong>#" + id + "</strong> to the registry. Click its card below to serve fresh samples from that generator through the API.");
  }
  async function loadRuns() {
    try { const res = await fetch("/api/runs"); const data = await res.json(); renderRuns(data.runs); }
    catch (e) { /* registry unavailable */ }
  }

  /* ---------------- control bindings ---------------- */
  function bind() {
    el("saveBtn").onclick = () => send({ action: "save" });
    el("playBtn").onclick = () => send({ action: "toggle" });
    el("resetBtn").onclick = () => { send({ action: "reset" }); setCoach("Reset. Press <strong>Train</strong> to watch it learn from scratch."); };
    el("demoBtn").onclick = () => {
      send({ action: "demo" });
      setCoach("Demo loaded: a deliberately unstable vanilla GAN. Hit <strong>Train</strong> and watch it <em>collapse</em> onto a single mode. Then <strong>Reset</strong> and switch <strong>Loss → Wasserstein</strong> to see all three modes return - or try to claw it back live with <strong>Instance noise</strong> + a lower learning rate.");
    };
    el("dataset").onchange = (e) => send({ action: "config", value: { dataset: e.target.value } });
    el("ttur").onchange = (e) => send({ action: "config", value: { ttur: e.target.checked } });
    el("inoise").onchange = (e) => send({ action: "config", value: { instance_noise: e.target.checked } });

    el("lr").oninput = (e) => {
      const lr = Math.pow(10, parseFloat(e.target.value));
      el("lrVal").textContent = fmtLr(lr);
      send({ action: "config", value: { learning_rate: lr } });
    };
    el("dSteps").oninput = (e) => {
      el("dStepsVal").textContent = e.target.value;
      send({ action: "config", value: { d_steps: parseInt(e.target.value, 10) } });
    };
    for (const b of el("lossToggle").querySelectorAll(".seg"))
      b.onclick = () => send({ action: "config", value: { loss: b.dataset.loss } });
    for (const b of el("speedToggle").querySelectorAll(".seg"))
      b.onclick = () => send({ action: "speed", value: b.dataset.speed });
  }

  window.addEventListener("resize", rerender);
  setInterval(() => { state.fps = state.frames; state.frames = 0; el("fps").textContent = state.fps; }, 1000);

  bind();
  loadRuns();
  connect();
})();
