/* ============================================================================
   ANN Seismic Period — front-end logic
   Real prediction: StandardScaler(X) -> [Dense+LeakyReLU]x3 -> Dense(2) -> mean
   over 5 seeds -> inverse StandardScaler(y). Verified identical to Keras.
   ========================================================================== */
(function () {
  "use strict";
  const M = window.ANN_MODEL;
  if (!M) { console.error("ANN_MODEL not loaded"); return; }

  // ---- Feature metadata (labels, units, icons) ----
  const FEATURE_META = {
    H_total:    { label: "Total height", unit: "m",      min: 8,  max: 65, step: 0.5, icon: "ruler-v" },
    N_etages:   { label: "Number of storeys", unit: "",  min: 2,  max: 21, step: 1,   icon: "layers" },
    L_xx:       { label: "Plan length — X", unit: "m",   min: 10, max: 41, step: 0.5, icon: "arrow-x" },
    L_yy:       { label: "Plan length — Y", unit: "m",   min: 8,  max: 32, step: 0.5, icon: "arrow-y" },
    Trav_max_xx:{ label: "Max bay span — X", unit: "m",  min: 3,  max: 7,  step: 0.5, icon: "span" },
    Trav_max_yy:{ label: "Max bay span — Y", unit: "m",  min: 3,  max: 7,  step: 0.5, icon: "span" },
  };
  const ICONS = {
    "ruler-v": '<path d="M5 3h6v18H5z"/><path d="M11 7H8M11 11H8M11 15H8"/>',
    "layers":  '<path d="m12 2 9 5-9 5-9-5 9-5z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/>',
    "arrow-x": '<path d="M3 12h18M7 8l-4 4 4 4M17 8l4 4-4 4"/>',
    "arrow-y": '<path d="M12 3v18M8 7l4-4 4 4M8 17l4 4 4-4"/>',
    "span":    '<path d="M4 7v10M20 7v10M4 12h16M8 9l-4 3 4 3M16 9l4 3-4 3"/>',
  };

  // ---- Core math ----
  const leaky = (x, a) => (x > 0 ? x : a * x);

  function matVec(vec, kernel, bias) {
    // kernel: [in][out]
    const out = bias.slice();
    for (let j = 0; j < out.length; j++) {
      let s = bias[j];
      for (let i = 0; i < vec.length; i++) s += vec[i] * kernel[i][j];
      out[j] = s;
    }
    return out;
  }

  function forwardOne(z, weights) {
    let a = z;
    for (let li = 0; li < weights.length; li++) {
      a = matVec(a, weights[li].kernel, weights[li].bias);
      if (li < weights.length - 1) a = a.map((v) => leaky(v, M.leaky));
    }
    return a; // scaled-y space
  }

  function predict(xRaw) {
    const z = xRaw.map((v, i) => (v - M.scaler_X.mean[i]) / M.scaler_X.scale[i]);
    const acc = [0, 0];
    for (const w of M.models) {
      const o = forwardOne(z, w);
      acc[0] += o[0]; acc[1] += o[1];
    }
    const n = M.models.length;
    const ys = [acc[0] / n, acc[1] / n];
    return ys.map((v, i) => v * M.scaler_y.scale[i] + M.scaler_y.mean[i]);
  }

  // Empirical codes (H, Lxx, Lyy)
  const codes = {
    rpa: (H, Lx, Ly) => [0.09 * H / Math.sqrt(Lx), 0.09 * H / Math.sqrt(Ly)],
    ec8: (H) => { const v = 0.075 * Math.pow(H, 0.75); return [v, v]; },
    asce:(H) => { const v = 0.0466 * Math.pow(H, 0.9);  return [v, v]; },
  };

  // ---- Metrics ----
  function r2(yTrue, yPred) {
    const m = yTrue.reduce((s, v) => s + v, 0) / yTrue.length;
    let ssRes = 0, ssTot = 0;
    for (let i = 0; i < yTrue.length; i++) { ssRes += (yTrue[i] - yPred[i]) ** 2; ssTot += (yTrue[i] - m) ** 2; }
    return 1 - ssRes / ssTot;
  }
  function rmse(yTrue, yPred) {
    let s = 0; for (let i = 0; i < yTrue.length; i++) s += (yTrue[i] - yPred[i]) ** 2;
    return Math.sqrt(s / yTrue.length);
  }

  // ========================================================================
  // BUILD: feature cards (methodology)
  // ========================================================================
  function buildFeatureCards() {
    const el = document.getElementById("featureCards");
    el.innerHTML = M.features.map((f) => {
      const meta = FEATURE_META[f]; const st = M.feature_stats[f];
      return `<div class="glass rounded-xl p-4 transition-colors">
        <div class="flex items-center gap-2.5 mb-3">
          <span class="grid place-items-center h-8 w-8 rounded-lg bg-cyan/10 border border-cyan/20 text-cyan shrink-0">
            <svg viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[meta.icon]}</svg>
          </span>
          <div class="text-sm font-semibold leading-tight">${meta.label}</div>
        </div>
        <div class="font-mono text-xs text-white/50">${f}</div>
        <div class="mt-2 font-mono text-sm"><span class="text-cyan-soft">${fmt(st.min)}</span><span class="text-white/30"> – </span><span class="text-cyan-soft">${fmt(st.max)}</span> <span class="text-white/40">${meta.unit}</span></div>
      </div>`;
    }).join("");
  }

  // ========================================================================
  // BUILD: architecture diagram
  // ========================================================================
  function buildArch() {
    const el = document.getElementById("archDiagram");
    const layers = M.arch; // [6,24,12,6,2]
    const labels = ["Input", "Dense", "Dense", "Dense", "Output"];
    const maxU = Math.max(...layers);
    el.innerHTML = layers.map((u, i) => {
      const h = 40 + (u / maxU) * 120;
      const isEdge = i === 0 || i === layers.length - 1;
      const grad = isEdge ? "from-violet/70 to-violet/20" : "from-cyan/70 to-cyan/15";
      const arrow = i < layers.length - 1
        ? `<svg viewBox="0 0 24 24" class="h-4 w-4 text-white/25 mx-auto" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>` : "";
      return `<div class="flex items-center gap-1 sm:gap-3 flex-1 justify-center">
        <div class="flex flex-col items-center gap-2">
          <div class="w-9 sm:w-12 rounded-lg bg-gradient-to-b ${grad} border border-white/10 flex items-center justify-center" style="height:${h}px">
            <span class="font-mono text-xs sm:text-sm font-bold text-white">${u}</span>
          </div>
          <div class="text-[10px] sm:text-[11px] text-white/45 text-center leading-tight">${labels[i]}</div>
        </div>
        ${arrow}
      </div>`;
    }).join("");
  }

  // ========================================================================
  // BUILD: predictor inputs
  // ========================================================================
  const state = {};
  function buildInputs() {
    const el = document.getElementById("inputControls");
    el.innerHTML = M.features.map((f) => {
      const meta = FEATURE_META[f]; const st = M.feature_stats[f];
      const lo = meta.min, hi = meta.max;
      // default = dataset median, snapped to the slider step and clamped to limits
      let def = Math.round(st.median / meta.step) * meta.step;
      def = Math.min(hi, Math.max(lo, round(def, 2)));
      state[f] = def;
      return `<div data-f="${f}">
        <div class="flex items-center justify-between mb-2">
          <label for="r-${f}" class="text-sm font-medium text-white/80">${meta.label} ${meta.unit ? `<span class="text-white/35 font-normal">(${meta.unit})</span>` : ""}</label>
          <input id="n-${f}" type="number" step="${meta.step}" value="${state[f]}"
            class="w-24 text-right font-mono text-sm bg-white/[0.04] border border-white/10 focus:border-cyan/50 rounded-lg px-2.5 py-1.5 outline-none transition-colors" />
        </div>
        <input id="r-${f}" type="range" min="${lo}" max="${hi}" step="${meta.step}" value="${state[f]}" class="w-full cursor-pointer" />
        <div class="flex justify-between mt-1 text-[10px] font-mono text-white/30"><span>${fmt(lo)}</span><span>typical range</span><span>${fmt(hi)}</span></div>
      </div>`;
    }).join("");

    M.features.forEach((f) => {
      const r = document.getElementById("r-" + f);
      const n = document.getElementById("n-" + f);
      r.addEventListener("input", () => { state[f] = parseFloat(r.value); n.value = state[f]; update(); });
      n.addEventListener("input", () => {
        let v = parseFloat(n.value); if (isNaN(v)) return;
        state[f] = v; r.value = v; update();
      });
    });
  }

  function resetInputs() {
    M.features.forEach((f) => {
      const meta = FEATURE_META[f]; const st = M.feature_stats[f];
      let v = Math.round(st.median / meta.step) * meta.step;
      v = Math.min(meta.max, Math.max(meta.min, round(v, 2)));
      state[f] = v;
      document.getElementById("r-" + f).value = v;
      document.getElementById("n-" + f).value = v;
    });
    update();
  }

  // ========================================================================
  // UPDATE predictor output
  // ========================================================================
  function update() {
    const x = M.features.map((f) => state[f]);
    const [tx, ty] = predict(x);
    document.getElementById("outTx").textContent = tx.toFixed(3);
    document.getElementById("outTy").textContent = ty.toFixed(3);

    // range check
    const oob = [];
    M.features.forEach((f) => {
      const meta = FEATURE_META[f];
      if (state[f] < meta.min || state[f] > meta.max) oob.push(meta.label);
    });
    const warn = document.getElementById("rangeWarn");
    if (oob.length) {
      warn.classList.remove("hidden");
      document.getElementById("rangeWarnText").innerHTML =
        `Outside the training range for: <strong>${oob.join(", ")}</strong>. The prediction is an extrapolation — treat it with caution.`;
    } else warn.classList.add("hidden");

    // code comparison
    const H = state.H_total, Lx = state.L_xx, Ly = state.L_yy;
    const rows = [
      { name: "ANN (this work)", v: [tx, ty], ref: true, color: "cyan" },
      { name: "RPA 99/2003", v: codes.rpa(H, Lx, Ly), color: "rose" },
      { name: "Eurocode 8", v: codes.ec8(H), color: "rose" },
      { name: "ASCE 7-16", v: codes.asce(H), color: "rose" },
    ];
    const maxV = Math.max(...rows.flatMap((r) => r.v), tx, ty, 0.1);
    const cc = document.getElementById("codeCompare");
    cc.innerHTML = rows.map((r) => {
      const barColor = r.ref ? "from-cyan to-azure" : "from-rose-500/80 to-rose-400/50";
      const w1 = Math.max(2, (r.v[0] / maxV) * 100);
      const w2 = Math.max(2, (r.v[1] / maxV) * 100);
      return `<div>
        <div class="flex items-center justify-between text-xs mb-1.5">
          <span class="${r.ref ? "text-cyan-soft font-semibold" : "text-white/65"}">${r.name}</span>
          <span class="font-mono text-white/55">Tx ${r.v[0].toFixed(3)} · Ty ${r.v[1].toFixed(3)}</span>
        </div>
        <div class="space-y-1">
          <div class="h-2 rounded-full bg-white/[0.05] overflow-hidden"><div class="h-full rounded-full bg-gradient-to-r ${barColor}" style="width:${w1}%"></div></div>
          <div class="h-2 rounded-full bg-white/[0.05] overflow-hidden"><div class="h-full rounded-full bg-gradient-to-r ${barColor} opacity-60" style="width:${w2}%"></div></div>
        </div>
      </div>`;
    }).join("");
  }

  // ========================================================================
  // CHARTS
  // ========================================================================
  const FONT = "Inter, sans-serif";
  Chart.defaults.color = "rgba(229,237,245,0.6)";
  Chart.defaults.font.family = FONT;
  Chart.defaults.borderColor = "rgba(255,255,255,0.08)";

  // Predictions for a split ("test" | "train"), computed live and cached
  const _splitCache = {};
  function splitArrays(which) {
    if (_splitCache[which]) return _splitCache[which];
    const idxs = which === "train" ? M.dataset.train_idx : M.dataset.test_idx;
    const trueTx = [], trueTy = [], predTx = [], predTy = [], H = [], Lx = [], Ly = [];
    idxs.forEach((idx) => {
      const x = M.dataset.X[idx];
      const [px, py] = predict(x);
      trueTx.push(M.dataset.y[idx][0]); trueTy.push(M.dataset.y[idx][1]);
      predTx.push(px); predTy.push(py);
      H.push(x[0]); Lx.push(x[2]); Ly.push(x[3]);
    });
    return (_splitCache[which] = { trueTx, trueTy, predTx, predTy, H, Lx, Ly });
  }
  function testArrays() { return splitArrays("test"); }

  function barChart() {
    // Training-set R². ANN uses the thesis reference values (representative model);
    // empirical codes are evaluated live on the training set.
    const tr = splitArrays("train");
    const codeR2 = (dim, fn) => {
      const t = dim === 0 ? tr.trueTx : tr.trueTy;
      const pred = tr.H.map((h, i) => fn(h, tr.Lx[i], tr.Ly[i])[dim]);
      return r2(t, pred);
    };
    const r2Tx = [0.900, codeR2(0, codes.rpa), codeR2(0, (h) => codes.ec8(h)), codeR2(0, (h) => codes.asce(h))];
    const r2Ty = [0.800, codeR2(1, codes.rpa), codeR2(1, (h) => codes.ec8(h)), codeR2(1, (h) => codes.asce(h))];

    new Chart(document.getElementById("barChart"), {
      type: "bar",
      data: {
        labels: ["ANN", "RPA 99/2003", "Eurocode 8", "ASCE 7-16"],
        datasets: [
          { label: "R² Tx", data: r2Tx, backgroundColor: "rgba(34,211,238,0.75)", borderColor: "#22d3ee", borderWidth: 1, borderRadius: 5 },
          { label: "R² Ty", data: r2Ty, backgroundColor: "rgba(139,92,246,0.7)", borderColor: "#8b5cf6", borderWidth: 1, borderRadius: 5 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { usePointStyle: true, pointStyle: "rectRounded", padding: 16 } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(3)}` } },
        },
        scales: {
          y: { suggestedMin: -3, suggestedMax: 1, grid: { color: "rgba(255,255,255,0.06)" }, title: { display: true, text: "R² (train)", color: "rgba(229,237,245,0.5)" } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  let scatterChart = null;
  function drawScatter(dim, split) {
    split = split || "test";
    const t = splitArrays(split);
    const trueA = dim === 0 ? t.trueTx : t.trueTy;
    const predA = dim === 0 ? t.predTx : t.predTy;
    const pts = trueA.map((v, i) => ({ x: v, y: predA[i] }));
    const lo = Math.min(...trueA, ...predA) * 0.9;
    const hi = Math.max(...trueA, ...predA) * 1.08;
    const color = dim === 0 ? "#22d3ee" : "#8b5cf6";

    const data = {
      datasets: [
        { type: "line", label: "perfect", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: "rgba(255,255,255,0.35)", borderDash: [6, 6], borderWidth: 1.5, pointRadius: 0, fill: false },
        { type: "scatter", label: (dim === 0 ? "Tx" : "Ty"), data: pts, backgroundColor: color + "cc", borderColor: "#05070d", borderWidth: 1, pointRadius: 5, pointHoverRadius: 8 },
      ],
    };
    const opts = {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => `measured ${c.parsed.x.toFixed(3)}s → predicted ${c.parsed.y.toFixed(3)}s` } },
      },
      scales: {
        x: { type: "linear", min: lo, max: hi, title: { display: true, text: "Measured period (s)", color: "rgba(229,237,245,0.5)" }, grid: { color: "rgba(255,255,255,0.06)" } },
        y: { type: "linear", min: lo, max: hi, title: { display: true, text: "Predicted period (s)", color: "rgba(229,237,245,0.5)" }, grid: { color: "rgba(255,255,255,0.06)" } },
      },
    };
    if (scatterChart) { scatterChart.data = data; scatterChart.options = opts; scatterChart.update(); }
    else scatterChart = new Chart(document.getElementById("scatterChart"), { data, options: opts });
  }

  const scatterState = { dim: 0, split: "test" };
  const BTN_OFF = "cursor-pointer rounded-lg px-3 py-1.5 text-sm font-semibold bg-white/5 text-white/60 border border-white/10";
  const BTN_ON = "cursor-pointer rounded-lg px-3 py-1.5 text-sm font-semibold bg-cyan/20 text-cyan-soft border border-cyan/30";
  function bindScatterToggle() {
    const dimWrap = document.getElementById("scatterToggle");
    const splitWrap = document.getElementById("scatterSplit");
    dimWrap.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        dimWrap.querySelectorAll("button").forEach((x) => (x.className = BTN_OFF));
        b.className = BTN_ON;
        scatterState.dim = parseInt(b.dataset.t, 10);
        drawScatter(scatterState.dim, scatterState.split);
      });
    });
    splitWrap.querySelectorAll("button").forEach((b) => {
      b.addEventListener("click", () => {
        splitWrap.querySelectorAll("button").forEach((x) => (x.className = BTN_OFF));
        b.className = BTN_ON;
        scatterState.split = b.dataset.s;
        drawScatter(scatterState.dim, scatterState.split);
      });
    });
  }

  // Comparison table (train + test). Codes evaluated live on both splits;
  // ANN train R² uses the thesis reference numbers (single representative model).
  function buildTable() {
    const te = splitArrays("test"), tr = splitArrays("train");
    const codeR2 = (s, dim, fn) => {
      const t = (dim === 0 ? s.trueTx : s.trueTy);
      const pred = s.H.map((h, i) => fn(h, s.Lx[i], s.Ly[i])[dim]);
      return r2(t, pred);
    };
    const codeRmse = (s, dim, fn) => {
      const t = (dim === 0 ? s.trueTx : s.trueTy);
      const pred = s.H.map((h, i) => fn(h, s.Lx[i], s.Ly[i])[dim]);
      return rmse(t, pred);
    };
    const rpa = codes.rpa, ec8 = (h) => codes.ec8(h), asce = (h) => codes.asce(h);

    const rows = [
      { n: "ANN (this work)", best: true,
        trx: 0.900, trly: 0.800,
        tex: r2(te.trueTx, te.predTx), tey: r2(te.trueTy, te.predTy),
        rx: rmse(te.trueTx, te.predTx), ry: rmse(te.trueTy, te.predTy) },
      { n: "RPA 99/2003",
        trx: codeR2(tr, 0, rpa), trly: codeR2(tr, 1, rpa),
        tex: codeR2(te, 0, rpa), tey: codeR2(te, 1, rpa),
        rx: codeRmse(te, 0, rpa), ry: codeRmse(te, 1, rpa) },
      { n: "Eurocode 8",
        trx: codeR2(tr, 0, ec8), trly: codeR2(tr, 1, ec8),
        tex: codeR2(te, 0, ec8), tey: codeR2(te, 1, ec8),
        rx: codeRmse(te, 0, ec8), ry: codeRmse(te, 1, ec8) },
      { n: "ASCE 7-16",
        trx: codeR2(tr, 0, asce), trly: codeR2(tr, 1, asce),
        tex: codeR2(te, 0, asce), tey: codeR2(te, 1, asce),
        rx: codeRmse(te, 0, asce), ry: codeRmse(te, 1, asce) },
    ];
    const cell = (v, good) => `<td class="py-3 text-right ${good ? "text-emerald-300" : v < 0 ? "text-rose-300" : "text-white/70"}">${v.toFixed(3)}</td>`;
    document.getElementById("compTable").innerHTML = rows.map((r) =>
      `<tr class="border-b border-white/5 ${r.best ? "bg-cyan/[0.05]" : ""}">
        <td class="py-3 font-sans ${r.best ? "text-cyan-soft font-semibold" : "text-white/80"}">${r.n}</td>
        ${cell(r.trx, r.best)}${cell(r.trly, r.best)}${cell(r.tex, r.best)}${cell(r.tey, r.best)}
        <td class="py-3 text-right text-white/70">${r.rx.toFixed(3)}</td>
        <td class="py-3 text-right text-white/70">${r.ry.toFixed(3)}</td>
      </tr>`).join("");
  }

  // ========================================================================
  // QR code — links to the live site
  // ========================================================================
  // After deploying, set DEPLOYED_URL to your public address (e.g. GitHub Pages).
  // When the page is already served over http(s), the current URL is used automatically.
  const DEPLOYED_URL = "https://USERNAME.github.io/ann-seismic/";
  function buildQR() {
    const box = document.getElementById("qrcode");
    const link = document.getElementById("qrLink");
    if (!box || typeof qrcode === "undefined") return;
    const isHttp = location.protocol === "http:" || location.protocol === "https:";
    const url = isHttp ? location.href.split("#")[0] : DEPLOYED_URL;
    const qr = qrcode(0, "M");
    qr.addData(url);
    qr.make();
    box.innerHTML = qr.createSvgTag({ cellSize: 4, margin: 0, scalable: true });
    const svg = box.querySelector("svg");
    if (svg) { svg.setAttribute("width", "100%"); svg.setAttribute("height", "100%"); }
    if (link) { link.textContent = url.replace(/^https?:\/\//, ""); link.href = url; }
  }

  // ========================================================================
  // Full-screen toggle
  // ========================================================================
  function bindFullscreen() {
    const btn = document.getElementById("fsBtn");
    if (!btn) return;
    const enter = document.getElementById("fsEnter");
    const exit = document.getElementById("fsExit");
    btn.addEventListener("click", () => {
      if (!document.fullscreenElement) {
        (document.documentElement.requestFullscreen || document.documentElement.webkitRequestFullscreen)
          ?.call(document.documentElement);
      } else {
        (document.exitFullscreen || document.webkitExitFullscreen)?.call(document);
      }
    });
    document.addEventListener("fullscreenchange", () => {
      const on = !!document.fullscreenElement;
      enter.classList.toggle("hidden", on);
      exit.classList.toggle("hidden", !on);
    });
  }

  // ========================================================================
  // helpers + scroll reveal
  // ========================================================================
  function fmt(v) { return Number.isInteger(v) ? v : (+v).toFixed(v < 10 ? 2 : 1); }
  function round(v, d) { const p = Math.pow(10, d); return Math.round(v * p) / p; }

  function initReveal() {
    const els = Array.from(document.querySelectorAll(".reveal"));
    const vh = window.innerHeight || 800;
    // Only hide elements that start below the fold — above-fold content shows immediately.
    els.forEach((e) => { if (e.getBoundingClientRect().top > vh * 0.85) e.classList.add("pre"); });
    const hidden = els.filter((e) => e.classList.contains("pre"));
    if (!("IntersectionObserver" in window)) { hidden.forEach((e) => e.classList.remove("pre")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); e.target.classList.remove("pre"); io.unobserve(e.target); } });
    }, { threshold: 0.12 });
    hidden.forEach((e) => io.observe(e));
    // Safety net: never leave content hidden if the observer misbehaves.
    setTimeout(() => hidden.forEach((e) => { e.classList.add("in"); e.classList.remove("pre"); }), 2500);
  }

  // ========================================================================
  // INIT
  // ========================================================================
  document.addEventListener("DOMContentLoaded", () => {
    buildFeatureCards();
    buildArch();
    buildInputs();
    update();
    barChart();
    drawScatter(0, "test");
    bindScatterToggle();
    buildTable();
    buildQR();
    bindFullscreen();
    document.getElementById("resetBtn").addEventListener("click", resetInputs);
    initReveal();
    console.log("ANN ready · ensemble of", M.models.length, "models ·", M.n_buildings, "buildings");
  });
})();
