/* ============================================================================
   ANN Fundamental Period — front-end logic
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
    return a;
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

  // ---- Empirical codes (H only — mixed/voiles defaults, most common in Algeria) ----
  const codes = {
    rpa:  (H) => { const v = 0.050 * Math.pow(H, 0.75); return [v, v]; },
    ec8:  (H) => { const v = 0.075 * Math.pow(H, 0.75); return [v, v]; },
    asce: (H) => { const v = 0.0466 * Math.pow(H, 0.90); return [v, v]; },
    bsl:  (H) => { const v = 0.020 * H;                  return [v, v]; },
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
  // THEME (light / dark)
  // ========================================================================
  let _theme = "dark";

  function getChartColors() {
    const isLight = _theme === "light";
    return {
      text:   isLight ? "rgba(15,23,42,0.65)"  : "rgba(229,237,245,0.65)",
      grid:   isLight ? "rgba(15,23,42,0.08)"  : "rgba(255,255,255,0.08)",
      perfect: isLight ? "rgba(15,23,42,0.28)" : "rgba(255,255,255,0.35)",
      ptBorder: isLight ? "#cbd5e1" : "#05070d",
    };
  }

  function _updateThemeBtn() {
    const sun  = document.getElementById("themeIconSun");
    const moon = document.getElementById("themeIconMoon");
    if (sun)  sun.classList.toggle("hidden",  _theme === "light");
    if (moon) moon.classList.toggle("hidden", _theme === "dark");
  }

  function applyTheme(theme) {
    _theme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    const cc = getChartColors();
    Chart.defaults.color       = cc.text;
    Chart.defaults.borderColor = cc.grid;
    _updateThemeBtn();
    // Redraw charts with updated palette
    barChart();
    drawScatter(scatterState.dim, scatterState.split);
  }

  function initTheme() {
    const saved = localStorage.getItem("theme") || "dark";
    _theme = saved;
    document.documentElement.setAttribute("data-theme", saved);
    const cc = getChartColors();
    Chart.defaults.color       = cc.text;
    Chart.defaults.borderColor = cc.grid;
    _updateThemeBtn();
    document.getElementById("themeBtn")?.addEventListener("click", () =>
      applyTheme(_theme === "dark" ? "light" : "dark"));
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
    const layers = M.arch;
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
  // BUILD: predictor inputs (no real-time prediction — needs Calculate click)
  // ========================================================================
  const state = {};
  function buildInputs() {
    const el = document.getElementById("inputControls");
    el.innerHTML = M.features.map((f) => {
      const meta = FEATURE_META[f]; const st = M.feature_stats[f];
      const lo = meta.min, hi = meta.max;
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
      // Sync slider ↔ number input only, no auto-calculate
      r.addEventListener("input", () => { state[f] = parseFloat(r.value); n.value = state[f]; });
      n.addEventListener("input", () => {
        let v = parseFloat(n.value); if (isNaN(v)) return;
        state[f] = v; r.value = v;
      });
    });
  }

  // ---- Reset to medians (also triggers calculation) ----
  function resetInputs() {
    M.features.forEach((f) => {
      const meta = FEATURE_META[f]; const st = M.feature_stats[f];
      let v = Math.round(st.median / meta.step) * meta.step;
      v = Math.min(meta.max, Math.max(meta.min, round(v, 2)));
      state[f] = v;
      document.getElementById("r-" + f).value = v;
      document.getElementById("n-" + f).value = v;
    });
    runCalculation();
  }

  // ---- Calculate button with loading animation ----
  function runCalculation() {
    const btn     = document.getElementById("calcBtn");
    const label   = document.getElementById("calcLabel");
    const icon    = document.getElementById("calcIcon");
    const spinner = document.getElementById("calcSpinner");

    if (!btn) { update(); return; }

    btn.disabled = true;
    if (label)   label.textContent = "Computing…";
    if (icon)    icon.classList.add("hidden");
    if (spinner) spinner.classList.remove("hidden");

    setTimeout(() => {
      update();
      btn.disabled = false;
      if (label)   label.textContent = "Calculate Periods";
      if (icon)    icon.classList.remove("hidden");
      if (spinner) spinner.classList.add("hidden");
    }, 1000);
  }

  // ========================================================================
  // UPDATE predictor output
  // ========================================================================
  function update() {
    const x = M.features.map((f) => state[f]);
    const [tx, ty] = predict(x);
    document.getElementById("outTx").textContent = tx.toFixed(3);
    document.getElementById("outTy").textContent = ty.toFixed(3);

    // Range check
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

    // Empirical codes comparison (H only — no Lx/Ly)
    const H = state.H_total;
    const rows = [
      { name: "ANN (this work)", v: [tx, ty], ref: true, color: "cyan" },
      { name: "RPA 99/2003",   v: codes.rpa(H),  color: "rose" },
      { name: "Eurocode 8",    v: codes.ec8(H),  color: "rose" },
      { name: "ASCE 7-16",     v: codes.asce(H), color: "rose" },
    ];
    const maxV = Math.max(...rows.flatMap((r) => r.v), 0.1);
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
  Chart.defaults.font.family = "Inter, sans-serif";

  // Predictions for a split ("test" | "train" | "all"), computed live and cached
  const _splitCache = {};
  function splitArrays(which) {
    if (_splitCache[which]) return _splitCache[which];
    let idxs;
    if (which === "all") {
      idxs = [...M.dataset.train_idx, ...M.dataset.test_idx].sort((a, b) => a - b);
    } else {
      idxs = which === "train" ? M.dataset.train_idx : M.dataset.test_idx;
    }
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

  // ---- Bar chart: R² comparison (ANN=train, codes=full dataset) ----
  let barChartInst = null;
  function barChart() {
    if (barChartInst) { barChartInst.destroy(); barChartInst = null; }
    const el = document.getElementById("barChart");
    if (!el) return;

    const all = splitArrays("all");
    const codeR2 = (dim, fn) => {
      const t = dim === 0 ? all.trueTx : all.trueTy;
      const pred = all.H.map((h) => fn(h)[dim]);
      return r2(t, pred);
    };

    // ANN: thesis reference values. LR: V7 analysis values.
    const r2Tx = [0.900, codeR2(0, codes.rpa), codeR2(0, codes.ec8), codeR2(0, codes.asce), 0.342];
    const r2Ty = [0.800, codeR2(1, codes.rpa), codeR2(1, codes.ec8), codeR2(1, codes.asce), 0.216];
    const cc = getChartColors();

    const colorsTx = [
      "rgba(34,211,238,0.85)",   // ANN — cyan
      "rgba(244,114,182,0.72)",  // RPA — rose
      "rgba(244,114,182,0.72)",  // EC8
      "rgba(244,114,182,0.72)",  // ASCE
      "rgba(52,211,153,0.85)",   // LR — emerald
    ];
    const colorsTy = [
      "rgba(139,92,246,0.82)",
      "rgba(244,114,182,0.45)",
      "rgba(244,114,182,0.45)",
      "rgba(244,114,182,0.45)",
      "rgba(52,211,153,0.52)",
    ];
    const borderTx = ["#22d3ee","#f472b6","#f472b6","#f472b6","#34d399"];
    const borderTy = ["#8b5cf6","#f472b6","#f472b6","#f472b6","#34d399"];

    const mob = window.innerWidth < 640;
    const fs  = mob ? 10 : 12;
    const labels = mob
      ? ["ANN", "RPA", "EC8", "ASCE", "Lin.Reg"]
      : ["ANN\n(this work)", "RPA 99/2003", "Eurocode 8", "ASCE 7-16", ["Linear", "Regression"]];

    barChartInst = new Chart(el, {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "R² Tx", data: r2Tx, backgroundColor: colorsTx, borderColor: borderTx, borderWidth: 1.5, borderRadius: mob ? 4 : 7, barPercentage: 0.8 },
          { label: "R² Ty", data: r2Ty, backgroundColor: colorsTy, borderColor: borderTy, borderWidth: 1.5, borderRadius: mob ? 4 : 7, barPercentage: 0.8 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { usePointStyle: true, pointStyle: "rectRounded", padding: mob ? 10 : 20, color: cc.text, font: { size: fs + 1 } },
          },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(3)}` } },
        },
        scales: {
          y: {
            suggestedMin: -0.6,
            suggestedMax: 1.05,
            grid: { color: cc.grid },
            ticks: { color: cc.text, font: { size: fs } },
            title: { display: !mob, text: "R²", color: cc.text, font: { size: fs + 1 } },
          },
          x: {
            grid: { display: false },
            ticks: { color: cc.text, maxRotation: mob ? 35 : 0, minRotation: 0, font: { size: fs } },
          },
        },
      },
    });
  }

  // ---- Scatter chart ----
  let scatterChart = null;
  const scatterState = { dim: 0, split: "test" };

  function drawScatter(dim, split) {
    split = split || "test";
    if (scatterChart) { scatterChart.destroy(); scatterChart = null; }
    const el = document.getElementById("scatterChart");
    if (!el) return;

    const t = splitArrays(split);
    const trueA = dim === 0 ? t.trueTx : t.trueTy;
    const predA = dim === 0 ? t.predTx : t.predTy;
    const pts = trueA.map((v, i) => ({ x: v, y: predA[i] }));
    const lo = Math.min(...trueA, ...predA) * 0.9;
    const hi = Math.max(...trueA, ...predA) * 1.08;
    const color = dim === 0 ? "#22d3ee" : "#8b5cf6";
    const cc = getChartColors();

    const mob = window.innerWidth < 640;
    const fs  = mob ? 10 : 12;

    scatterChart = new Chart(el, {
      data: {
        datasets: [
          {
            type: "line", label: "perfect",
            data: [{ x: lo, y: lo }, { x: hi, y: hi }],
            borderColor: cc.perfect, borderDash: [6, 6], borderWidth: 1.5, pointRadius: 0, fill: false,
          },
          {
            type: "scatter", label: (dim === 0 ? "Tx" : "Ty"),
            data: pts,
            backgroundColor: color + "cc", borderColor: cc.ptBorder, borderWidth: 1,
            pointRadius: mob ? 3 : 5, pointHoverRadius: mob ? 5 : 8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => `measured ${c.parsed.x.toFixed(3)}s → predicted ${c.parsed.y.toFixed(3)}s` } },
        },
        scales: {
          x: {
            type: "linear", min: lo, max: hi,
            title: { display: !mob, text: "Measured period (s)", color: cc.text, font: { size: fs } },
            grid: { color: cc.grid },
            ticks: { color: cc.text, font: { size: fs }, maxTicksLimit: mob ? 5 : 8 },
          },
          y: {
            type: "linear", min: lo, max: hi,
            title: { display: !mob, text: "Predicted period (s)", color: cc.text, font: { size: fs } },
            grid: { color: cc.grid },
            ticks: { color: cc.text, font: { size: fs }, maxTicksLimit: mob ? 5 : 8 },
          },
        },
      },
    });
  }

  const BTN_OFF = "cursor-pointer rounded-lg px-3 py-1.5 text-sm font-semibold bg-white/5 text-white/60 border border-white/10";
  const BTN_ON  = "cursor-pointer rounded-lg px-3 py-1.5 text-sm font-semibold bg-cyan/20 text-cyan-soft border border-cyan/30";
  function bindScatterToggle() {
    const dimWrap   = document.getElementById("scatterToggle");
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

  // ---- Comparison table ----
  // ANN: R²-Train | R²-Test | RMSE-Test
  // Codes: — (no training) | R²-full | RMSE-full
  function buildTable() {
    const te  = splitArrays("test");
    const all = splitArrays("all");
    const evR2   = (s, dim, fn) => { const t = dim===0?s.trueTx:s.trueTy; return r2(t, s.H.map(h=>fn(h)[dim])); };
    const evRmse = (s, dim, fn) => { const t = dim===0?s.trueTx:s.trueTy; return rmse(t, s.H.map(h=>fn(h)[dim])); };
    const rpa = codes.rpa, ec8 = codes.ec8, asce = codes.asce;

    const rows = [
      { n: "ANN (this work)", best: true,
        trx: 0.900,  trly: 0.800,
        r2x: r2(te.trueTx, te.predTx),   r2y: r2(te.trueTy, te.predTy),
        rx:  rmse(te.trueTx, te.predTx), ry:  rmse(te.trueTy, te.predTy) },
      { n: "RPA 99/2003", best: false,
        trx: null, trly: null,
        r2x: evR2(all,0,rpa),   r2y: evR2(all,1,rpa),
        rx:  evRmse(all,0,rpa), ry:  evRmse(all,1,rpa) },
      { n: "Eurocode 8", best: false,
        trx: null, trly: null,
        r2x: evR2(all,0,ec8),   r2y: evR2(all,1,ec8),
        rx:  evRmse(all,0,ec8), ry:  evRmse(all,1,ec8) },
      { n: "ASCE 7-16", best: false,
        trx: null, trly: null,
        r2x: evR2(all,0,asce),   r2y: evR2(all,1,asce),
        rx:  evRmse(all,0,asce), ry:  evRmse(all,1,asce) },
    ];

    const cellNum = (v, good) =>
      `<td class="py-3 text-right ${good ? "text-emerald-300" : v < 0 ? "text-rose-300" : "text-white/70"}">${v.toFixed(3)}</td>`;
    const cellNA = () =>
      `<td class="py-3 text-right text-white/25 font-sans text-xs not-italic">—</td>`;

    document.getElementById("compTable").innerHTML = rows.map((r) =>
      `<tr class="border-b border-white/5 ${r.best ? "bg-cyan/[0.05]" : ""}">
        <td class="py-3 font-sans ${r.best ? "text-cyan-soft font-semibold" : "text-white/80"}">${r.n}</td>
        ${r.trx !== null ? cellNum(r.trx, r.best) : cellNA()}
        ${r.trly !== null ? cellNum(r.trly, r.best) : cellNA()}
        ${cellNum(r.r2x, r.best)}${cellNum(r.r2y, r.best)}
        <td class="py-3 text-right text-white/70">${r.rx.toFixed(3)}</td>
        <td class="py-3 text-right text-white/70">${r.ry.toFixed(3)}</td>
      </tr>`).join("");
  }

  // ========================================================================
  // QR code — links to the live site + click-to-zoom modal
  // ========================================================================
  const DEPLOYED_URL = "https://kianokiano72.github.io/ann-seismic/";
  let _qrUrl = "";

  function buildQR() {
    const box  = document.getElementById("qrcode");
    const link = document.getElementById("qrLink");
    if (!box || typeof qrcode === "undefined") return;
    const isHttp = location.protocol === "http:" || location.protocol === "https:";
    _qrUrl = isHttp ? location.href.split("#")[0] : DEPLOYED_URL;

    const qr = qrcode(0, "M");
    qr.addData(_qrUrl);
    qr.make();
    box.innerHTML = qr.createSvgTag({ cellSize: 4, margin: 0, scalable: true });
    const svg = box.querySelector("svg");
    if (svg) { svg.setAttribute("width", "100%"); svg.setAttribute("height", "100%"); }
    if (link) { link.textContent = _qrUrl.replace(/^https?:\/\//, ""); link.href = _qrUrl; }

    // Click-to-zoom (attach to the whole white card wrapper)
    const qrWrap = box.closest(".qr-wrap") || box;
    qrWrap.addEventListener("click", openQRModal);

    // Close modal on backdrop click or Escape
    const modal = document.getElementById("qrModal");
    if (modal) {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) closeQRModal();
      });
    }
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeQRModal();
    });
  }

  function openQRModal() {
    const modal  = document.getElementById("qrModal");
    const canvas = document.getElementById("qrModalCanvas");
    if (!modal || !canvas || !_qrUrl) return;
    const qr2 = qrcode(0, "M");
    qr2.addData(_qrUrl);
    qr2.make();
    // Use PNG data URL — renders pixel-perfect, no SVG alignment issues
    const dataUrl = qr2.createDataURL(6, 2);
    canvas.innerHTML = `<img src="${dataUrl}" alt="QR code" style="width:240px;height:240px;display:block;image-rendering:pixelated;" />`;
    modal.classList.remove("hidden");
    modal.classList.add("flex");
  }

  function closeQRModal() {
    const modal = document.getElementById("qrModal");
    if (modal) { modal.classList.add("hidden"); modal.classList.remove("flex"); }
  }

  // ========================================================================
  // Full-screen toggle
  // ========================================================================
  function bindFullscreen() {
    const btn   = document.getElementById("fsBtn");
    if (!btn) return;
    const enter = document.getElementById("fsEnter");
    const exit  = document.getElementById("fsExit");

    const isFs = () => !!(document.fullscreenElement || document.webkitFullscreenElement);
    const sync = () => {
      const on = isFs();
      enter?.classList.toggle("hidden", on);
      exit?.classList.toggle("hidden", !on);
    };

    document.addEventListener("fullscreenchange",       sync);
    document.addEventListener("webkitfullscreenchange", sync);

    btn.addEventListener("click", () => {
      if (isFs()) {
        (document.exitFullscreen || document.webkitExitFullscreen)?.call(document);
      } else {
        const el = document.documentElement;
        const fn = el.requestFullscreen || el.webkitRequestFullscreen;
        fn?.call(el).catch?.(() => {});
      }
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
    els.forEach((e) => { if (e.getBoundingClientRect().top > vh * 0.85) e.classList.add("pre"); });
    const hidden = els.filter((e) => e.classList.contains("pre"));
    if (!("IntersectionObserver" in window)) { hidden.forEach((e) => e.classList.remove("pre")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); e.target.classList.remove("pre"); io.unobserve(e.target); } });
    }, { threshold: 0.12 });
    hidden.forEach((e) => io.observe(e));
    setTimeout(() => hidden.forEach((e) => { e.classList.add("in"); e.classList.remove("pre"); }), 2500);
  }

  // ========================================================================
  // INIT
  // ========================================================================
  document.addEventListener("DOMContentLoaded", () => {
    initTheme();         // must be first — sets Chart.defaults before any chart is drawn
    buildFeatureCards();
    buildArch();
    buildInputs();
    // NOTE: no update() on load — outputs show "—" until user clicks Calculate
    barChart();
    drawScatter(0, "test");
    bindScatterToggle();
    buildTable();
    buildQR();
    bindFullscreen();
    document.getElementById("resetBtn")?.addEventListener("click", resetInputs);
    document.getElementById("calcBtn")?.addEventListener("click", runCalculation);
    initReveal();
    console.log("ANN ready · ensemble of", M.models.length, "models ·", M.n_buildings, "buildings");
  });
})();
