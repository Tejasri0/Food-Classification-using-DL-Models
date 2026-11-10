(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Element references
  // ------------------------------------------------------------------
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const previewImg = document.getElementById("previewImg");
  const uploaderIcon = document.getElementById("uploaderIcon");

  const modelSelect = document.getElementById("modelSelect");
  const predictBtn = document.getElementById("predictBtn");
  const statusLine = document.getElementById("statusLine");

  const classColA = document.getElementById("classColA");
  const classColB = document.getElementById("classColB");

  const classPickerZone = document.getElementById("classPickerZone");
  const classPickerSummary = document.getElementById("classPickerSummary");
  const classPickerSummaryValue = document.getElementById("classPickerSummaryValue");
  const classPickerChangeBtn = document.getElementById("classPickerChangeBtn");

  const nutritionPlaceholder = document.getElementById("nutritionPlaceholder");
  const nutritionCards = document.getElementById("nutritionCards");

  const metricsPlaceholder = document.getElementById("metricsPlaceholder");
  const metricsCards = document.getElementById("metricsCards");

  const predictBanner = document.getElementById("predictBanner");
  const predictBannerBody = document.getElementById("predictBannerBody");
  const gaugeRingFill = document.getElementById("gaugeRingFill");
  const gaugeValue = document.getElementById("gaugeValue");
  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 27;

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  const state = {
    file: null,
    model: null,
    actualClass: null,
  };

  const NUTRITION_LABELS = {
    protein: "Protein",
    carbohydrates: "Carbohydrates",
    carbs: "Carbohydrates",
    fat: "Fat",
    fiber: "Fiber",
    sugar: "Sugar",
    sodium: "Sodium",
    cholesterol: "Cholesterol",
    saturated_fat: "Saturated Fat",
    serving_size: "Serving Size",
  };

  function formatNutritionLabel(key) {
    if (NUTRITION_LABELS[key]) return NUTRITION_LABELS[key];
    return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  const METRIC_LABELS = [
    ["accuracy", "Accuracy"],
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1_score", "F1 Score"],
  ];

  const CM_LABELS = {
    tp: "True Positive",
    fp: "False Positive",
    fn: "False Negative",
    tn: "True Negative",
  };

  // Data sources aren't always consistent about key casing/format
  // (f1_score vs f1-score vs F1, TP vs tp, ...). Normalize any object's
  // keys to lowercase-with-underscores so lookups below are forgiving.
  function normalizeKeys(obj) {
    if (!obj || typeof obj !== "object") return {};
    const out = {};
    Object.entries(obj).forEach(([key, value]) => {
      const normKey = key.trim().toLowerCase().replace(/[\s-]+/g, "_");
      out[normKey] = value;
    });
    return out;
  }

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------
  init();

  async function init() {
    wireUploader();
    wireModelSelect();
    wirePredict();
    wireClassPickerChange();
    await Promise.all([loadClasses(), loadModelAvailability()]);
  }

  // ------------------------------------------------------------------
  // Upload
  // ------------------------------------------------------------------
  function wireUploader() {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) {
        setFile(fileInput.files[0]);
      }
    });

    ["dragenter", "dragover"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("is-dragover");
      })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("is-dragover");
      })
    );
    dropzone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) setFile(file);
    });
  }

  function setFile(file) {
    if (!file.type.startsWith("image/")) {
      showStatus("Please choose an image file.");
      return;
    }
    state.file = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.hidden = false;
    uploaderIcon.style.display = "none";
    dropzone.classList.add("has-image");
    clearStatus();
    updatePredictAvailability();
  }

  // ------------------------------------------------------------------
  // Model selection
  // ------------------------------------------------------------------
  function wireModelSelect() {
    modelSelect.addEventListener("click", (e) => {
      const btn = e.target.closest(".segmented__option");
      if (!btn) return;
      Array.from(modelSelect.children).forEach((b) =>
        b.setAttribute("aria-checked", "false")
      );
      btn.setAttribute("aria-checked", "true");
      state.model = btn.dataset.model;
      updatePredictAvailability();
    });
  }

  async function loadModelAvailability() {
    try {
      const res = await fetch("/api/models");
      if (!res.ok) return;
      const data = await res.json();
      Array.from(modelSelect.children).forEach((btn) => {
        const info = data[btn.dataset.model];
        if (info && info.available === false) {
          const badge = document.createElement("span");
          badge.className = "badge-demo";
          badge.textContent = "Demo";
          badge.title = "Weight file not found on the server yet — predictions will be simulated.";
          btn.appendChild(badge);
        }
      });
    } catch (err) {
      // Non-fatal: model buttons still work, badges just won't appear.
    }
  }

  // ------------------------------------------------------------------
  // Class picker
  // ------------------------------------------------------------------
  async function loadClasses() {
    try {
      const res = await fetch("/api/classes");
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Could not load classes");
      renderClassGrid(data.classes || []);
    } catch (err) {
      classColA.innerHTML = `<p class="empty-metric">${escapeHtml(err.message)}</p>`;
    }
  }

  function renderClassGrid(classes) {
    const mid = Math.ceil(classes.length / 2);
    const colA = classes.slice(0, mid);
    const colB = classes.slice(mid);

    [
      [colA, classColA],
      [colB, classColB],
    ].forEach(([list, col]) => {
      col.innerHTML = "";
      list.forEach((name) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "class-chip";
        chip.setAttribute("role", "radio");
        chip.setAttribute("aria-checked", "false");
        chip.dataset.class = name;
        chip.textContent = name;
        chip.addEventListener("click", () => selectClass(name));
        col.appendChild(chip);
      });
    });
  }

  function wireClassPickerChange() {
    if (!classPickerChangeBtn) return;
    classPickerChangeBtn.addEventListener("click", () => {
      classPickerSummary.hidden = true;
      classPickerZone.hidden = false;
    });
  }

  function selectClass(name) {
    state.actualClass = name;
    document.querySelectorAll(".class-chip").forEach((chip) => {
      chip.setAttribute(
        "aria-checked",
        chip.dataset.class === name ? "true" : "false"
      );
    });
    updatePredictAvailability();
  }

  // ------------------------------------------------------------------
  // Predict
  // ------------------------------------------------------------------
  function updatePredictAvailability() {
    predictBtn.disabled = !(state.file && state.model && state.actualClass);
  }

  function wirePredict() {
    predictBtn.addEventListener("click", runPredict);
  }

  async function runPredict() {
    if (!state.file || !state.model || !state.actualClass) return;

    clearStatus();
    predictBtn.classList.add("is-loading");
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", state.file);
    formData.append("model", state.model);
    formData.append("actual_class", state.actualClass);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed.");
      renderResults(data);
    } catch (err) {
      showStatus(err.message);
    } finally {
      predictBtn.classList.remove("is-loading");
      updatePredictAvailability();
    }
  }

  // ------------------------------------------------------------------
  // Rendering: nutrition + metrics + banner
  // ------------------------------------------------------------------
  function renderResults(data) {
    renderBanner(data);
    renderNutritionCards(data);
    renderMetricsCards(data);
    collapseClassPicker(data.actual_class);
  }

  function collapseClassPicker(name) {
    if (!classPickerZone || !classPickerSummary) return;
    classPickerZone.hidden = true;
    classPickerSummaryValue.textContent = name;
    classPickerSummary.hidden = false;
  }

  function renderBanner(data) {
    const match = data.is_correct;
    predictBanner.hidden = false;
    predictBanner.classList.toggle("is-match", match);
    predictBanner.classList.toggle("is-mismatch", !match);

    // Drive the SVG confidence ring.
    const pct = Math.max(0, Math.min(100, data.confidence));
    const offset = GAUGE_CIRCUMFERENCE * (1 - pct / 100);
    gaugeRingFill.style.strokeDasharray = `${GAUGE_CIRCUMFERENCE}`;
    // Force reflow so the transition replays on repeated predictions.
    // eslint-disable-next-line no-unused-expressions
    gaugeRingFill.getBoundingClientRect();
    gaugeRingFill.style.strokeDashoffset = `${offset}`;
    gaugeRingFill.classList.toggle("is-match", match);
    gaugeRingFill.classList.toggle("is-mismatch", !match);
    gaugeValue.textContent = `${pct}%`;

    predictBannerBody.innerHTML = `
      <span class="predict-banner__icon ${match ? "is-match" : "is-mismatch"}">
        ${match ? checkIcon() : crossIcon()}
      </span>
      <div class="predict-banner__text">
        <strong>Predicted "${escapeHtml(data.predicted_class)}"</strong>
        <span class="predict-banner__meta">${escapeHtml(data.model)}</span>
        <p>${match ? "Matches the actual class you selected." : `You selected "${escapeHtml(data.actual_class)}" as the actual class.`}</p>
        ${data.demo_mode ? '<span class="predict-banner__demo">Demo prediction</span>' : ""}
      </div>
    `;
  }

  function macroRatioHtml(nutrition) {
    const protein = parseFloat(nutrition.protein);
    const carbs = parseFloat(
      nutrition.carbohydrates !== undefined ? nutrition.carbohydrates : nutrition.carbs
    );
    const fat = parseFloat(nutrition.fat);
    const vals = [protein, carbs, fat].map((v) => (isNaN(v) || v < 0 ? 0 : v));
    const total = vals[0] + vals[1] + vals[2];
    if (total <= 0) return "";

    let p = Math.round((vals[0] / total) * 100);
    let c = Math.round((vals[1] / total) * 100);
    let f = 100 - p - c; // absorb rounding drift into the last segment

    return `<div class="nutri-card__macro">
      <div class="nutri-card__macro-bar" role="img" aria-label="Protein ${p}%, carbohydrates ${c}%, fat ${f}%">
        <span class="macro-seg macro-seg--protein" style="width:${p}%"></span>
        <span class="macro-seg macro-seg--carbs" style="width:${c}%"></span>
        <span class="macro-seg macro-seg--fat" style="width:${f}%"></span>
      </div>
      <div class="nutri-card__macro-legend">
        <span class="macro-legend-item"><i class="macro-dot macro-dot--protein"></i>Protein <b>${p}%</b></span>
        <span class="macro-legend-item"><i class="macro-dot macro-dot--carbs"></i>Carbs <b>${c}%</b></span>
        <span class="macro-legend-item"><i class="macro-dot macro-dot--fat"></i>Fat <b>${f}%</b></span>
      </div>
    </div>`;
  }

  function nutritionCardHtml(kind, label, name, nutrition) {
    if (!nutrition || Object.keys(nutrition).length === 0) {
      return `<div class="nutri-card nutri-card--${kind}">
        <div class="nutri-card__eyebrow"><span class="dot"></span>${label} <span class="nutri-card__eyebrow-name">${escapeHtml(name)}</span></div>
        <p class="empty-metric">No nutrition data found.</p>
      </div>`;
    }

    const { calories, source, ...rest } = nutrition;

    const rows = Object.entries(rest)
      .filter(([, value]) => value !== undefined && value !== null && value !== "")
      .map(
        ([key, value]) => `<div class="nutri-card__row">
        <span class="nutri-card__row-label">${escapeHtml(formatNutritionLabel(key))}</span>
        <span class="nutri-card__row-value">${escapeHtml(String(value))}</span>
      </div>`
      )
      .join("");

    const sourceHtml = source
      ? `<a class="nutri-card__source" href="${escapeAttr(source)}" target="_blank" rel="noopener noreferrer">View source ↗</a>`
      : "";

    return `<div class="nutri-card nutri-card--${kind}">
      <div class="nutri-card__eyebrow"><span class="dot"></span>${label} <span class="nutri-card__eyebrow-name">${escapeHtml(name)}</span></div>
      <div class="nutri-card__calories">
        <span class="nutri-card__calories-label">Calories</span>
        <span class="nutri-card__calories-value">${calories !== undefined ? escapeHtml(String(calories)) : "—"}</span>
      </div>
      ${macroRatioHtml(nutrition)}
      ${rows}
      ${sourceHtml}
    </div>`;
  }

  function renderNutritionCards(data) {
    nutritionPlaceholder.hidden = true;
    nutritionCards.hidden = false;
    nutritionCards.innerHTML =
      nutritionCardHtml("actual", data.actual_class, "(Actual class)", data.actual_nutrition) +
      nutritionCardHtml("predicted", data.predicted_class, "(Predicted class)", data.predicted_nutrition);
  }

  // ------------------------------------------------------------------
  // Rendering: metrics + inline confusion matrix (one-vs-rest, per class)
  // ------------------------------------------------------------------
  function confusionMiniGridHtml(cm, className) {
    const normCm = normalizeKeys(cm);
    if (Object.keys(normCm).length === 0) return "";
    const { tp = 0, fp = 0, fn = 0, tn = 0 } = normCm;
    const vals = { tp: Number(tp) || 0, fp: Number(fp) || 0, fn: Number(fn) || 0, tn: Number(tn) || 0 };
    const total = vals.tp + vals.fp + vals.fn + vals.tn;
    const max = Math.max(vals.tp, vals.fp, vals.fn, vals.tn, 1);
    const shortName = className && className.length > 16 ? `${className.slice(0, 15)}…` : className;

    const cell = (kind) => {
      const value = vals[kind];
      const pct = total > 0 ? Math.round((value / total) * 100) : 0;
      const intensity = (value / max).toFixed(2);
      return `<div class="cm-cell cm-cell--${kind}" style="--intensity:${intensity}" title="${CM_LABELS[kind]} — ${value} (${pct}%)">
        <span class="cm-cell__label">${kind.toUpperCase()}</span>
        <span class="cm-cell__value">${value}</span>
        <span class="cm-cell__pct">${pct}%</span>
      </div>`;
    };

    return `<div class="metric-card__cm">
      <span class="metric-card__cm-title">Confusion matrix${className ? ` <span class="metric-card__cm-title-name" title="${escapeAttr(className)}">· ${escapeHtml(shortName)}</span>` : ""}</span>
      <div class="cm-matrix">
        <span class="cm-matrix__corner">Actual ↓ / Pred →</span>
        <span class="cm-matrix__colhead">Yes</span>
        <span class="cm-matrix__colhead">No</span>
        <span class="cm-matrix__rowhead">Yes</span>
        ${cell("tp")}${cell("fn")}
        <span class="cm-matrix__rowhead">No</span>
        ${cell("fp")}${cell("tn")}
      </div>
    </div>`;
  }

  function metricCardHtml(kind, label, name, metrics, cm) {
    const normMetrics = normalizeKeys(metrics);
    const normCm = normalizeKeys(cm);
    const hasMetrics = Object.keys(normMetrics).length > 0;
    const hasCm = Object.keys(normCm).length > 0;

    if (!hasMetrics && !hasCm) {
      return `<div class="metric-card metric-card--${kind}">
        <div class="metric-card__eyebrow"><span class="dot"></span>${label} <span class="metric-card__eyebrow-name">${escapeHtml(name)}</span></div>
        <p class="empty-metric">No metrics found.</p>
      </div>`;
    }

    const rows = hasMetrics
      ? METRIC_LABELS.map(([key, rowLabel]) => {
          if (normMetrics[key] === undefined) return "";
          const pct = Math.round(normMetrics[key] * 100);
          return `<div class="metric-row">
            <div class="metric-row__top">
              <span class="metric-row__label">${rowLabel}</span>
              <span class="metric-row__value">${pct}%</span>
            </div>
            <div class="metric-row__bar"><div class="metric-row__bar-fill" style="width:${pct}%"></div></div>
          </div>`;
        }).join("")
      : `<p class="empty-metric">No metrics found.</p>`;

    return `<div class="metric-card metric-card--${kind}">
      <div class="metric-card__eyebrow"><span class="dot"></span>${label} <span class="metric-card__eyebrow-name">${escapeHtml(name)}</span></div>
      ${rows}
      ${confusionMiniGridHtml(normCm, label)}
    </div>`;
  }

  function renderMetricsCards(data) {
    metricsPlaceholder.hidden = true;
    metricsCards.hidden = false;
    metricsCards.innerHTML =
      metricCardHtml("actual", data.actual_class, "(Actual class)", data.actual_metrics, data.actual_confusion_matrix) +
      metricCardHtml("predicted", data.predicted_class, "(Predicted class)", data.predicted_metrics, data.predicted_confusion_matrix);
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------
  function showStatus(msg) {
    statusLine.textContent = msg;
  }
  function clearStatus() {
    statusLine.textContent = "";
  }
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
  }
  function escapeAttr(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  function checkIcon() {
    return `<svg viewBox="0 0 24 24" width="18" height="18" fill="none"><path d="M5 12.5 10 17l9-10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  }
  function crossIcon() {
    return `<svg viewBox="0 0 24 24" width="18" height="18" fill="none"><path d="M6 6l12 12M18 6 6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;
  }
})();

/* =========================================================================
   Fit-to-viewport: measures the app's natural height and scales it down
   (never up) so everything fits the visible window with no scrolling.
   The app's layout width is 100vw / scale, so the layout can change with
   the scale; we therefore re-measure a few times until it settles.
   ========================================================================= */
(function () {
  const MIN_SCALE = 0.4;
  const desktop = window.matchMedia("(min-width: 1081px)");
  const root = document.documentElement;
  let raf = 0;

  function naturalHeightAt(app, workspace, scale) {
    root.style.setProperty("--fit-scale", String(scale));
    app.style.height = "auto";
    workspace.style.flex = "none";
    const h = app.offsetHeight;          // forces layout; nothing paints in between
    app.style.height = "";
    workspace.style.flex = "";
    return h;
  }

  function fit() {
    raf = 0;
    const app = document.querySelector(".app");
    const workspace = document.querySelector(".workspace");
    if (!app || !workspace) return;

    if (!desktop.matches) {              // tablet / phone: normal scrolling layout
      root.style.removeProperty("--fit-scale");
      return;
    }

    const available = window.innerHeight;
    let scale = 1;
    for (let i = 0; i < 5; i++) {
      const natural = naturalHeightAt(app, workspace, scale);
      const next = Math.max(MIN_SCALE, Math.min(1, available / (natural || available)));
      if (Math.abs(next - scale) < 0.004) { scale = Math.min(scale, next); break; }
      scale = next;
    }
    // final safety check at the chosen scale
    const finalNatural = naturalHeightAt(app, workspace, scale);
    if (finalNatural * scale > available + 1) {
      scale = Math.max(MIN_SCALE, available / finalNatural);
    }
    scale = Math.floor(scale * 1000) / 1000;
    root.style.setProperty("--fit-scale", String(scale));
  }

  function schedule() {
    if (!raf) raf = requestAnimationFrame(fit);
  }

  window.addEventListener("resize", schedule);
  window.addEventListener("orientationchange", schedule);
  window.addEventListener("load", schedule);
  document.addEventListener("load", schedule, true);   // images finishing loading
  if (desktop.addEventListener) desktop.addEventListener("change", schedule);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(schedule);

  // Re-fit whenever results render, panels swap, or the class picker collapses
  const mo = new MutationObserver(schedule);
  document.querySelectorAll(".panel").forEach((p) =>
    mo.observe(p, { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden", "class", "src"] })
  );

  schedule();
})();