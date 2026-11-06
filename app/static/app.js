// Fracture Check page logic: upload -> POST /api/predict -> draw the result.
import { CARE_SOURCES, EMERGENCY_NUMBERS, REGIONS, careFor } from "/static/care.js";

const $ = (id) => document.getElementById(id);
const stage = $("main");
const ACCEPTED = ["image/jpeg", "image/png"];
let info = null; // from /api/info
let maxBytes = 15 * 1024 * 1024;
let objectUrl = null;

const pct = (p) => {
  const v = 100 * p;
  // One decimal below 10% and above 99%, where rounding would hide the difference
  // from the flag line (9.7%) or from certainty.
  return v < 10 || v > 99 ? `${v.toFixed(1)}%` : `${Math.round(v)}%`;
};
const inTen = (x) => Math.round(10 * x);
const inHundred = (x) => Math.round(100 * x);

// ------------------------------------------------------------------ start-up
async function start() {
  renderCare("general");
  setupUpload();
  setupLayers();
  $("again").addEventListener("click", reset);
  $("problem-retry").addEventListener("click", () => { reset(); $("file").click(); });

  try {
    const health = await fetch("/api/health");
    if (!health.ok) {
      const body = await health.json().catch(() => ({}));
      showBanner(`The models are not loaded on this server, so X-rays can't be checked. ${body.error || ""}`);
      return;
    }
    info = await (await fetch("/api/info")).json();
    maxBytes = info.limits.max_upload_mb * 1024 * 1024;
    fillInfo(info);
  } catch {
    showBanner("The server can't be reached. Check that it is running, then reload the page.");
  }
}

function showBanner(text) {
  const b = document.createElement("div");
  b.className = "banner";
  b.setAttribute("role", "alert");
  b.textContent = text;
  document.body.prepend(b);
}

function fillInfo(i) {
  const mc = i.model_card;
  const set = (key, text) => document.querySelectorAll(`[data-info="${key}"]`).forEach((n) => { n.textContent = text; });
  set("threshold-pct", pct(i.threshold));
  set("cutoff", i.box_cutoff.toFixed(2));
  set("top1", `${inHundred(mc.detector.top1_pointing)} of every 100`);
  set("hw", `${mc.detector.hardware_top1_pointing[0]} of ${mc.detector.hardware_top1_pointing[1]}`);

  const found = Math.round(mc.classifier.recall * mc.test_fractured);
  const stats = [
    ["Fractures caught", `${found} of ${mc.test_fractured}`, `recall ${mc.classifier.recall.toFixed(3)}`],
    ["Fracture-free X-rays cleared", `${inHundred(mc.classifier.specificity)}%`, `specificity ${mc.classifier.specificity.toFixed(3)}`],
    ["Flagged X-rays that had a fracture", `${inTen(mc.classifier.precision)} in 10`, `precision ${mc.classifier.precision.toFixed(3)}`],
    ["Cleared X-rays that had no fracture", `${inHundred(mc.classifier.npv)} in 100`, `negative predictive value ${mc.classifier.npv.toFixed(3)}`],
    ["Ranking quality", mc.classifier.roc_auc.toFixed(2), "ROC-AUC (1.00 is perfect)"],
    ["Top box on the fracture", `${inHundred(mc.detector.top1_pointing)}%`, "of fractured X-rays"],
    ["Fractures boxed at 1 false box per image", `${inHundred(mc.detector.sens_at_1_fp)}%`, "detector sensitivity"],
    ["Test set", `${mc.test_images}`, `X-rays, ${mc.test_fractured} fractured`],
  ];
  const dl = $("stats");
  dl.replaceChildren(...stats.map(([label, value, note]) => {
    const div = document.createElement("div");
    div.className = "stat";
    div.innerHTML = `<dt></dt><dd><span></span><small></small></dd>`;
    div.querySelector("dt").textContent = label;
    div.querySelector("dd span").textContent = value;
    div.querySelector("small").textContent = note;
    return div;
  }));
}

// ------------------------------------------------------------------- upload
function setupUpload() {
  const drop = $("drop");
  const input = $("file");
  input.addEventListener("change", () => input.files[0] && handleFile(input.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));
  // Drop a new X-ray anywhere on the lightbox, even while a result is showing.
  const box = $("lightbox");
  box.addEventListener("dragover", (e) => e.preventDefault());
  box.addEventListener("drop", (e) => { e.preventDefault(); e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]); });
  window.addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files || [])].find((x) => ACCEPTED.includes(x.type));
    if (f) handleFile(f);
  });
}

async function handleFile(file) {
  if (stage.dataset.state === "working") return;
  if (!ACCEPTED.includes(file.type)) return showProblem("That file isn't a JPEG or PNG. Choose an X-ray image in one of those formats.");
  if (file.size > maxBytes) return showProblem(`That image is larger than ${Math.round(maxBytes / 1048576)} MB. Choose a smaller file.`);
  if (!info) return showProblem("The server isn't ready to check X-rays yet. Reload the page in a moment.");

  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  $("xray").src = objectUrl;
  $("heat").removeAttribute("src");
  $("boxes").replaceChildren();
  $("drop").hidden = true;
  $("problem").hidden = true;
  $("film").hidden = false;
  $("layers").hidden = true;
  $("result").hidden = true;
  stage.dataset.state = "working";
  $("film-caption").textContent = "Analysing the X-ray";

  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `The server answered with an error (${res.status}).`);
    showResult(body);
  } catch (err) {
    showProblem(err.message || "The X-ray couldn't be checked. Try again.");
  }
}

function showProblem(text) {
  stage.dataset.state = "empty";
  $("film").hidden = true;
  $("drop").hidden = true;
  $("layers").hidden = true;
  $("result").hidden = true;
  $("problem-text").textContent = text;
  $("problem").hidden = false;
}

function reset() {
  stage.dataset.state = "empty";
  $("film").hidden = true;
  $("problem").hidden = true;
  $("layers").hidden = true;
  $("result").hidden = true;
  $("drop").hidden = false;
  $("file").value = "";
}

// ------------------------------------------------------------------- result
function showResult(r) {
  $("heat").src = `data:image/png;base64,${r.heatmap_png}`;
  drawBoxes(r.boxes);

  const result = $("result");
  result.classList.toggle("is-flagged", r.flagged);
  result.classList.toggle("is-clear", !r.flagged);
  $("verdict").textContent = r.flagged ? "Possible fracture" : "No fracture flagged";
  $("score-num").textContent = pct(r.probability);
  $("scale").setAttribute("aria-label", `Fracture score ${pct(r.probability)}; the flag line is at ${pct(r.threshold)}.`);
  $("scale-flag").style.left = `${100 * r.threshold}%`;
  $("meaning").replaceChildren(...meaning(r).map(([cls, text]) => {
    const p = document.createElement("p");
    if (cls) p.className = cls;
    p.textContent = text;
    return p;
  }));

  $("toggle-boxes").hidden = !r.flagged;
  $("film-caption").textContent = r.flagged ? "X-ray with the model's heatmap and likely fracture location" : "X-ray with the model's heatmap";
  stage.dataset.state = "done";
  result.hidden = false;
  $("layers").hidden = false;
  applyLayers();
  // Let the bar grow from zero: the one moment of motion in the result.
  $("scale-fill").style.width = "0";
  $("scale-mark").style.left = "0";
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("scale-fill").style.width = `${100 * r.probability}%`;
    $("scale-mark").style.left = `${100 * r.probability}%`;
  }));
}

function meaning(r) {
  const mc = info.model_card;
  const secs = ((r.timing_ms.classifier_and_heatmap + r.timing_ms.detector) / 1000).toFixed(1);
  if (r.flagged) {
    const lines = [["", `The score is above the flag line (${pct(r.threshold)}), so this X-ray should be checked by a doctor.`]];
    const top = r.boxes[0];
    if (!top) lines.push(["", "The detector found no likely location. The heatmap shows the area the classifier relied on."]);
    else if (top.confident) lines.push(["", `The orange box marks the most likely location (detector score ${top.score.toFixed(2)}).`]);
    else lines.push(["", `The dashed box is the detector's best guess, with low confidence (score ${top.score.toFixed(2)}).`]);
    lines.push(["fine", `In testing, about ${inTen(mc.classifier.precision)} in 10 flagged X-rays had a fracture. A flag means "look closer", not "broken". Analysed in ${secs} s.`]);
    return lines;
  }
  const missed = mc.test_fractured - Math.round(mc.classifier.recall * mc.test_fractured);
  return [
    ["", `The score is below the flag line (${pct(r.threshold)}), so the model sees no sign of a fracture.`],
    ["fine", `In testing, ${inHundred(mc.classifier.npv)} in 100 cleared X-rays had no fracture, but the model still missed ${missed} of ${mc.test_fractured} fractures. If it hurts badly, can't take weight or looks out of shape, get it checked anyway. Analysed in ${secs} s.`],
  ];
}

function drawBoxes(boxes) {
  const svg = $("boxes");
  const ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren(...boxes.map((b, i) => {
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", b.x0);
    rect.setAttribute("y", b.y0);
    rect.setAttribute("width", Math.max(0, b.x1 - b.x0));
    rect.setAttribute("height", Math.max(0, b.y1 - b.y0));
    rect.setAttribute("class", i === 0 ? `box-top${b.confident ? "" : " box-low"}` : "box-other");
    return rect;
  }));
}

function setupLayers() {
  ["show-heat", "show-boxes", "heat-opacity"].forEach((id) => $(id).addEventListener("input", applyLayers));
}

function applyLayers() {
  const heatOn = $("show-heat").checked;
  const boxesOn = $("show-boxes").checked;
  $("heat").style.opacity = heatOn ? $("heat-opacity").value / 100 : 0;
  $("boxes").style.display = boxesOn ? "" : "none";
  $("show-heat").closest(".toggle").classList.toggle("is-off", !heatOn);
  $("show-boxes").closest(".toggle").classList.toggle("is-off", !boxesOn);
  $("heat-opacity").disabled = !heatOn;
}

// --------------------------------------------------------------- care guide
function renderCare(region) {
  const wrap = $("regions");
  if (!wrap.children.length) {
    REGIONS.forEach((r) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "region";
      b.setAttribute("role", "radio");
      b.dataset.region = r.id;
      b.textContent = r.label;
      b.addEventListener("click", () => renderCare(r.id));
      wrap.append(b);
    });
    wrap.addEventListener("keydown", (e) => {
      if (!["ArrowRight", "ArrowLeft"].includes(e.key)) return;
      const ids = REGIONS.map((r) => r.id);
      const now = ids.indexOf(wrap.querySelector('[aria-checked="true"]').dataset.region);
      const next = ids[(now + (e.key === "ArrowRight" ? 1 : ids.length - 1)) % ids.length];
      renderCare(next);
      wrap.querySelector(`[data-region="${next}"]`).focus();
    });
    $("care-sources").replaceChildren(...CARE_SOURCES.flatMap((s, i) => {
      const a = document.createElement("a");
      a.href = s.url;
      a.textContent = s.title;
      a.rel = "noopener";
      return i < CARE_SOURCES.length - 1 ? [a, document.createTextNode("; ")] : [a];
    }));
    $("numbers").textContent = EMERGENCY_NUMBERS;
  }
  wrap.querySelectorAll(".region").forEach((b) => {
    const on = b.dataset.region === region;
    b.setAttribute("aria-checked", String(on));
    b.tabIndex = on ? 0 : -1;
  });

  const c = careFor(region);
  const list = (id, items) => $(id).replaceChildren(...items.map((t) => Object.assign(document.createElement("li"), { textContent: t })));
  list("care-emergency", c.emergency);
  $("emergency-note").hidden = !c.emergencyNote;
  $("emergency-note").textContent = c.emergencyNote || "";
  list("care-first", c.firstAid);
  list("care-treat", c.treatment);
  $("care-heal").textContent = c.healing;
  list("care-dos", c.dos);
  list("care-donts", c.donts);
  list("care-warn", c.warning);
}

start();
