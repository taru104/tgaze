"use strict";
// tgaze Web アプリ（フロントエンド）。カメラ映像を JPEG にしてローカルサーバ (server.py) へ送り、
// 返ってきた視線点（画面正規化座標 [0,1]）を描画・記録する。外部への通信はしない。

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// キャリブレーションの9点（中央から開始し、外周を時計回り）。tgaze.NINE_POINTS と同じ位置。
const CALIB_POINTS = [[0.5, 0.5], [0.1, 0.1], [0.5, 0.1], [0.9, 0.1], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9], [0.1, 0.9], [0.1, 0.5]];
const SETTLE_MS = 1300;         // 点が移動してから視線が追いつくまで待つ時間（0.8 → 1.8 → 1.3 秒）
const SAMPLES_PER_POINT = 12;   // 1点で集めるフレーム数（tgaze.GazeTracker.calibrate と同じ）
const POINT_TIMEOUT_MS = 4000;  // 顔が取れなくてもこの時間で次の点へ
const VIEW_DIST_CM = 60;        // 誤差を角度に換算するときの想定距離
const FIX_DISPERSION_PX = 80;   // 注視（fixation）とみなす広がりの上限（ステージ上の px）
const FIX_MIN_MS = 100;         // 注視とみなす最短時間
const AREA_COLORS = { "メインビジュアル": "#2f6bff", "エントリーボタン": "#6a2cff", "説明文エリア": "#19b3c9",
  "ナビゲーション": "#3c4d78", "その他": "#c34bd6" };
const STORE_KEY = "tgaze.sessions.v1";

const state = { stream: null, calib: null, calibrated: false, info: { model: "", capture: [640, 480] } };

// サーバが使うモデルと、撮る解像度（M8 系は 1920x1080）を聞いて、画面の隅にモデル名を出す
const infoReady = fetch("/api/info").then((r) => r.json()).then((i) => {
  state.info = i;
  const tag = document.createElement("div");
  tag.textContent = "モデル: " + i.model;
  Object.assign(tag.style, { position: "fixed", right: "8px", bottom: "6px", zIndex: 9999, font: "11px sans-serif",
    color: "#888", background: "rgba(255,255,255,.6)", padding: "1px 6px", borderRadius: "4px", pointerEvents: "none" });
  document.body.appendChild(tag);
}).catch(() => {});

// ── サーバとの通信 ─────────────────────────────
async function post(path, body) {
  const r = await fetch(path, { method: "POST", body: body || null, headers: body ? { "Content-Type": "image/jpeg" } : {} });
  return r.json();
}
const grabCanvas = document.createElement("canvas");
function grab() {
  const v = $("#video");
  const cw = state.info.capture[0], ch = state.info.capture[1];
  const w = cw, h = Math.round(cw * (v.videoHeight || ch) / (v.videoWidth || cw));
  grabCanvas.width = w; grabCanvas.height = h;
  grabCanvas.getContext("2d").drawImage(v, 0, 0, w, h);
  return new Promise((res) => grabCanvas.toBlob(res, "image/jpeg", 0.85));
}

// フレームを送り続けるループ。新しいループを始めるか stopLoop() で前のループは止まる。
let loopId = 0;
function startLoop(onResult) {
  const id = ++loopId;
  (async () => {
    while (id === loopId) {
      try {
        if (!state.stream) { await sleep(200); continue; }
        const r = await post("/api/frame", await grab());
        if (id === loopId) onResult(r);
      } catch (e) { await sleep(500); }
    }
  })();
  return id;
}
const stopLoop = () => { loopId++; };

async function ensureCamera() {
  if (state.stream) return true;
  try {
    await infoReady;
    const [cw, ch] = state.info.capture;
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: cw }, height: { ideal: ch } }, audio: false });
    $("#video").srcObject = state.stream;
    await $("#video").play();
    return true;
  } catch (e) {
    $("#camerr").hidden = false;
    $("#camerr").textContent = "カメラを開けませんでした。ブラウザのアドレスバーからカメラの使用を許可して、ページを再読み込みしてください。（" + e.name + "）";
    return false;
  }
}

// ── 画面座標の変換 ─────────────────────────────
// サーバの視線点は「画面全体」に対する [0,1]。ブラウザのページ内座標（client）との相互変換。
const isFs = () => !!document.fullscreenElement;
function chrome() {
  const side = Math.max(0, (window.outerWidth - window.innerWidth) / 2);
  return { left: side, top: Math.max(0, window.outerHeight - window.innerHeight - side) };
}
function screenToClient(gx, gy) {
  if (isFs()) return [gx * screen.width, gy * screen.height - (screen.height - innerHeight)];
  const c = chrome();
  return [gx * screen.width - (window.screenX - (screen.left || 0)) - c.left,
          gy * screen.height - (window.screenY - (screen.top || 0)) - c.top];
}
function clientToScreen(cx, cy) {
  if (isFs()) return [cx / screen.width, (cy + screen.height - innerHeight) / screen.height];
  const c = chrome();
  return [(cx + window.screenX - (screen.left || 0) + c.left) / screen.width,
          (cy + window.screenY - (screen.top || 0) + c.top) / screen.height];
}
async function enterFs() { try { if (!isFs()) await document.documentElement.requestFullscreen(); } catch (e) { /* 拒否されたら窓のまま */ } }
async function exitFs() { try { if (isFs()) await document.exitFullscreen(); } catch (e) { /* noop */ } }

// ── ルーティング ─────────────────────────────
const PAGES = ["home", "check", "guide", "ready", "calib", "done", "measure", "results", "game", "keyboard"];
const STEP_OF = { check: 1, guide: 2, ready: 3, calib: 3, done: 4 };
let current = null;

function route() {
  const name = (location.hash.replace(/^#\/?/, "") || "home").split("?")[0];
  const page = PAGES.includes(name) ? name : "home";
  if (current && LEAVE[current]) LEAVE[current]();
  current = page;
  PAGES.forEach((p) => { $("#page-" + p).hidden = p !== page; });
  const step = STEP_OF[page];
  $("#topbar").hidden = page === "calib" || page === "game" || page === "keyboard";
  $("#steps").hidden = !step;
  $$("#steps li").forEach((li) => {
    const s = +li.dataset.step;
    li.classList.toggle("on", s === step); li.classList.toggle("done", s < step);
  });
  $("#mainnav").hidden = !!step || page === "measure";
  $("#measurebar").hidden = page !== "measure";
  $("#btnOpenApp").hidden = page !== "home";
  $$("#mainnav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === (page === "results" ? "results" : "home")));
  window.scrollTo(0, 0);
  if (ENTER[page]) ENTER[page]();
}
const ENTER = {}, LEAVE = {};

$$("[data-scroll]").forEach((a) => a.addEventListener("click", (e) => {
  if (current === "home") { e.preventDefault(); $("#" + a.dataset.scroll).scrollIntoView({ behavior: "smooth" }); }
}));

// ── ② カメラ・姿勢確認 ─────────────────────────────
ENTER.check = async () => {
  $(".camwrap").prepend($("#video"));
  $("#btnCheckNext").disabled = true;
  if (!(await ensureCamera())) { $("#checkhint").textContent = "カメラが使えません。"; return; }
  $("#checkhint").textContent = "4つの項目がすべて緑になると「次へ」を押せます。";
  let okSince = 0;
  startLoop((r) => {
    const box = $("#facebox");
    if (r.bbox) {
      const W = $(".camwrap").clientWidth, H = $(".camwrap").clientHeight;
      const [x0, y0, x1, y1] = r.bbox;
      box.hidden = false;   // 映像は鏡像表示なので x を反転
      Object.assign(box.style, { left: (1 - x1) * W + "px", top: y0 * H + "px", width: (x1 - x0) * W + "px", height: (y1 - y0) * H + "px" });
    } else box.hidden = true;
    const ck = r.checks || {};
    $$("#checklist li").forEach((li) => li.classList.toggle("ok", !!ck[li.dataset.k]));
    const all = ["face_full", "eyes_open", "frontal", "bright"].every((k) => ck[k]);
    if (all) okSince = okSince || performance.now(); else okSince = 0;
    if (okSince && performance.now() - okSince > 600) $("#btnCheckNext").disabled = false;
    if (!r.face) $("#checkhint").textContent = "顔が見つかりません。カメラの正面に座ってください。";
    else if (!all) $("#checkhint").textContent = "4つの項目がすべて緑になると「次へ」を押せます。";
    else $("#checkhint").textContent = "準備ができました。";
  });
};
LEAVE.check = stopLoop;
$("#btnCheckNext").addEventListener("click", () => { location.hash = "#/guide"; });

// ── ③ 説明（画面サイズの記憶） ─────────────────────────────
try { const v = localStorage.getItem("tgaze.screenInch"); if (v) $("#screenSize").value = v; } catch (e) { /* noop */ }
$("#screenSize").addEventListener("change", (e) => { try { localStorage.setItem("tgaze.screenInch", e.target.value); } catch (_) { /* noop */ } });
function screenCm() {
  const inch = parseFloat($("#screenSize").value) || 13.6, ar = screen.width / screen.height;
  const w = inch * 2.54 * ar / Math.hypot(ar, 1);
  return [w, w / ar];
}

// ── ④ 準備完了 → ⑤ キャリブレーション ─────────────────────────────
ENTER.ready = () => { ensureCamera(); };
$("#btnStartCalib").addEventListener("click", async () => {
  if (!(await ensureCamera())) { location.hash = "#/check"; return; }
  await enterFs();
  location.hash = "#/calib";
});

let calibAbort = false;
ENTER.calib = async () => {
  calibAbort = false;
  const dot = $("#calibDot"), msg = $("#calibMsg");
  msg.textContent = "点を見つめてください（Esc で中止）";
  await sleep(300);           // 全画面への切り替えが落ち着くまで
  await post("/api/calib/reset");
  for (let i = 0; i < CALIB_POINTS.length; i++) {
    const [tx, ty] = CALIB_POINTS[i];
    $("#calibCount").textContent = `${i + 1} / ${CALIB_POINTS.length}`;
    $("#calibBar").style.width = (i / CALIB_POINTS.length * 100) + "%";
    const cx = tx * innerWidth, cy = ty * innerHeight;
    dot.style.left = cx + "px"; dot.style.top = cy + "px"; dot.style.setProperty("--s", 1);
    const [sx, sy] = clientToScreen(cx, cy);   // サーバに渡す正解は画面全体に対する位置
    const t0 = performance.now();
    while (performance.now() - t0 < SETTLE_MS) {   // 待つ間もフレームは送る（追跡を途切れさせない）
      if (calibAbort) return;
      try { await post("/api/frame", await grab()); } catch (e) { await sleep(100); }
    }
    let got = 0; const t1 = performance.now();
    while (got < SAMPLES_PER_POINT && performance.now() - t1 < POINT_TIMEOUT_MS) {
      if (calibAbort) return;
      try {
        const r = await post(`/api/calib/sample?x=${sx}&y=${sy}`, await grab());
        if (r.ok) { got++; dot.style.setProperty("--s", 1 - 0.6 * got / SAMPLES_PER_POINT); }
        msg.textContent = r.ok ? "点を見つめてください（Esc で中止）" : "顔が検出できません。カメラの正面を向いてください";
      } catch (e) { await sleep(100); }
    }
  }
  $("#calibBar").style.width = "100%";
  msg.textContent = "計算しています…";
  state.calib = await post("/api/calib/fit");
  state.calibrated = !!state.calib.ok;
  calibAbort = true;          // 以降の全画面解除を「中止」と扱わない
  await exitFs();
  location.hash = "#/done";
};
LEAVE.calib = () => { calibAbort = true; };
document.addEventListener("fullscreenchange", () => {
  if (!isFs()) {
    document.body.classList.remove("fs");
    if (current === "calib" && !calibAbort) { calibAbort = true; location.hash = "#/guide"; }
  }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && current === "calib") { calibAbort = true; location.hash = "#/guide"; } });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && (current === "game" || current === "keyboard")) location.hash = state.calibrated ? "#/done" : "#/ready"; });

// ── ⑥ 完了 ─────────────────────────────
ENTER.done = () => {
  const c = state.calib;
  if (!c) { location.hash = "#/guide"; return; }
  const ok = c.ok;
  $("#doneMark").textContent = ok ? "✓" : "!";
  $("#doneMark").classList.toggle("warn", !ok);
  $("#doneTitle").textContent = ok ? "キャリブレーションが完了しました！" : "キャリブレーションできませんでした";
  $("#doneLead").innerHTML = ok ? "視線の推定が可能になりました。<br>精度を確認し、必要に応じて再調整してください。"
    : "顔が検出できたフレームが足りませんでした。明るさと顔の位置を確認して、もう一度お試しください。";
  $('#page-done a[href="#/measure"]').hidden = !ok;
  if (!ok) { $("#doneErr").textContent = "–"; $("#doneErrSub").textContent = ""; $("#donePts").textContent = "–"; $("#doneStab").textContent = "–"; return; }
  const [w, h] = screenCm();
  const cm = Math.hypot(c.loo_x * w, c.loo_y * h);
  const deg = Math.atan(cm / VIEW_DIST_CM) * 180 / Math.PI;
  $("#doneErr").textContent = cm.toFixed(2) + " cm";
  $("#doneErrSub").textContent = `約 ${deg.toFixed(1)}°（${VIEW_DIST_CM}cm の距離を想定）`;
  $("#donePts").textContent = `${c.valid_points} / ${CALIB_POINTS.length}`;
  const stab = cm < 2.5 ? ["● 良好", "var(--ok)"] : cm < 4 ? ["● 普通", "#f0a020"] : ["● 要再調整", "var(--danger)"];
  $("#doneStab").textContent = stab[0]; $("#doneStab").style.color = stab[1];
};

// ── ⑦ リアルタイム計測 ─────────────────────────────
const meas = { samples: [], start: 0, pausedAt: 0, pausedTotal: 0, paused: false, timerId: 0 };
const activeMs = () => (meas.paused ? meas.pausedAt : performance.now()) - meas.start - meas.pausedTotal;
const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

ENTER.measure = async () => {
  $("#miniCam").append($("#video"));
  if (!(await ensureCamera())) { location.hash = "#/check"; return; }
  Object.assign(meas, { samples: [], start: performance.now(), pausedAt: 0, pausedTotal: 0, paused: false });
  setPaused(false);
  meas.timerId = setInterval(() => { $("#timer").textContent = fmt(activeMs() / 1000); }, 250);
  const trail = $("#trail"), tctx = trail.getContext("2d"), dot = $("#gazedot");
  let checked = false;
  startLoop((r) => {
    if (!checked) { checked = true; if (!r.calibrated) { alert("先にキャリブレーションを行ってください。"); location.hash = "#/guide"; return; } }
    $("#nofaceBadge").hidden = r.face;
    if (meas.paused) return;
    const stage = $("#stage").getBoundingClientRect();
    const t = activeMs();
    if (!r.gaze) { meas.samples.push([t, -1, -1, -1]); dot.hidden = true; return; }
    const [cx, cy] = screenToClient(r.gaze[0], r.gaze[1]);
    const size = +$("#optSize").value;
    dot.hidden = !$("#optDot").checked;
    Object.assign(dot.style, { left: cx + "px", top: cy + "px", width: size + "px", height: size + "px", margin: `-${size / 2}px 0 0 -${size / 2}px` });
    const x = (cx - stage.left) / stage.width, y = (cy - stage.top) / stage.height;
    const inside = x >= 0 && x <= 1 && y >= 0 && y <= 1;
    let area = -1;
    if (inside) {
      const el = document.elementFromPoint(cx, cy);
      const a = el && el.closest("[data-area]");
      area = a ? AREAS.indexOf(a.dataset.area) : AREAS.indexOf("その他");
    }
    meas.samples.push([t, inside ? x : -1, inside ? y : -1, area]);
    // 軌跡
    if (trail.width !== stage.width || trail.height !== stage.height) { trail.width = stage.width; trail.height = stage.height; }
    tctx.clearRect(0, 0, trail.width, trail.height);
    if ($("#optTrail").checked) {
      const pts = meas.samples.slice(-$("#optLen").value).filter((s) => s[1] >= 0);
      tctx.lineWidth = 3; tctx.lineCap = "round";
      for (let i = 1; i < pts.length; i++) {
        tctx.strokeStyle = `rgba(47,107,255,${(i / pts.length) * 0.8})`;
        tctx.beginPath(); tctx.moveTo(pts[i - 1][1] * stage.width, pts[i - 1][2] * stage.height);
        tctx.lineTo(pts[i][1] * stage.width, pts[i][2] * stage.height); tctx.stroke();
      }
    }
  });
};
LEAVE.measure = () => { stopLoop(); clearInterval(meas.timerId); $("#gazedot").hidden = true; document.body.classList.remove("fs"); exitFs(); };
const AREAS = [...new Set([...$$("#demo [data-area]").map((e) => e.dataset.area), "その他"])];

function setPaused(p) {
  if (p && !meas.paused) meas.pausedAt = performance.now();
  if (!p && meas.paused) meas.pausedTotal += performance.now() - meas.pausedAt;
  meas.paused = p;
  $("#btnPause").textContent = p ? "▶ 再開" : "一時停止";
  $("#measurebar").classList.toggle("paused", p);
  $("#pausedBadge").hidden = !p;
}
$("#btnPause").addEventListener("click", () => setPaused(!meas.paused));
$("#optCam").addEventListener("change", (e) => { $("#miniCam").hidden = !e.target.checked; });
$("#btnFull").addEventListener("click", async () => { await enterFs(); if (isFs()) document.body.classList.add("fs"); });
$("#stage").addEventListener("click", (e) => {   // クリック位置 = 見ていた位置として補正
  if (meas.paused) return;
  const [sx, sy] = clientToScreen(e.clientX, e.clientY);
  post(`/api/click?x=${sx}&y=${sy}`);
});
$("#btnStop").addEventListener("click", () => {
  const st = $("#stage").getBoundingClientRect();
  const s = { id: Date.now(), date: new Date().toISOString(), dur: activeMs() / 1000, w: Math.round(st.width), h: Math.round(st.height),
    areas: AREAS, samples: meas.samples.map((r) => [Math.round(r[0]), +r[1].toFixed(4), +r[2].toFixed(4), r[3]]) };
  saveSession(s);
  location.hash = "#/results";
});

// ── セッションの保存 ─────────────────────────────
function loadSessions() { try { return JSON.parse(localStorage.getItem(STORE_KEY) || "[]"); } catch (e) { return memSessions; } }
let memSessions = [];
function saveSession(s) {
  const all = loadSessions(); all.push(s);
  memSessions = all;
  // localStorage が一杯なら古い記録から捨てる
  for (let n = all.length; n > 0; n--) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(all.slice(-n))); return; } catch (e) { /* 次は1件減らす */ }
  }
}

// ── 解析 ─────────────────────────────
function analyze(s) {
  const valid = s.samples.filter((r) => r[1] >= 0);
  // 注視（I-DT 法）：広がりが FIX_DISPERSION_PX 以下で FIX_MIN_MS 以上続いた区間
  const fix = []; let i = 0;
  while (i < valid.length) {
    let j = i, x0 = valid[i][1], x1 = x0, y0 = valid[i][2], y1 = y0;
    while (j + 1 < valid.length) {
      const [, x, y] = valid[j + 1];
      const nx0 = Math.min(x0, x), nx1 = Math.max(x1, x), ny0 = Math.min(y0, y), ny1 = Math.max(y1, y);
      if ((nx1 - nx0) * s.w + (ny1 - ny0) * s.h > FIX_DISPERSION_PX || valid[j + 1][0] - valid[j][0] > 300) break;
      x0 = nx0; x1 = nx1; y0 = ny0; y1 = ny1; j++;
    }
    const d = valid[j][0] - valid[i][0];
    if (d >= FIX_MIN_MS) {
      const seg = valid.slice(i, j + 1);
      const cnt = {}; seg.forEach((r) => { cnt[r[3]] = (cnt[r[3]] || 0) + 1; });
      const area = +Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a])[0];
      fix.push({ t: valid[i][0], d, x: seg.reduce((a, r) => a + r[1], 0) / seg.length, y: seg.reduce((a, r) => a + r[2], 0) / seg.length, area });
      i = j + 1;
    } else i++;
  }
  // ページ滞在時間：視線がページ上にあった時間（サンプル間隔の合計、間隔 300ms 超は途切れとみなす）
  let dwell = 0;
  for (let k = 1; k < s.samples.length; k++) {
    const dt = s.samples[k][0] - s.samples[k - 1][0];
    if (s.samples[k][1] >= 0 && dt < 300) dwell += dt;
  }
  const areaMs = {}; s.areas.forEach((a) => { areaMs[a] = 0; });
  fix.forEach((f) => { const a = s.areas[f.area] || "その他"; areaMs[a] += f.d; });
  const fixTotal = fix.reduce((a, f) => a + f.d, 0);
  return { fix, dwell: dwell / 1000, areaMs, fixTotal, meanFix: fix.length ? fixTotal / fix.length / 1000 : 0 };
}

// ── ⑧ 結果分析 ─────────────────────────────
const TAB_TITLES = { summary: "サマリー", heat: "注視点の分布", path: "視線の軌跡", areas: "エリア分析", time: "時間別データ", compare: "比較", report: "レポート" };
let tab = "summary";
$$(".side a").forEach((a) => a.addEventListener("click", () => { tab = a.dataset.tab; renderResults(); }));
$("#sessionSel").addEventListener("change", renderResults);
ENTER.results = () => {
  const all = loadSessions();
  $("#sessionSel").innerHTML = all.map((s, i) => `<option value="${i}">${new Date(s.date).toLocaleString("ja-JP")}（${fmt(s.dur)}）</option>`).reverse().join("");
  renderResults();
};

function renderResults() {
  const all = loadSessions();
  $$(".side a").forEach((a) => a.classList.toggle("on", a.dataset.tab === tab));
  $("#rTitle").textContent = TAB_TITLES[tab];
  $("#rEmpty").hidden = all.length > 0; $("#rBody").hidden = !all.length;
  if (!all.length) return;
  $$(".tab").forEach((t) => { t.hidden = t.dataset.tab !== tab; });
  const s = all[+$("#sessionSel").value] || all[all.length - 1];
  const A = analyze(s);
  if (tab === "summary") {
    $("#kDur").textContent = fmt(s.dur);
    $("#kFix").innerHTML = `${A.fix.length} <small>回</small>`;
    $("#kMean").innerHTML = `${A.meanFix.toFixed(2)} <small>s</small>`;
    $("#kDwell").innerHTML = `${Math.round(A.dwell)} <small>s</small>`;
    drawThumb($("#thumbSummary"), s, A, "dots");
    $("#areaList").innerHTML = areaRows(A).map(([a, ms, pct]) =>
      `<li><i style="background:${AREA_COLORS[a] || "#999"}"></i>${a}<b>${pct.toFixed(1)}%</b></li>`).join("");
  } else if (tab === "heat") drawThumb($("#thumbHeat"), s, A, "heat");
  else if (tab === "path") drawThumb($("#thumbPath"), s, A, "path");
  else if (tab === "areas") {
    $("#areaBars").innerHTML = `<div class="bars">` + areaRows(A).map(([a, ms, pct]) =>
      `<div class="row"><span>${a}</span><div class="track"><div class="fill" style="width:${pct}%;background:${AREA_COLORS[a] || "#999"}"></div></div>` +
      `<span class="v">${(ms / 1000).toFixed(1)} s（${pct.toFixed(1)}%）</span></div>`).join("") + `</div>` +
      `<p class="note">注視時間の合計 ${(A.fixTotal / 1000).toFixed(1)} 秒に対する割合です。</p>`;
  } else if (tab === "time") drawTime($("#timeChart"), s);
  else if (tab === "compare") {
    $("#cmpTable").innerHTML = `<tr><th>計測日時</th><th>計測時間</th><th>注視回数</th><th>平均注視時間</th><th>滞在時間</th><th>最も見られたエリア</th></tr>` +
      all.map((x) => { const B = analyze(x), top = areaRows(B)[0];
        return `<tr><td>${new Date(x.date).toLocaleString("ja-JP")}</td><td>${fmt(x.dur)}</td><td>${B.fix.length}</td><td>${B.meanFix.toFixed(2)} s</td>` +
          `<td>${Math.round(B.dwell)} s</td><td>${top && top[1] ? `${top[0]}（${top[2].toFixed(0)}%）` : "–"}</td></tr>`; }).reverse().join("");
  } else if (tab === "report") {
    $("#report").innerHTML = `<h3>視線計測レポート</h3><p class="note">${new Date(s.date).toLocaleString("ja-JP")} ／ tgaze</p>` +
      `<table><tr><td>計測時間</td><td>${fmt(s.dur)}</td></tr><tr><td>注視回数</td><td>${A.fix.length} 回</td></tr>` +
      `<tr><td>平均注視時間</td><td>${A.meanFix.toFixed(2)} 秒</td></tr><tr><td>ページ滞在時間</td><td>${Math.round(A.dwell)} 秒</td></tr></table>` +
      `<b>エリア別の注視時間</b><table>` + areaRows(A).map(([a, ms, pct]) => `<tr><td>${a}</td><td>${(ms / 1000).toFixed(1)} 秒</td><td>${pct.toFixed(1)}%</td></tr>`).join("") +
      `</table><b>注視点の分布</b><div class="thumb" id="thumbReport"></div>`;
    drawThumb($("#thumbReport"), s, A, "heat");
  }
}
function areaRows(A) {
  return Object.entries(A.areaMs).map(([a, ms]) => [a, ms, A.fixTotal ? ms / A.fixTotal * 100 : 0]).sort((p, q) => q[1] - p[1]);
}

// 計測ページの縮小コピーの上に、点・軌跡・ヒートマップを描く
function drawThumb(box, s, A, mode) {
  const W = box.clientWidth || 600, H = Math.round(W * s.h / s.w), k = W / s.w;
  box.style.height = H + "px";
  box.innerHTML = "";
  const m = document.createElement("div");
  m.className = "mirror"; Object.assign(m.style, { width: s.w + "px", height: s.h + "px", transform: `scale(${k})` });
  m.append($("#demo").cloneNode(true)); m.firstChild.removeAttribute("id");
  const cv = document.createElement("canvas"); cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
  box.append(m, cv);
  const c = cv.getContext("2d"); c.scale(devicePixelRatio, devicePixelRatio);
  if (mode === "dots") {
    s.samples.filter((r) => r[1] >= 0).forEach((r) => {
      c.fillStyle = (AREA_COLORS[s.areas[r[3]]] || "#2f6bff") + "88";
      c.beginPath(); c.arc(r[1] * W, r[2] * H, 3, 0, 7); c.fill();
    });
  } else if (mode === "path") {
    const f = A.fix.slice(0, 30);
    c.strokeStyle = "rgba(106,44,255,.7)"; c.lineWidth = 2; c.beginPath();
    f.forEach((p, i) => (i ? c.lineTo(p.x * W, p.y * H) : c.moveTo(p.x * W, p.y * H))); c.stroke();
    f.forEach((p, i) => {
      const r = 6 + Math.min(24, p.d / 60);
      c.fillStyle = "rgba(47,107,255,.75)"; c.beginPath(); c.arc(p.x * W, p.y * H, r, 0, 7); c.fill();
      c.fillStyle = "#fff"; c.font = "bold 11px sans-serif"; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(i + 1, p.x * W, p.y * H);
    });
  } else {
    // ヒートマップ：注視時間で重みを付けたぼかしを重ね、濃さを色に変換
    const off = document.createElement("canvas"); off.width = W; off.height = H;
    const o = off.getContext("2d"), R = Math.max(30, W / 14);
    const maxD = Math.max(1, ...A.fix.map((f) => f.d));
    A.fix.forEach((f) => {
      const g = o.createRadialGradient(f.x * W, f.y * H, 0, f.x * W, f.y * H, R);
      g.addColorStop(0, `rgba(0,0,0,${0.25 + 0.6 * f.d / maxD})`); g.addColorStop(1, "rgba(0,0,0,0)");
      o.fillStyle = g; o.fillRect(f.x * W - R, f.y * H - R, 2 * R, 2 * R);
    });
    const img = o.getImageData(0, 0, W, H), d = img.data;
    for (let p = 0; p < d.length; p += 4) {
      const a = d[p + 3] / 255; if (!a) continue;
      const [r, g, b] = a < 0.33 ? [47, 107, 255] : a < 0.66 ? [25, 200, 120] : a < 0.85 ? [255, 200, 40] : [255, 60, 60];
      d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = Math.min(200, 60 + a * 200);
    }
    o.putImageData(img, 0, 0);
    c.drawImage(off, 0, 0);
  }
}

function drawTime(cv, s) {
  const W = cv.clientWidth || 800, H = 260;
  cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
  const c = cv.getContext("2d"); c.scale(devicePixelRatio, devicePixelRatio);
  const T = Math.max(1, s.samples.length ? s.samples[s.samples.length - 1][0] : 1), L = 40, B = 24;
  c.strokeStyle = "#e3e7f3"; c.fillStyle = "#5a6286"; c.font = "11px sans-serif";
  for (let v = 0; v <= 1; v += 0.25) { const y = (1 - v) * (H - B - 10) + 10; c.beginPath(); c.moveTo(L, y); c.lineTo(W, y); c.stroke(); c.fillText(v.toFixed(2), 4, y + 4); }
  for (let t = 0; t <= T / 1000; t += Math.max(1, Math.round(T / 1000 / 8))) c.fillText(t + "s", L + t * 1000 / T * (W - L) - 6, H - 6);
  [[1, "#2f6bff"], [2, "#6a2cff"]].forEach(([k, col]) => {
    c.strokeStyle = col; c.lineWidth = 1.5; c.beginPath(); let pen = false;
    s.samples.forEach((r) => {
      if (r[k] < 0) { pen = false; return; }
      const x = L + r[0] / T * (W - L), y = (1 - r[k]) * (H - B - 10) + 10;
      if (pen) c.lineTo(x, y); else c.moveTo(x, y); pen = true;
    });
    c.stroke();
  });
}

$("#btnCsv").addEventListener("click", () => {
  const all = loadSessions(); if (!all.length) return;
  const s = all[+$("#sessionSel").value] || all[all.length - 1];
  const rows = ["time_s,x,y,on_page,area"].concat(s.samples.map((r) =>
    `${(r[0] / 1000).toFixed(3)},${r[1] >= 0 ? r[1] : ""},${r[2] >= 0 ? r[2] : ""},${r[1] >= 0 ? 1 : 0},${r[3] >= 0 ? s.areas[r[3]] : ""}`));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + rows.join("\n")], { type: "text/csv" }));   // BOM 付き（Excel で文字化けしない）
  a.download = `tgaze_${s.date.replace(/[:.]/g, "-")}.csv`; a.click();
});

// ── ターゲットゲーム ─────────────────────────────
// 画面のどこかにターゲットが出る → 当たりの輪の中を GAME_DWELL_MS 見つめ続けると弾けて消え、離れた場所に次が出る。時間切れはない。
// 当たりの輪は今の精度（平均誤差 約3 cm）に合わせた半径。輪の中にいる間は外周のゲージがたまり、外れると少しずつ減る。
const GAME_HIT_CM = 3.5, GAME_DWELL_MS = 500, GAME_POP_MS = 380;
const game = { count: 0, times: [], prog: 0, last: 0, spawnAt: 0, popping: false, pos: null, timers: [] };
const gameR = () => GAME_HIT_CM * screen.width / screenCm()[0];   // 当たりの半径（px。画面の幅と実寸から）

async function startGame() { await enterFs(); location.hash = "#/game"; }
$("#btnGameZero").addEventListener("click", startGame);
$("#btnGameDone").addEventListener("click", startGame);
infoReady.then(() => { $("#btnGameZero").hidden = !state.info.zero_shot; });

function gameSpawn() {
  const R = gameR(), W = innerWidth, H = innerHeight, mx = R + 30, y0 = R + 90, y1 = H * 0.88 - R;
  const far = 0.4 * Math.hypot(W, H);
  let best = null, bestD = -1;
  for (let i = 0; i < 60; i++) {              // 前のターゲットから十分離れた場所を選ぶ（無理なら一番遠い候補）
    const p = [mx + Math.random() * (W - 2 * mx), y0 + Math.random() * Math.max(1, y1 - y0)];
    const d = game.pos ? Math.hypot(p[0] - game.pos[0], p[1] - game.pos[1]) : Infinity;
    if (d >= far) { best = p; break; }
    if (d > bestD) { best = p; bestD = d; }
  }
  game.pos = best; game.prog = 0; game.spawnAt = performance.now();
  const t = $("#gameTarget"), size = 2.2 * R;
  Object.assign(t.style, { width: size + "px", height: size + "px", left: best[0] - size / 2 + "px", top: best[1] - size / 2 + "px" });
  $("#gameCharge").style.strokeDashoffset = 628.3;
  t.classList.remove("pop", "spawn"); void t.offsetWidth; t.classList.add("spawn");
}

function gamePop() {
  game.popping = true;
  game.count++; game.times.push((performance.now() - game.spawnAt) / 1000);
  $("#gameCount").textContent = game.count;
  $("#gameAvg").textContent = (game.times.reduce((a, b) => a + b, 0) / game.times.length).toFixed(1) + " 秒";
  $("#gameTarget").classList.add("pop");
  const burst = $("#gameBurst"), [x, y] = game.pos, R = gameR();
  for (let i = 0; i < 14; i++) {              // 光の粒が外に散る
    const dot = document.createElement("i"), a = (i / 14) * 2 * Math.PI + Math.random() * 0.3, d = R * (0.9 + Math.random() * 0.6);
    Object.assign(dot.style, { left: x + "px", top: y + "px" });
    burst.appendChild(dot);
    requestAnimationFrame(() => { dot.style.transform = `translate(${Math.cos(a) * d}px, ${Math.sin(a) * d}px) scale(.3)`; dot.style.opacity = 0; });
    game.timers.push(setTimeout(() => dot.remove(), 650));
  }
  game.timers.push(setTimeout(() => { gameSpawn(); game.popping = false; }, GAME_POP_MS));
}

ENTER.game = async () => {
  if (!(await ensureCamera())) { location.hash = "#/check"; return; }
  await sleep(300);                          // 全画面への切り替えが落ち着くまで
  Object.assign(game, { count: 0, times: [], prog: 0, last: performance.now(), popping: false, pos: null });
  $("#gameCount").textContent = "0"; $("#gameAvg").textContent = "–";
  gameSpawn();
  startLoop((r) => {
    const now = performance.now(), dt = Math.min(200, now - game.last); game.last = now;
    $("#gameInfo").innerHTML = `モデル: ${r.model || state.info.model}<br>${r.calibrated ? "キャリブ済み" : "キャリブなし"}　/　当たりの範囲: 半径 ${GAME_HIT_CM} cm`;
    const dotEl = $("#gameDot");
    if (!r.gaze) {
      dotEl.hidden = true; game.prog = Math.max(0, game.prog - 2 * dt);
      $("#gameMsg").textContent = !r.face ? "顔が見つかりません。カメラの正面を向いてください" : "先にキャリブレーションをしてください（Esc で戻る）";
      return;
    }
    $("#gameMsg").textContent = "ターゲットを見つめると消えます（Esc で終了）";
    const [cx, cy] = screenToClient(r.gaze[0], r.gaze[1]);
    dotEl.hidden = false; Object.assign(dotEl.style, { left: cx + "px", top: cy + "px" });
    if (game.popping || !game.pos) return;
    const inside = Math.hypot(cx - game.pos[0], cy - game.pos[1]) <= gameR();
    game.prog = inside ? game.prog + dt : Math.max(0, game.prog - 2 * dt);
    $("#gameCharge").style.strokeDashoffset = 628.3 * (1 - Math.min(1, game.prog / GAME_DWELL_MS));
    if (game.prog >= GAME_DWELL_MS) gamePop();
  });
};
LEAVE.game = () => { stopLoop(); game.timers.forEach(clearTimeout); game.timers = []; $("#gameBurst").innerHTML = ""; exitFs(); };

// ── 視線キーボード ─────────────────────────────
// 既定は QWERTY（画面いっぱいの大きなキー。1キー 約2.9 cm 幅で平均誤差と同じくらいなので、光ったキーを見て視線を合わせ直す前提）。
// 「かな」で、横4×縦3の大きなキーを2段階で選ぶひらがな入力に切り替え（行を選ぶ → 文字を選ぶ）。「ABC」で QWERTY に戻る。
// キーを KB_DWELL_MS 見つめると決定。決定の直後 KB_COOLDOWN_MS は次を受け付けない（同じ場所の新しいキーを続けて押さないため）。
const KB_DWELL_MS = 800, KB_COOLDOWN_MS = 700;
const KB_ROWS = [["あ", "あいうえお"], ["か", "かきくけこ"], ["さ", "さしすせそ"], ["た", "たちつてと"], ["な", "なにぬねの"], ["は", "はひふへほ"],
                 ["ま", "まみむめも"], ["や", "やゆよ"], ["ら", "らりるれろ"], ["わ", "わをんー"]];
const kb = { text: "", mode: "qwerty", level: null, keys: [], hot: -1, prog: 0, last: 0, cool: 0 };

async function startKeyboard() { await enterFs(); location.hash = "#/keyboard"; }
$("#btnKbZero").addEventListener("click", startKeyboard);
$("#btnKbDone").addEventListener("click", startKeyboard);
infoReady.then(() => { $("#btnKbZero").hidden = !state.info.zero_shot; });

function kbLayout() {
  // keys: [{label, sub, cls, act, pos}]。pos = [行, 開始列, 列数]（QWERTY は 20列のグリッドで半キーずらす）
  const keys = [];
  const grid = $("#kbGrid");
  if (kb.mode === "qwerty") {
    grid.classList.add("qwerty");
    const rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"], offs = [1, 2, 4];
    rows.forEach((row, r) => [...row].forEach((ch, i) =>
      keys.push({ label: ch, pos: [r + 1, offs[r] + 2 * i, 2], act: () => { kb.text += ch; } })));
    keys.push({ label: "かな", cls: "fn", pos: [4, 1, 3], act: () => { kb.mode = "kana"; kb.level = null; } });
    keys.push({ label: "スペース", cls: "fn space", pos: [4, 4, 9], act: () => { kb.text += " "; } });
    keys.push({ label: "⌫", sub: "1文字消す", cls: "fn", pos: [4, 13, 4], act: () => { kb.text = [...kb.text].slice(0, -1).join(""); } });
    keys.push({ label: "✕", sub: "全部消す", cls: "fn", pos: [4, 17, 4], act: () => { kb.text = ""; } });
  } else {
    grid.classList.remove("qwerty");
    if (kb.level === null) {
      KB_ROWS.forEach(([head, chars], i) => keys.push({ label: head, sub: chars, act: () => { kb.level = i; } }));
      keys.push({ label: "⌫", sub: "1文字消す", cls: "fn", act: () => { kb.text = [...kb.text].slice(0, -1).join(""); } });
      keys.push({ label: "ABC", sub: "英字に切り替え", cls: "fn", act: () => { kb.mode = "qwerty"; } });
    } else {
      [...KB_ROWS[kb.level][1]].forEach((ch) => keys.push({ label: ch, act: () => { kb.text += ch; kb.level = null; } }));
      while (keys.length < 11) keys.push({ cls: "blank" });
      keys.push({ label: "↩", sub: "戻る", cls: "fn", act: () => { kb.level = null; } });
    }
    while (keys.length < 12) keys.push({ cls: "blank" });
  }
  kb.keys = keys;
  grid.innerHTML = keys.map((k) => {
    const st = k.pos ? ` style="grid-row:${k.pos[0]};grid-column:${k.pos[1]} / span ${k.pos[2]}"` : "";
    return `<div class="kb-key ${k.cls || ""}"${st}>${k.label || ""}${k.sub ? `<small>${k.sub}</small>` : ""}<i class="fill"></i></div>`;
  }).join("");
  kb.hot = -1; kb.prog = 0;
}
function kbRenderText() {
  $("#kbText").innerHTML = kb.text ? `${kb.text.replace(/[<>&]/g, "")}<span class="caret"></span>`
    : `<span class="kb-placeholder">見つめて入力してみましょう</span>`;
}

ENTER.keyboard = async () => {
  if (!(await ensureCamera())) { location.hash = "#/check"; return; }
  await sleep(300);
  Object.assign(kb, { mode: "qwerty", level: null, hot: -1, prog: 0, last: performance.now(), cool: 0 });
  kbLayout(); kbRenderText();
  startLoop((r) => {
    const now = performance.now(), dt = Math.min(200, now - kb.last); kb.last = now;
    $("#kbInfo").innerHTML = `モデル: ${r.model || state.info.model}<br>${r.calibrated ? "キャリブ済み" : "キャリブなし"}　/　0.8秒見つめて決定`;
    const dot = $("#kbDot");
    if (!r.gaze) {
      dot.hidden = true;
      $("#kbMsg").textContent = !r.face ? "顔が見つかりません。カメラの正面を向いてください" : "先にキャリブレーションをしてください（Esc で戻る）";
      return;
    }
    $("#kbMsg").textContent = "キーを見つめると選べます（Esc で終了）";
    const [cx, cy] = screenToClient(r.gaze[0], r.gaze[1]);
    dot.hidden = false; Object.assign(dot.style, { left: cx + "px", top: cy + "px" });
    const els = $$("#kbGrid .kb-key");
    let hot = els.findIndex((el, i) => { const b = el.getBoundingClientRect(); return kb.keys[i].act && cx >= b.left && cx <= b.right && cy >= b.top && cy <= b.bottom; });
    if (now < kb.cool) hot = -1;
    if (hot !== kb.hot) { if (kb.hot >= 0 && els[kb.hot]) { els[kb.hot].classList.remove("hot"); els[kb.hot].querySelector(".fill").style.width = "0"; } kb.hot = hot; kb.prog = 0; }
    if (hot < 0) return;
    els[hot].classList.add("hot");
    kb.prog += dt;
    els[hot].querySelector(".fill").style.width = Math.min(100, kb.prog / KB_DWELL_MS * 100) + "%";
    if (kb.prog >= KB_DWELL_MS) {
      kb.keys[hot].act();
      kb.cool = now + KB_COOLDOWN_MS;
      kbLayout(); kbRenderText();
      const el = $$("#kbGrid .kb-key")[hot]; if (el) el.classList.add("done");
    }
  });
};
LEAVE.keyboard = () => { stopLoop(); exitFs(); };

window.addEventListener("hashchange", route);
route();
