// 録画：顔（カメラ映像）と画面（このタブ）を、この PC の webapp/recordings/<日時>/ に保存する。
// - 顔：カメラが使えるようになったら自動で始める
// - 画面：ブラウザの決まりで、人がクリックした瞬間にしか始められないので、最初のクリックで「画面全体」の録画の許可を求める
//   「このタブ」を録画すると、Chrome はタブを全画面にしたとき中身を録画の大きさに縮めて左右を黒く埋める（キャリブで黒い縦帯が出た原因）。
//   なので画面全体（monitor）を録画する。タブを選ばれたら止めて、画面全体を選び直してもらう
// - 1秒ごとに断片をサーバに送って追記する（途中でタブを閉じても、そこまでは残る）
// - URL に ?rec=0 を付けると録画しない
(() => {
  if (new URLSearchParams(location.search).get("rec") === "0") return;
  const rec = { sid: null, starting: null, face: null, screen: null, stopped: false };

  // 画面の隅の表示（録画中・停止ボタン）
  const bar = document.createElement("div");
  Object.assign(bar.style, { position: "fixed", left: "8px", bottom: "6px", zIndex: 9999, font: "12px sans-serif",
    color: "#13194a", background: "rgba(255,255,255,.85)", border: "1px solid #e3e7f3", borderRadius: "8px",
    padding: "3px 8px", display: "flex", gap: "8px", alignItems: "center" });
  const label = document.createElement("span");
  const stopBtn = document.createElement("button");
  stopBtn.textContent = "録画を停止";
  Object.assign(stopBtn.style, { font: "12px sans-serif", border: "1px solid #c9cede", borderRadius: "6px", background: "#fff", cursor: "pointer" });
  bar.append(label, stopBtn);
  const show = () => {
    const parts = [];
    if (rec.face) parts.push("顔");
    if (rec.screen) parts.push("画面");
    label.innerHTML = rec.stopped ? "録画を停止しました" + (rec.sid ? `（recordings/${rec.sid}）` : "")
      : parts.length ? `<span style="color:#ff4d5e">●</span> 録画中（${parts.join("・")}）`
      : "録画の準備中（画面の録画はクリックで開始）";
    stopBtn.hidden = rec.stopped || !parts.length;
  };
  document.addEventListener("DOMContentLoaded", () => { document.body.appendChild(bar); show(); });
  if (document.body) { document.body.appendChild(bar); show(); }

  async function meta(obj) {
    if (!rec.sid) return;
    try { await fetch(`/api/rec/meta?sid=${rec.sid}`, { method: "POST", body: JSON.stringify(obj) }); } catch (e) { /* noop */ }
  }
  async function session() {
    if (rec.sid) return rec.sid;
    if (!rec.starting) {
      rec.starting = fetch("/api/rec/start", { method: "POST", body: JSON.stringify({
        event: "session_start", wall_ms: Date.now(), perf_origin_ms: performance.timeOrigin,
        model: (typeof state !== "undefined" && state.info) ? state.info.model : null,
        screen: [screen.width, screen.height], dpr: devicePixelRatio, ua: navigator.userAgent }) })
        .then((r) => r.json()).then((j) => { rec.sid = j.sid; return j.sid; });
    }
    return rec.starting;
  }
  function pickType() {
    // mp4（Chrome の MediaRecorder）は 1秒ごとの断片を追記すると途中から崩れたので（2026-10-29 に確認）、webm を優先する
    for (const t of ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4;codecs=avc1", "video/mp4"]) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
    }
    return "";
  }
  async function startRecorder(kind, stream, bps) {
    const sid = await session();
    const type = pickType(), ext = type.startsWith("video/mp4") ? "mp4" : "webm";
    const mr = new MediaRecorder(stream, type ? { mimeType: type, videoBitsPerSecond: bps } : { videoBitsPerSecond: bps });
    let chain = Promise.resolve();          // 断片を順番どおりに送る
    mr.ondataavailable = (e) => {
      if (!e.data || !e.data.size) return;
      const blob = e.data;
      chain = chain.then(() => fetch(`/api/rec/chunk?sid=${sid}&kind=${kind}&ext=${ext}`, { method: "POST", body: blob })).catch(() => {});
    };
    mr.onstop = () => { chain.then(() => meta({ event: "stop", kind, wall_ms: Date.now() })); };
    mr.start(1000);
    const s = stream.getVideoTracks()[0].getSettings();
    meta({ event: "start", kind, wall_ms: Date.now(), perf_ms: performance.now(), mime: mr.mimeType, width: s.width, height: s.height, fps: s.frameRate });
    rec[kind] = mr;
    show();
    return mr;
  }

  // 顔：カメラの映像（state.stream）が用意できたら自動で開始
  const faceTimer = setInterval(() => {
    if (rec.stopped) return clearInterval(faceTimer);
    if (typeof state !== "undefined" && state.stream && !rec.face) {
      clearInterval(faceTimer);
      startRecorder("face", state.stream, 8e6).catch((e) => console.warn("face rec", e));
    }
  }, 500);

  // 画面：最初のクリックで「画面全体」の録画の許可を求める（断られたら顔だけ）
  let asked = false;
  document.addEventListener("pointerdown", async (ev) => {
    if (asked || rec.stopped || ev.target === stopBtn) return;
    asked = true;
    try {
      const ds = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: "monitor", frameRate: 30, width: { max: 1920 }, height: { max: 1250 } }, audio: false,
        preferCurrentTab: false, selfBrowserSurface: "exclude", monitorTypeSurfaces: "include", surfaceSwitching: "exclude" });
      const surface = ds.getVideoTracks()[0].getSettings().displaySurface;
      if (surface && surface !== "monitor") {          // タブやウィンドウを選ばれたら、全画面で黒い帯が出るので使わない
        ds.getTracks().forEach((t) => t.stop());
        asked = false;                                  // 次のクリックでもう一度聞く
        label.textContent = "画面の録画は「画面全体」を選んでください（タブを選ぶと全画面で左右が黒くなります）。次のクリックでもう一度聞きます";
        return;
      }
      ds.getVideoTracks()[0].addEventListener("ended", () => { if (rec.screen && rec.screen.state !== "inactive") rec.screen.stop(); });
      await startRecorder("screen", ds, 6e6);
    } catch (e) {
      meta({ event: "screen_declined", wall_ms: Date.now(), error: String(e) });
      label.textContent = "画面の録画は許可されませんでした（顔だけ録画）";
    }
  }, true);

  stopBtn.addEventListener("click", () => {
    rec.stopped = true;
    for (const k of ["face", "screen"]) {
      const mr = rec[k];
      if (mr && mr.state !== "inactive") {
        mr.stop();
        if (k === "screen") mr.stream.getTracks().forEach((t) => t.stop());   // 画面共有を終える（カメラは計測に使うので止めない）
      }
    }
    show();
  });
})();
