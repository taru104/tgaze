// スライド用の画面撮影：?mode=market / keyboard。撮る直前の状態を作り、注視点の位置を /log に送る
(function () {
  const mode = (location.search.match(/mode=(\w+)/) || [, "market"])[1];
  const send = (o) => fetch("/log?" + encodeURIComponent(JSON.stringify(o)));
  const center = (el) => {
    const b = el.getBoundingClientRect();
    return { u: (b.left + b.width / 2) / 1280, v: (b.top + b.height / 2) / 832,
      box: [b.left / 1280, b.top / 832, b.right / 1280, b.bottom / 832] };
  };
  const css = document.createElement("style");
  css.textContent = "html,body{width:1280px;height:832px;overflow:hidden} .kb{position:absolute!important;left:0;top:0;width:1280px;height:832px;inset:auto!important} #topbar,#panel,#measurebar{display:none!important} *{transition:none!important;animation:none!important}" +
    (mode === "market" ? ".page.measure{position:absolute;left:0;top:0;padding:0!important;display:block!important;max-width:none}" +
      ".page.measure{width:1280px;height:832px}.stage{position:absolute!important;left:0;top:0;width:1280px;height:832px;border:0!important;border-radius:0!important}#demo{opacity:1!important}" : "");
  document.head.appendChild(css);
  const hideRecBadge = () => $$("body > div").forEach((d) => { if (/録画/.test(d.textContent)) d.style.display = "none"; });
  function go() {
    document.querySelectorAll("main > section, body > section").forEach((s) => { s.hidden = true; });
    hideRecBadge();
    if (mode === "market") {
      $("#page-measure").hidden = false;
      send({ mode, target: "スマートウォッチ", ...center($('#demo [data-area="スマートウォッチ"] .shop-img')), W: innerWidth, H: innerHeight, dpr: devicePixelRatio });
    } else {
      $("#page-keyboard").hidden = false;
      kb.mode = "qwerty"; kbLayout(); kb.text = "HELLO"; kbRenderText();
      const g = $$("#kbGrid .kb-key").find((e) => e.firstChild && e.firstChild.textContent === "G");
      g.classList.add("hot"); g.querySelector(".fill").style.width = "65%";
      send({ mode, target: "G", ...center(g), W: innerWidth, H: innerHeight, dpr: devicePixelRatio });
    }
  }
  go(); setTimeout(go, 300); setTimeout(go, 1200);
  setInterval(hideRecBadge, 100);   // 左下の「録画の準備中」の帯（rec.js が後から足す）を隠し続ける
})();
