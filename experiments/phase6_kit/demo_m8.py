"""M8 視線推定デモ（MacBook Air 13" M4 実機、CPU のみ）。撮影キットの部品を使うデモ専用アプリで、記録は残さない。

  cd ~/tgaze/experiments/phase6_kit && .venv/bin/python demo_m8.py              # 凍結 M8
  cd ~/tgaze/experiments/phase6_kit && .venv/bin/python demo_m8.py --eyepca     # M8 + EyePCA16（目の画像も使う、副次候補）

凍結 M8（model/final_cpu_model.onnx）か、--eyepca / --model で別モデル（models_alt/*）を使って毎フレーム推定し、その場で取ったキャリブ点で補正する。
補正は本番評価と同じ事前登録の方式（config.calibration.models: 5/9点は λ=25 cm² の affine）。
画面上の位置は全画面キャンバス基準なので、起動すると全画面になる。

キー:
  C  5点キャリブ（中央＋四隅 ±0.7、約12秒）      9  9点キャリブ（約20秒）
  T  精度チェック（ランダムな点を見る → 誤差 cm を表示。Space で次、Esc で戻る）
  S  平滑化 ON/OFF（One Euro、cm 空間）           D  補正前の生の推定（橙）も表示 ON/OFF
  R  キャリブをリセット（補正なしに戻す）         Q / Esc  終了
  G  マス目モード（見ているマスが光る。発表動画用）のオン・オフ
  1〜6 または画面下のボタン: モデルを切り替え（切り替えるとキャリブはリセット。補正はモデルごとに別物のため）
"""
import argparse
import os
import random
import sys
import time

import numpy as np

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT_DIR)
import common as C  # noqa: E402
from collect import ESC, SPACE, Camera, UI  # noqa: E402

# 切り替えられるモデル（表示名, フォルダ）。None = 本番（凍結 M8、model/）。フォルダが無いものは出さない
MODELS = [("1 本番 M8（20人）", None),
          ("2 EyePCA16（20人）", "models_alt/m8_eyepca16"),
          ("3 120人・顔の点のみ", "models_alt/m8_all120"),
          ("4 120人＋EyePCA16", "models_alt/m8_eyepca16_all120"),
          ("5 1234人・顔の点", "models_alt/m8_phase8_1234"),
          ("6 1234人＋EyePCA16", "models_alt/m8_eyepca16_phase8_1234")]

# マス目モード（G）: 横3×縦2の大きなマス。視線が同じマスに GRID_DWELL_S 秒とどまると光る
GRID_COLS, GRID_ROWS, GRID_DWELL_S = 3, 2, 0.3
GRID_LABELS = [("りんご", (70, 70, 200)), ("さかな", (200, 140, 60)), ("ほし", (40, 190, 230)),
               ("くるま", (90, 160, 90)), ("ふうせん", (170, 90, 200)), ("ほん", (60, 120, 170))]

GREEN, ORANGE, WHITE, RED, GRAY = (90, 230, 90), (0, 165, 255), (235, 235, 235), (60, 60, 230), (150, 150, 150)


class OneEuro2D:
    """One Euro filter（Casiez 2012）を cm 空間の x, y に別々にかける。ゆっくり＝強く平滑、速い＝素通し（サッカード）。"""

    def __init__(self, min_cutoff=0.8, beta=0.01, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _a(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, z, dt):
        z = np.asarray(z, np.float64)
        if self.x is None:
            self.x, self.dx = z, np.zeros(2)
            return z
        dx = (z - self.x) / dt
        self.dx = self.dx + self._a(self.dc, dt) * (dx - self.dx)
        a = self._a(self.mc + self.beta * np.abs(self.dx), dt)
        self.x = self.x + a * (z - self.x)
        return self.x

    def reset(self):
        self.x = self.dx = None


class Demo:
    def __init__(self, model_dir=None):
        import m8_cpu
        self.cfg = C.load_config()
        tg = C.tgaze_dir(self.cfg)
        K, _ = m8_cpu.load_camera_matrix(self.cfg, tg)
        self.rt = m8_cpu.M8Runtime(self.cfg, tg, K)       # 顔の点・目パッチの抽出（と凍結 M8 の推定）
        # 別モデル: 推定だけ OfflineM8 に任せる（EyePCA は目パッチも入力にする。ライブでも保存データと同じ前処理）。起動時に全部読み込む
        self.models = [(name, d, m8_cpu.OfflineM8(d, K, tg) if d else None) for name, d in MODELS
                       if d is None or os.path.isdir(os.path.join(KIT_DIR, d))]
        if model_dir and not any(d == model_dir for _, d, _ in self.models):
            self.models.append((os.path.basename(os.path.normpath(model_dir)), model_dir, m8_cpu.OfflineM8(model_dir, K, tg)))
        self.lay, _ = C.ScreenLayout.detect()
        self.ui = UI(self.lay)
        self._clicked = None
        self.ui.cv2.setMouseCallback(UI.WIN, self._on_mouse)
        cam = self.cfg["camera"]
        self.cam = Camera(cam["index"], cam["width"], cam["height"])
        self.tm = self.cfg["timing"]
        self.corr = C.Correction("none")
        self.n_cal = 0
        self.smooth, self.show_raw = True, False
        self.filt = OneEuro2D()
        self._idx, self._ts_ms, self._t_prev = -1, 0, None
        self._fps_t, self._fps = [], float("nan")
        self.grid, self._grid_cache, self._grid_tile, self._grid_since = False, {}, None, 0.0
        self.select_model(next(i for i, (_, d, _) in enumerate(self.models) if d == model_dir))

    # ------------------------------------------------------------------ モデル切り替え
    def select_model(self, i):
        name, d, alt = self.models[i]
        self.model_i, self.model_name, self.alt = i, name, alt
        self.needs_patch = alt is not None and alt.eye_pca is not None
        self.corr, self.n_cal = C.Correction("none"), 0      # 補正はモデルごとに別物なのでリセット
        self.filt.reset()

    def _buttons(self):
        """画面下のモデル切り替えボタンの矩形 [(x0, y0, x1, y1), ...]"""
        n, h, gap = len(self.models), 44, 12
        w = min(300, (self.lay.cw - 60 - gap * (n - 1)) // n)
        x0, y0 = (self.lay.cw - (w * n + gap * (n - 1))) // 2, self.lay.ch - h - 18
        return [(x0 + i * (w + gap), y0, x0 + i * (w + gap) + w, y0 + h) for i in range(n)]

    def _on_mouse(self, event, x, y, flags, param):
        if event == self.ui.cv2.EVENT_LBUTTONDOWN:
            for i, (a, b, c, d) in enumerate(self._buttons()):
                if a <= x <= c and b <= y <= d:
                    self._clicked = i

    def _draw_buttons(self, img):
        """下の帯（ボタン）は選択が変わったときだけ描き、毎フレームはコピーするだけ（日本語の描画が重いため）。"""
        y0 = self._buttons()[0][1] - 4
        if getattr(self, "_btn_cache", (None,))[0] != (self.model_i, img[y0:].shape):
            cv2, strip = self.ui.cv2, img[y0:].copy()
            for i, (a, b, c, d) in enumerate(self._buttons()):
                on = i == self.model_i
                b, d = b - y0, d - y0
                cv2.rectangle(strip, (a, b), (c, d), (70, 130, 70) if on else (60, 60, 60), -1)
                cv2.rectangle(strip, (a, b), (c, d), GREEN if on else GRAY, 2)
                self.ui.text(strip, [self.models[i][0]], org=(a + 14, b + 30), size=18, color=WHITE if on else (200, 200, 200))
            self._btn_cache = ((self.model_i, img[y0:].shape), strip)
        img[y0:] = self._btn_cache[1]

    # ------------------------------------------------------------------ 推定
    def step(self):
        """新しいフレームを1枚処理。(生の uv, 補正後の uv, 表示用 uv) か、顔が無ければ None。"""
        fr, ts, idx = self.cam.latest(self._idx)
        if idx == self._idx:
            return None
        self._idx = idx
        self._ts_ms = max(self._ts_ms + 1, int(ts * 1000))     # MediaPipe VIDEO モードは単調増加が必要
        out = self.rt.process(fr, self._ts_ms, want_patch=self.needs_patch)
        now = time.monotonic()
        self._fps_t = [t for t in self._fps_t if t > now - 2.0] + [now]
        self._fps = (len(self._fps_t) - 1) / (self._fps_t[-1] - self._fps_t[0]) if len(self._fps_t) > 2 else float("nan")
        if not out["ok"]:
            return None
        raw = np.asarray(out["uv"], np.float64)
        if self.alt is not None:
            patch = None if out["patch"] is None else out["patch"][None].astype(np.float64)
            if self.needs_patch and patch is None:
                return None
            uv, ok = self.alt.predict_lms(out["lms"][None], patch)
            if not ok[0]:
                return None
            raw = uv[0]
        cal = np.asarray(self.corr.apply(raw), np.float64)
        shown = cal
        if self.smooth:
            dt = 1 / 30 if self._t_prev is None else max(1e-3, now - self._t_prev)
            shown = self.filt(C.uv_to_cm(cal), dt) / C.HALF_CM
        self._t_prev = now
        return raw, cal, shown

    def _px(self, uv):
        x, y = self.lay.uv_to_canvas(uv)
        return int(np.clip(x, 0, self.lay.cw - 1)), int(np.clip(y, 0, self.lay.ch - 1))

    def _cursor(self, img, uv, color, r_mm=6.0):
        cv2 = self.ui.cv2
        r = int(r_mm * self.lay.px_per_mm())
        cv2.circle(img, self._px(uv), r, color, 3, cv2.LINE_AA)
        cv2.circle(img, self._px(uv), 4, color, -1, cv2.LINE_AA)

    def _target(self, img, uv, scale=1.0, color=WHITE):
        cv2, ppm, t = self.ui.cv2, self.lay.px_per_mm(), self.cfg["target"]
        c = self._px(uv)
        cv2.circle(img, c, int(t["outer_radius_mm"] * ppm * scale), color, max(1, int(t["ring_width_mm"] * ppm)), cv2.LINE_AA)
        cv2.circle(img, c, int(t["center_dot_radius_mm"] * ppm), tuple(t["dot_color_bgr"]), -1, cv2.LINE_AA)

    def _hud(self, img, face):
        cal = f"{self.n_cal}点キャリブ済み" if self.n_cal else "キャリブなし（zero-shot）"
        lines = [f"{self.model_name}  |  {cal}  |  平滑化 {'ON' if self.smooth else 'OFF'}  |  "
                 f"{self._fps:4.1f} FPS  |  {'顔あり' if face else '顔が見つかりません'}",
                 "C: 5点キャリブ   9: 9点キャリブ   T: 精度チェック   S: 平滑化   D: 生の推定   R: リセット   G: マス目   1〜6: モデル切替   Q: 終了"]
        band = img[0:80]                    # 文字の帯だけを変換する（ui.text は渡した配列全体を PIL に変換するため）
        self.ui.text(band, lines, org=(30, 36), size=18, color=WHITE if face else RED, step=28)

    # ------------------------------------------------------------------ 1点ぶんの記録（キャリブ・精度チェック共通）
    def look_at(self, uv, label):
        """ターゲットを縮めながら出し → 静止 → 記録。記録中の補正前 uv の中央値を返す（取れなければ None）。Esc で 'esc'。"""
        move, settle, rec = self.tm["move_ms"] / 1e3, self.tm["settle_ms"] / 1e3, self.tm["record_ms"] / 1e3
        shrink = self.cfg["target"]["shrink_from_mm"] / self.cfg["target"]["outer_radius_mm"]
        t0, got = time.monotonic(), []
        while True:
            el = time.monotonic() - t0
            if el > move + settle + rec:
                break
            res = self.step()
            if res is not None and el > move + settle:
                got.append(res[0])
            img = self.ui.blank(self.cfg["target"]["background_gray"])
            s = shrink + (1 - shrink) * min(1.0, el / move)
            self._target(img, uv, s, GREEN if el > move + settle else WHITE)
            self.ui.text(img, [label], org=(30, 36), size=18, color=GRAY)
            self.ui.show(img)
            k = self.ui.key(1)
            if k in (ESC, ord("q")):
                return "esc"
        return np.median(np.array(got), 0) if len(got) >= 5 else None

    def calibrate(self, n):
        pts = C.calibration_points(self.cfg, n)
        order = random.sample(pts, len(pts))
        img = self.ui.blank(30)
        self.ui.text(img, [f"{n}点キャリブ", "", "点が出たら、中心の赤い点を自然に見てください（頭は動いてもかまいません）。",
                           "点が緑になっている間に記録します。", "", "Space: 開始    Esc: やめる"],
                     org=(int(self.lay.cw * 0.15), int(self.lay.ch * 0.35)), size=26)
        if not self.ui.wait_space(img):
            return
        cal = []
        for i, p in enumerate(order):
            m = self.look_at(p, f"キャリブ {i + 1}/{len(order)}")
            if isinstance(m, str):
                return
            if m is not None:
                cal.append((p, m))
        try:
            self.corr = C.fit_correction(self.cfg, n, cal)
            self.n_cal = n
        except ValueError as e:
            print(f"[demo] キャリブ失敗: {e}")
        self.filt.reset()

    def accuracy_check(self):
        grid = [(u, v) for u in self.cfg["grids"]["evaluation_uv"] for v in self.cfg["grids"]["evaluation_uv"]]
        errs = []
        while True:
            uv = random.choice(grid)
            m = self.look_at(uv, "精度チェック: 中心の赤い点を見てください")
            if isinstance(m, str):
                return
            if m is None:
                continue
            p = np.asarray(self.corr.apply(m))
            e = float(C.cm_err(p, uv))
            errs.append(e)
            img = self.ui.blank(self.cfg["target"]["background_gray"])
            self._target(img, uv)
            self._cursor(img, p, GREEN)
            self.ui.cv2.line(img, self._px(uv), self._px(p), GRAY, 1, self.ui.cv2.LINE_AA)
            self.ui.text(img, [f"誤差 {e:.1f} cm   （{len(errs)}回の平均 {np.mean(errs):.1f} cm、中央値 {np.median(errs):.1f} cm）",
                               "白＝見た点、緑＝推定（補正後の記録中の中央値）   Space: 次   Esc: 戻る"],
                         org=(30, 36), size=20, step=30)
            self.ui.show(img)
            while True:
                k = self.ui.key(30)
                if k == SPACE:
                    break
                if k in (ESC, ord("q")):
                    return

    # ------------------------------------------------------------------ マス目モード
    def _grid_rects(self):
        x0, x1, y0, y1, gap = 40, self.lay.cw - 40, 100, self.lay.ch - 100, 28
        w, h = (x1 - x0 - gap * (GRID_COLS - 1)) // GRID_COLS, (y1 - y0 - gap * (GRID_ROWS - 1)) // GRID_ROWS
        return [(x0 + c * (w + gap), y0 + r * (h + gap), x0 + c * (w + gap) + w, y0 + r * (h + gap) + h)
                for r in range(GRID_ROWS) for c in range(GRID_COLS)]

    def _grid_base(self, lit):
        """マス目の絵（lit 番のマスが光った状態）。状態ごとに1回だけ描いて使い回す（日本語の描画が重いため）。"""
        if lit not in self._grid_cache:
            cv2, img = self.ui.cv2, self.ui.blank(25)
            for i, (a, b, c, d) in enumerate(self._grid_rects()):
                name, col = GRID_LABELS[i % len(GRID_LABELS)]
                on = i == lit
                fill = tuple(int(v) for v in col) if on else (48, 48, 48)
                cv2.rectangle(img, (a, b), (c, d), fill, -1)
                cv2.rectangle(img, (a, b), (c, d), WHITE if on else (90, 90, 90), 8 if on else 2)
                size = 44
                tile = img[b:d, a:c]
                self.ui.text(tile, [name], org=((c - a) // 2 - size * len(name) // 2, (d - b) // 2 + size // 2),
                             size=44, color=WHITE if on else (170, 170, 170))
            self._grid_cache[lit] = img
        return self._grid_cache[lit].copy()

    def _grid_frame(self, res):
        now = time.monotonic()
        tile = None
        if res is not None:
            x, y = self._px(res[2])
            tile = next((i for i, (a, b, c, d) in enumerate(self._grid_rects()) if a <= x <= c and b <= y <= d), None)
        if tile != self._grid_tile:
            self._grid_tile, self._grid_since = tile, now
        lit = tile if tile is not None and now - self._grid_since >= GRID_DWELL_S else -1
        img = self._grid_base(lit)
        if res is not None:      # 視線の点は小さく薄く（主役は光るマス）
            cv2 = self.ui.cv2
            cv2.circle(img, self._px(res[2]), 10, (200, 200, 200), 2, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ メイン
    def run(self):
        try:
            while True:
                res = self.step()     # cam.latest は新しいフレームを待つので、None はほぼ「顔が無い」
                if self.grid:
                    img = self._grid_frame(res)
                else:
                    img = self.ui.blank(25)
                    if res is not None:
                        raw, _, shown = res
                        if self.show_raw:
                            self._cursor(img, raw, ORANGE, 4.0)
                        self._cursor(img, shown, GREEN)
                self._hud(img, res is not None)
                self._draw_buttons(img)
                self.ui.show(img)
                k = self.ui.key(1)
                if k in (ESC, ord("q")):
                    break
                if self._clicked is not None:
                    self.select_model(self._clicked)
                    self._clicked = None
                elif ord("1") <= k <= ord(str(min(len(self.models), 8))):
                    self.select_model(k - ord("1"))
                if k == ord("c"):
                    self.calibrate(5)
                    self._clicked = None       # キャリブ中のクリックは無視
                elif k == ord("9"):
                    self.calibrate(9)
                    self._clicked = None
                elif k == ord("t"):
                    self.accuracy_check()
                    self._clicked = None
                elif k == ord("s"):
                    self.smooth = not self.smooth
                    self.filt.reset()
                elif k == ord("d"):
                    self.show_raw = not self.show_raw
                elif k == ord("g"):
                    self.grid, self._grid_tile = not self.grid, None
                    if self.grid:            # 光った状態の絵を先に全部作っておく（初めて光るときのカクつき防止、約0.4秒）
                        for i in range(-1, GRID_COLS * GRID_ROWS):
                            self._grid_base(i)
                elif k == ord("r"):
                    self.corr, self.n_cal = C.Correction("none"), 0
                    self.filt.reset()
        finally:
            self.cam.close()
            self.ui.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eyepca", action="store_true", help="M8 + EyePCA16（models_alt/m8_eyepca16）を使う")
    ap.add_argument("--model", default=None, help="models_alt/ 以下などのモデルフォルダ（凍結 M8 と同じ形式）")
    ap.add_argument("--grid", action="store_true", help="マス目モードで始める（G キーと同じ）")
    a = ap.parse_args()
    demo = Demo("models_alt/m8_eyepca16" if a.eyepca else a.model)
    if a.grid:
        demo.grid = True
        for i in range(-1, GRID_COLS * GRID_ROWS):
            demo._grid_base(i)
    demo.run()
