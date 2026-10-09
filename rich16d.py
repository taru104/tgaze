"""16D リッチ特徴の単一定義（ライブ／再処理／オフライン評価で共有）。

`rich_16d_from_lms(lms, w, h)` = 正規化ランドマーク配列(N,3) + 画像サイズ → 16D特徴。
ライブ(`features.py`)・生ログ再処理(`benchmarks/reprocess_raw_landmarks.py`)・
オフライン評価(`benchmarks/*`)が全部これを呼ぶことで、実装divergenceを防ぐ。

数式は `benchmarks/extract_rich_features.extract_rich`(画像版・全324k抽出に使用)と厳密一致。
定数値も同一（2026-07-17時点で extract_rich_features と一致することをテストで担保）。

16D の内訳:
  [0:7]   Lx,Ly,Rx,Ry, pitch, yaw, dist   （既存7D。目頭・目尻基準の虹彩位置＋頭部姿勢）
  [7]     roll
  [8:10]  L_EAR, R_EAR                     （眼開き）
  [10:12] L_iris_vert, R_iris_vert         （虹彩の眼内垂直位置; 縦視線）
  [12:14] L_iris_diam, R_iris_diam         （虹彩径/眼幅; 距離代理）
  [14:16] L_aspect, R_aspect               （虹彩アスペクト比; 横向き扁平化）
"""
import numpy as np
import cv2

# ─── landmark indices（extract_rich_features と同一値）───────────────────────
_FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0],
], dtype=np.float64)
_FACE_2D_IDX = [1, 152, 33, 263, 61, 291]
_LEFT_IRIS   = [468, 469, 470, 471, 472]
_RIGHT_IRIS  = [473, 474, 475, 476, 477]
_L_IN, _L_OUT = 133, 33
_R_IN, _R_OUT = 362, 263
_L_UP, _L_LO = 159, 145
_R_UP, _R_LO = 386, 374
_DIST = np.zeros((4, 1), dtype=np.float64)

RICH_DIM = 16

# 実測カメラ行列（任意）。None なら従来どおり f=画像幅・主点=画像中心。
# 画像サイズが一致するフレームにだけ適用する（解像度が違えば K は流用できない）。
_CAMERA_K = None   # ((w, h), K 3x3)


def set_camera_matrix(K, size, mirrored=False):
    """実測 K(3x3) を設定。size=(w,h)。mirrored=True はフレームを左右反転して使う場合（cx→w-cx）。
    K=None で解除。"""
    global _CAMERA_K
    if K is None:
        _CAMERA_K = None
        return
    K = np.array(K, dtype=np.float64)
    if mirrored:
        K[0, 2] = size[0] - K[0, 2]
    _CAMERA_K = ((int(size[0]), int(size[1])), K)


def _geo_normalize(pupil, inner, outer):
    """虹彩を目頭・目尻の中点基準にし、目の軸で回転、目幅で正規化。
    頭の平行移動に不変な『眼球内で虹彩がどこにあるか』の信号。"""
    vec = outer - inner
    length = np.linalg.norm(vec) + 1e-8
    center = (inner + outer) / 2.0
    rel = pupil - center
    ang = np.arctan2(vec[1], vec[0])
    ca, sa = np.cos(-ang), np.sin(-ang)
    rot = np.array([[ca, -sa], [sa, ca]])
    return (rot @ rel) / length


def rich_16d_from_lms(lms, w, h):
    """(N,3) 正規化ランドマーク配列 + 画像サイズ → 16D特徴(np.float32) または None。

    lms[i][0], lms[i][1] が i番目ランドマークの正規化x,y（[0,1]）であること。
    mediapipe の NormalizedLandmark を使う場合は事前に (N,3) 配列へ変換して渡す。
    """
    def P(i):
        return np.array([lms[i][0] * w, lms[i][1] * h])

    L_px = np.mean([P(i) for i in _LEFT_IRIS], axis=0)
    R_px = np.mean([P(i) for i in _RIGHT_IRIS], axis=0)

    if _CAMERA_K is not None and _CAMERA_K[0] == (int(w), int(h)):
        cam = _CAMERA_K[1]
    else:
        f = float(w)
        cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    face_2d = np.array([[lms[i][0] * w, lms[i][1] * h] for i in _FACE_2D_IDX], dtype=np.float64)
    # SQPNP は姿勢std を桁違いに安定化する(roll 76→8°)が、視線精度は改善せず合算30+が
    # 11.66→14.13cm と悪化した(2026-07-17検証)。姿勢のブレは横向き精度の主因ではない
    # (段階8の姿勢平滑と同じ結論)。ITERATIVE を維持。横向きはデータ(ETH-XGaze)で攻める。
    ok, rvec, _ = cv2.solvePnP(_FACE_3D_MODEL, face_2d, cam, _DIST,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0]) * np.pi / 180.0
    yaw   = float(angles[1]) * np.pi / 180.0
    roll  = float(angles[2]) * np.pi / 180.0
    if pitch > np.pi / 2:
        pitch = np.pi - pitch
    elif pitch < -np.pi / 2:
        pitch = -np.pi - pitch

    l_in, l_out = P(_L_IN), P(_L_OUT)
    r_in, r_out = P(_R_IN), P(_R_OUT)
    L_n = _geo_normalize(L_px, l_in, l_out)
    R_n = _geo_normalize(R_px, r_in, r_out)
    dist = float(np.linalg.norm(L_px - R_px) / w)

    l_up, l_lo = P(_L_UP), P(_L_LO)
    r_up, r_lo = P(_R_UP), P(_R_LO)
    l_width = np.linalg.norm(l_out - l_in) + 1e-8
    r_width = np.linalg.norm(r_out - r_in) + 1e-8
    l_height = np.linalg.norm(l_up - l_lo)
    r_height = np.linalg.norm(r_up - r_lo)
    L_EAR = l_height / l_width
    R_EAR = r_height / r_width
    l_eye_cy = (l_up[1] + l_lo[1]) / 2.0
    r_eye_cy = (r_up[1] + r_lo[1]) / 2.0
    L_iris_vert = (L_px[1] - l_eye_cy) / (l_height + 1e-8)
    R_iris_vert = (R_px[1] - r_eye_cy) / (r_height + 1e-8)
    l_diam = 2.0 * np.mean([np.linalg.norm(P(i) - L_px) for i in _LEFT_IRIS[1:]])
    r_diam = 2.0 * np.mean([np.linalg.norm(P(i) - R_px) for i in _RIGHT_IRIS[1:]])
    L_iris_diam = l_diam / l_width
    R_iris_diam = r_diam / r_width

    def iris_aspect(idx):
        ax1 = np.linalg.norm(P(idx[1]) - P(idx[3]))
        ax2 = np.linalg.norm(P(idx[2]) - P(idx[4]))
        lo, hi = min(ax1, ax2), max(ax1, ax2)
        return lo / (hi + 1e-8)
    L_aspect = iris_aspect(_LEFT_IRIS)
    R_aspect = iris_aspect(_RIGHT_IRIS)

    return np.array([L_n[0], L_n[1], R_n[0], R_n[1], pitch, yaw, dist,
                     roll, L_EAR, R_EAR, L_iris_vert, R_iris_vert,
                     L_iris_diam, R_iris_diam, L_aspect, R_aspect], dtype=np.float32)


def lms_to_array(mp_landmarks):
    """mediapipe の NormalizedLandmark 列 → (N,3) np.float64 配列。"""
    return np.array([[p.x, p.y, p.z] for p in mp_landmarks], dtype=np.float64)
