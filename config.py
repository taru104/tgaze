"""アプリの特徴・キャリブ モード切り替え。

MODE = '7d'     → 7D + H1Calibration。正面特化(実機1.4cm・使用感◎)。横向きは崩れる。
MODE = '16d'    → 16D + HuberCalibration。横向きが崩れにくい。正面は7Dよりやや落ちる。
MODE = 'hybrid' → 正面(|yaw|<閾値)=7D H1, 横向き(>=閾値)=16D Huber を自動ハード切替。
                  キャリブでは 7D と 16D の両方を並列学習する。正面の使用感 + 横向きの崩れにくさを両取り。

7Dに戻すのも 1行(MODE='7d')。
"""
MODE = '16d'                    # '7d' | '16d' | 'hybrid'  ← 16Dで実機検証中(研究でオフライン全姿勢2.2-2.7cm。要実機確認)
HYBRID_YAW_THRESH_DEG = 10.0    # hybrid: この角度(deg)以上の |yaw| で 16D に切替

# 画像クリップ(目パッチ)版をデフォルトにするか。
# True  = 16D幾何 + 目パッチ(48x32 CLAHE)PCA16。実機で「変なところに飛びにくい」安定性◎(ユーザ選好2026-07-24)。
# False = 現行の16D幾何のみ(点精度LOOは僅かに上だが暴走しやすい)。
# 16Dのみに戻すのも1行(USE_APPEARANCE=False) or `python main_16d.py`。
USE_APPEARANCE = True
