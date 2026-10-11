# Phase 6 パイロット結果（subject_00、2026-10-12、MacBook Air 13" M4 実機）

1人分のパイロットです。精度の数字は参考値で、本番（5人）の前に実験系が正しく動くかを確かめるためのもの。
顔画像・ランドマーク・目パッチ（試行ごとの npz と代表フレーム JPEG）は公開リポジトリには入れていません。

## 実験系チェック（`pilot_check_subject_00.md`）: 12項目すべて合格

- 1920x1080、実測K（host 一致）、mirrored=True、センターフレーム off
- 画面寸法を定規で確認（ノッチ帯を除く全画面 1470x924 pt、0.1977 mm/pt）
- 方向チェック合格: 右−左 24.3 cm、上−下 16.0 cm（基準 2 cm / 1 cm 以上）
- CPU のみ（ONNX Runtime CPUExecutionProvider、MediaPipe CPU、torch 未ロード）、M8 の sha256 は Phase 5 と一致
- 30.0 FPS（MediaPipe 5.8 ms、M8 ONNX 0.06 ms）
- 試行: キャリブ 18/18 VALID、評価 66 VALID / 9 RETAKE_RECOMMENDED / 1 INVALID

## 精度（評価 75 試行、cm。`calibration_comparison.csv` と `trials.csv` から）

| 補正 | NATURAL | EYE_DOMINANT | HEAD_ALLOWED | 全体 |
|---|---:|---:|---:|---:|
| 0点（zero-shot） | 2.74 | 4.10 | 2.83 | 3.22 |
| 1点 | 2.65 | 5.24 | 3.61 | 3.83 |
| 3点 | 3.06 | 6.59 | 4.23 | 4.63 |
| 5点 | 2.42 | 5.39 | 3.41 | 3.74 |
| 9点 | 2.25 | 5.15 | 3.25 | 3.55 |
| 現行 tgaze（16D+目パッチPCA, 9点） | 2.41 | 4.10 | 3.19 | 3.23 |

- NATURAL では zero-shot 2.7 cm → 9点 2.25 cm。合成データの M8（1.90 cm）に近く、現行 tgaze より少し良い。
- EYE_DOMINANT（頭を止めて目だけ）は 4〜5 cm に崩れ、補正でむしろ悪化。M8 は頭の向きに頼り、
  目だけの動きを小さく見積もっている（Phase 7 の「目の向きの推定がボトルネック」と一致）。
  キャリブは NATURAL で取るので、その癖を学んでしまい EYE_DOMINANT でずれる。

## 頭の向き（yaw）の注意

`trials.csv` などの `head_yaw_deg` は tgaze rich16d（6点 solvePnP）の値で、実際より大きく出る。

| 方法 | yaw |
|---|---:|
| M8 自身の Kabsch 回転（40点） | +3〜6° |
| rich16d solvePnP | +14〜22° |

- rich16d は Kabsch に対して約 +12° のオフセットと 2.7 倍の傾き（相関 0.78）。K の有無や mirrored を変えてもほぼ同じ。
- Phase 7 でも合成データで rich16d の角度は正解から平均 11°（Kabsch は 1.6°）ずれていた。
- M8 は head_pose 入力 12 次元のうち 3 次元に rich16d の pitch/yaw/roll を使っているので、synthetic→real ギャップの一因の可能性がある（未検証）。
  生ランドマークと R_head は全フレーム保存しているので、撮り直さずにオフラインで検証できる。

## 補足

- `analyze.py --include-pilot` は `report.md` を書く所で止まる（パイロットには NATURAL 2回目のブロックがなく
  `C0_natural_retest_cm` が無いため）。CSV と JSON はその前に書き出されている。本番の被験者では起きない。
- λ はキャリブだけの LOO で 100〜400 や bias_scale が少し良かったが（1人分なので）事前登録どおり 25 のまま。
