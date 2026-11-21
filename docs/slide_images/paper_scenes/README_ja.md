# 論文の「データの撮り方」を 3DCG で再現したイメージ（発表スライド用）

論文の撮り方を 3DCG で再現したイメージです。**論文の図の写しではありません。学習には使っていません**（学習データの生成も学習もしていません）。
置き場所・色・構図・服は、自分たちで決めました。文字のラベルは絵に入れていません（スライドで重ねる前提）。

## 場面1：GAFA（Nonaka ら、CVPR 2022）の撮り方

- 両側に本棚が並ぶ通路（幅 1.44 m、棚の高さ 2.05 m）。人（identity 228、服 male_casualsuit03、靴 shoes04）が、前の棚から 13 cm 引き出した黄色い本を見ている（ほかに黄色い本は置いていない）
- 人が着けている機材
  - メガネ型の視線計測器：細い黒い枠、目を撮る小さなカメラ（左右）、前を撮るカメラ（鼻の上）
  - 胸のカメラ：肩から胸にかけた X 字のベルト＋胸を一周するベルトで、胸に固定
- ArUco マーカー（OpenCV `DICT_4X4_50`、id 0〜6）：棚に 4 枚、床に 3 枚
- 監視カメラ 6 台：高さ 2.45〜2.6 m、人を見下ろす向き。CCTV0〜4 は 22 mm 相当の広角、CCTV5 は (a) 用に人へ近づけて 35 mm 相当
  （メガネ・胸のベルトが 960×540 でも見え、手前の本で画面が埋まらない位置。論文の写真の構図には寄せていない）
- 書き出し：(a) 監視カメラ 1 台（CCTV5、人の右前の上）からの 1 枚。(b) 6 台の絵を並べたもの

| ファイル | 中身 |
|---|---|
| `gafa_a_cctv5_1920x1080.png` / `.json` | (a) 監視カメラ CCTV5 からの 1 枚（1920×1080） |
| `gafa_b_all_cctv_960x540x6.png` / `.json` | (b) 6 台の絵（各 960×540）を 3 列 × 2 行に並べたもの。1 枚ずつは `gafa_b_tiles/` |
| `gafa_a_cctv5_preview960.png` / `.json` | (a) の最初の試し描き（直す前。CCTV5 の位置が違う） |

JSON には、元にした論文、再現した機材、ArUco マーカーの位置、監視カメラ 6 台の位置と向き、目の向きの残差を入れています。

## 作り方（`src/`）

- `render_paper_scene.py`：Blender 4.5.14 + MPFB2。人は `generate_gaze_dataset_v3.GeneratorV3.build()` に服と靴を足したもの（学習用の MacBook は消す）。
  腕・首・頭は骨を回して姿勢を付け、目は `create_macbook_gaze_poc.aim_with_mpfb_ik` で本へ向けた（残差 0.001° 以下）。
  Cycles（GPU）、デノイズ、色設定は Standard。
  ```
  cd ~/mpfb2-gaze-generator-poc
  blender --background --python <このフォルダ>/src/render_paper_scene.py -- gafa cam5 OUT_DIR 1920 1080 128     # (a)
  blender --background --python <このフォルダ>/src/render_paper_scene.py -- gafa all  OUT_DIR 960 540 128       # (b) の 6 枚
  ```
  `--nogear` を付けると、機材を外して目から本へ赤い線を引いた版になる。
- `markers/`：OpenCV（`cv2.aruco.generateImageMarker`、`DICT_4X4_50`）で作った ArUco マーカー（白い余白付き）。

## 注意

- 機材の形は、依頼文の説明（細いフレームに目を撮る小さなカメラ／ベルトで胸に固定したカメラ）から作ったもので、実在の製品の形ではありません。
- 論文の細かい条件（カメラの台数・高さ・レンズなど）は確認していません。数値はこの絵のために決めたものです。
