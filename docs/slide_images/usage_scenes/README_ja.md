# Tgaze が使われる場面（本選スライド用のイメージ）

スライド用のイメージです。**学習には使っていません**（学習データの生成も学習もしていません）。

| 場面 | identity | 服・髪 | 画面 | 注視点 |
|---|---|---|---|---|
| 1. 家の机で EC サイトを見る | 247 | male_casualsuit01・short02 | tgaze market（submission の webapp） | スマートウォッチの商品画像 |
| 2. オンライン授業（予定） | 236 | female_casualsuit01・long01 | 架空の授業ページ（`src/lecture.html`） | 板書の図 |
| 3. 視線キーボード（予定） | 242 | female_casualsuit02・afro01 | 視線キーボード（submission の webapp、QWERTY・HELLO・G を注視中） | G のキー |

各 PNG と同じ名前の JSON に、identity・服・髪・注視点（画面画像上の u, v と 3D 位置）・目の位置・カメラの位置を入れています。

## 作り方（`src/`）

- `render_usage_scene.py`：Blender 4.5.14 + MPFB2。人は `generate_gaze_dataset_v3.GeneratorV3.build()` に服と靴を足したもの。
  MacBook は `create_macbook_gaze_poc.build_macbook()`（GeneratorV3 の中で作られる）を机の上（天板 72 cm）へ動かし、本体と蓋を描画に出してアルミと黒い縁を付けました。
  目は `aim_with_mpfb_ik` で注視点へ向けています（残差 0.001° 以下）。座る姿勢は脚・背骨・首の骨を回し、腕は Blender の IK で手をキーボード（場面3はひざ）に置いています。
  Cycles（GPU）、デノイズ、色設定は Standard（AgX だと画面が白っぽく抜けるため）。
  ```
  cd ~/mpfb2-gaze-generator-poc
  blender --background --python <このフォルダ>/src/render_usage_scene.py -- 1 shoulder out.png out.json 1920 1080 128
  ```
  場面は 1〜3、視点は `shoulder`（人の右うしろ・肩越し、55 mm）か `side`（横からの 3/4）。
- 画面の画像（2560×1664）：submission ブランチの `webapp/static/index.html` を Chrome（ヘッドレス）で 1280×832・倍率2 で撮影。
  playwright は入っていなかったので、`shot.js` をページに足して画面の状態を作り、注視点の要素の位置を記録しました（`targets.json`、画面の左上が原点）。
  サーバの API は使っていません。
- 注視点：スマートウォッチ u=0.380, v=0.440 ／ G のキー u=0.500, v=0.432 ／ 板書の図 u=0.494, v=0.516。

## 制限

- 人のポーズは骨を回して付けたもので、手や指の形は細かく合わせていません。
- 部屋・机・いす・小物は単純な形です。ロゴや実在の商品は入れていません。
