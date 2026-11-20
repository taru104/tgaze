# Tgaze Web アプリ（提出用：`submission` ブランチ）

Web カメラだけで「画面のどこを見ているか」を推定し、ブラウザで計測・可視化するアプリです。
推定はこの PC の中だけで行い、映像や結果を外部に送りません。

- 視線モデル（M8 系）は、Blender（MPFB2）で作った**合成データだけ**で学習した小さなニューラルネット（ONNX、約 90 KB）です。
  実在の人の写真は学習に使っていません（顔の点を取る MediaPipe Face Landmarker は Google が学習したもの）。
- 推論は **CPU のみ**（ONNX Runtime の CPUExecutionProvider）。GPU は使いません。

## 動作環境

- macOS（Apple Silicon）。開発・確認は MacBook Air 13 インチ（M4）の内蔵カメラ、1920×1080
- Python 3.10 以上（確認は 3.12）
- Google Chrome（カメラと画面録画を使うため）

同梱のカメラ行列（`results/camera_calib/1920x1080_mac/camera_matrix.json`）は、開発に使った MacBook Air の内蔵カメラで実測したものです。
別の PC でも起動はできますが（警告を表示）、推定はずれる可能性があります。

**M8 系のモデルは、この MacBook Air のカメラと画面の位置関係を Blender で再現して学習した「機種専用」のモデルです。**
別の PC では、カメラの位置・画角・画面の大きさが違うため、キャリブなしの推定は本来の精度が出ません（9 点キャリブで一部補正されます）。
本来の環境での動作は、デモ動画を参照してください。

## セットアップ

```bash
git clone -b submission https://github.com/taru104/tgaze.git
cd tgaze
python3 -m venv .venv
.venv/bin/pip install -r webapp/requirements.txt
```

## 起動

```bash
.venv/bin/python webapp/server.py --model models_alt/m8_phase8_1234
```

Chrome で **http://localhost:8000** を開き、カメラの使用を許可します。サーバはこの PC からの接続（127.0.0.1）だけを受け付けます。

`--model` で推定に使うモデルを選びます。

| `--model` | 内容 | 学習データ（すべて合成） |
|---|---|---|
| `frozen` | 本番 M8（顔の点のみ） | 20人・2万枚（学習16人） |
| `models_alt/m8_eyepca16` | 顔の点＋目の画像（PCA16） | 20人・2万枚 |
| `models_alt/m8_all120` | 顔の点のみ | 120人・5万枚 |
| `models_alt/m8_eyepca16_all120` | 顔の点＋目の画像 | 120人・5万枚 |
| `models_alt/m8_phase8_1234` | 顔の点のみ | 1234人・14万6800枚（見た目・照明などを人ごとに変更） |
| `models_alt/m8_eyepca16_phase8_1234` | 顔の点＋目の画像 | 1234人・14万6800枚 |
| （指定なし） | 旧手法（関数版 tgaze：16次元の特徴＋目の画像 PCA16、Huber 回帰）。キャリブ必須 | 学習済みモデルなし（その場のキャリブで当てはめ） |

各モデルの学習の分け方・seed の選び方は、`experiments/phase6_kit/model*/…/final_model_preprocessing.json` に記録しています。

## 画面の流れ

1. 入口 → カメラ・姿勢の確認 → キャリブレーションの説明
2. 全画面で 9 点キャリブレーション（1 点あたり約 1.3 秒待ってから記録）。M8 系のモデルはキャリブなしでも推定できます
3. 完了（1 点を除いて当てはめ、除いた点を予測したときの誤差を表示）
4. 使えるデモ
   - **リアルタイム計測**：架空の EC サイト（tgaze market。商品・価格はすべて架空）の上で視線を計測し、商品ごとの注視時間・ヒートマップ・軌跡を分析。CSV に書き出し可
   - **ターゲットゲーム**：ターゲットの当たりの輪（半径 3.5 cm）を 0.5 秒見つめると消え、次が出る
   - **視線キーボード**：QWERTY の大きなキーを 0.8 秒見つめて入力。「かな」でひらがな入力（行 → 文字の 2 段階）に切り替え
5. Esc でいつでも終了

### 録画について

起動すると、顔（カメラ映像）を自動で録画し、最初のクリックで画面全体の録画の許可を求めます。
保存先はこの PC の `webapp/recordings/<日時>/` だけです（顔が写るので、公開・共有しないでください）。
録画しない場合は **http://localhost:8000/?rec=0** で開きます。

## 精度の目安（暫定）

開発者本人 1 人・75 試行のパイロット（MacBook Air M4、1920×1080、実測カメラ行列）での本番 M8 の平均誤差：
キャリブなし 3.22 cm（自然な姿勢 2.74 cm）、9 点キャリブ後 自然な姿勢 2.25 cm。頭を止めて目だけ動かすと誤差が大きくなります。
人数を増やした評価は進行中です。

## 構成（このブランチのうち、Web アプリに関係するもの）

```
webapp/                 Web アプリ（server.py：ローカルサーバ、m8_backend.py：M8 系の推定、static/：画面、requirements.txt）
experiments/phase6_kit/ M8 系の推論部品（m8_cpu.py、common.py、lib/）、評価プロトコル設定、モデル（ONNX）
tgaze/ と直下の *.py     旧手法（関数版 tgaze）と共通部品（rich16d.py：顔の点からの特徴、appearance.py：目の画像）
face_landmarker.task    MediaPipe Face Landmarker のモデル（Google、Apache 2.0）
results/camera_calib/   実測したカメラ行列
```

## 使った AI・ソフト・データ

- 開発：Claude Code（Anthropic、Claude Opus 4.8 / 5 / 5.5）。文献調査に NotebookLM
- 顔の点：MediaPipe Face Landmarker（Google、Apache 2.0）
- 合成データの生成：Blender、MPFB2（MakeHuman。コード GPLv3、素材 CC0）
- 推論：ONNX Runtime、OpenCV、NumPy、scikit-learn
- 合成画像は生成 AI ではなく、Blender の 3D CG です

## ライセンス

`LICENSE` と `NOTICE.md` を参照してください（source-available。利用には許諾が必要です）。
