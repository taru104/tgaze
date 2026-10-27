# rich16d の頭部角度を使わない M8（副次解析用）

Phase 6 パイロットで見つかった「M8 の head_pose 入力に rich16d（6点 solvePnP）の pitch/yaw/roll が入っている」問題の検証結果と、差し替え用モデルです。

- **本番で使うモデルは変えていません。** `model/`（sha256 `db19af0b…c25e`）と `config.json`（frozen_protocol_hash）はそのままです。
- ここにあるモデルは、副次解析の候補として扱います。

## 入っているモデル

| ディレクトリ | head_pose | ONNX sha256 |
|---|---|---|
| `m8_no_rich16d_angles/` | 9次元（Kabsch R_head の2列 + 剛体重心 xy + log s） | `61be2ee1…c160` |
| `m8_kabsch_euler_angles/` | 12次元（9次元 + Kabsch R_head のオイラー角） | `c916c578…b33b2f` |

各ディレクトリの中身:

- **`final_cpu_model.onnx`**:
  - 入力は `head_pose`（N×9 または N×12）と `eye_norm_r16`（N×168）、出力は `y`（N×2、画面 uv）。
  - 入力は生の特徴量で、標準化は ONNX の中で行います。凍結 M8 と同じ形式です。
- **`final_model_preprocessing.json`**:
  - 凍結 M8 と同じキーを持ち、追加で `"head_angles": "none" | "kabsch"` が入っています。
  - テンプレートと標準化は、このモデル自身の学習データから作ったものです。
- **`reference_io.npz`**: 合成テスト画像 40 枚分の lms / f16 と torch の出力。確認用です。

学習方法:

- **データ:** Phase 5 の最終分割（学習 16 人、検証 [6,17]、テスト [7,11]）。
- **レシピ:** Phase 7 の M8 再実装（AdamW、aux 0.3、ドロップアウト 0.1、早期終了）。
- **選び方:** 3 seed のうち検証誤差が最小のものを採用。9次元は seed 1、Kabsch は seed 0。

検証:

- `OfflineM8` で lms から予測し直すと、torch の出力と一致しました（最大差 0）。凍結 M8 でも 1.4e-7 です。
- 変更後の `m8_cpu.py` は、凍結 M8 に対して変更前とビット単位で同じ入力を作ります（75 フレームで差 0）。

## 1. 合成データでの rich16d の角度

- **計算方法:**
  - 比較する 3 つ（rich16d、Kabsch、正解）は、すべて rich16d と同じ規約（`cv2.RQDecomp3x3`）で度に換算しています。
  - Kabsch は凍結 M8 のテンプレートを使うので、Mac 版キットと同じ基準（ゼロ点）です。
  - 正解は Blender の頭部回転 H_cam に、一定の座標系オフセットを合わせたものです。
- **出力:** `experiments/phase7/results/diagnostics/rich16d_angles.json`

| 合成データ（yaw） | オフセット | 傾き | 相関 |
|---|---:|---:|---:|
| rich16d ~ Kabsch（Phase 5 v2、2万枚 20人） | −4.4° | 2.82 | 0.91 |
| rich16d ~ Kabsch（Phase 7 ids、124人） | −4.2° | 2.68 | 0.87 |
| rich16d ~ 正解（v2） | −4.7° | 3.52 | 0.89 |
| Kabsch ~ 正解（v2） | −0.1° | 1.26 | 0.98（残差 0.65°） |
| 1人ごとの rich16d ~ Kabsch（v2 の 20人） | −9.2〜−1.8°（中央値 −3.9°） | 1.96〜3.70（中央値 2.84） | 0.89〜0.98 |
| 1人ごとの rich16d ~ Kabsch（Phase 7 の 124人） | −10.5〜−0.5°（中央値 −4.0°） | 1.44〜5.91（中央値 2.54） | 0.78〜0.99 |
| **実機 subject_00** | **+12°** | **2.7** | **0.78** |

- **傾き 2.7 倍は分布の中:**
  - rich16d の yaw は、合成データでも正解の約 3.5 倍、Kabsch の約 2.8 倍に増幅されています。
  - 実機の 2.7 は合成データの中央値ほぼそのままで、6点汎用モデルによる solvePnP の性質です。
  - 平均絶対誤差（対正解）は rich16d yaw 8.4°、Kabsch 0.8°。
  - pitch は相関が負（rich16d ~ 正解で −0.63）で、roll の相関は 0.15〜0.53 です。
- **オフセット +12° は分布の外:**
  - 合成データでは 144 人全員が負（−10.5〜−0.5°）でした。実機はそこから約 16° 外れています。
  - 1人ごとの rich16d yaw の中央値で見ても、合成データは −28〜+6° です。実機の +14〜22° はどの合成人物よりも大きい値です。
  - Kabsch yaw が +3〜6° のサンプルに絞ると、合成データの rich16d yaw は中央値 11.5°、95 パーセンタイル 17.7°（v2）です。22° 以上は 0.04%（v2）、4%（Phase 7 ids）しかありません。
- **標準化との関係:** head_pose 標準化の yaw は平均 −3.7°、標準偏差 11.9° です。実機の値は z = +1.5〜+2.2 で、単独では極端ではありません。分布から外れているのは「Kabsch に対するずれ（オフセット）」のほうです。

## 2. 角度を除いた M8 / Kabsch オイラー角に置き換えた M8

identity 5-fold CV × 3 seeds で、Phase 7 の `baseline/m8_p7` と同じ条件です。5 点補正も Phase 7 と同じ合成キャリブレーションです。

| モデル | 平均 cm | P95 | pose invariance | 1点 | 3点 | 5点 | 9点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M8（rich16d 角度、12次元） | 1.943 ± 0.03 | 4.10 | 2.75 | 1.75 | 1.61 | 1.535 | 1.50 |
| M8（角度なし、9次元） | **1.891 ± 0.02** | **3.94** | 2.79 | 1.76 | 1.63 | 1.533 | 1.50 |
| M8（Kabsch オイラー角、12次元） | 1.918 ± 0.03 | 4.04 | 2.78 | 1.74 | 1.60 | 1.533 | 1.50 |

- **合成データでの差:**
  - 角度なしにすると zero-shot は 0.05 cm（約 2σ）良くなり、P95 も下がりました。
  - 補正後（5点・9点）は 3 モデルとも同じです。pose invariance も誤差の範囲で同じです。
  - 合成データ上では、rich16d の角度は精度に貢献していません。
- **感度テスト**（`diagnostics/rich16d_shift.json`）:
  - 現行 M8 に、テスト人物の rich16d yaw を一律 +16°（実機と合成のオフセット差）ずらして入れました。
  - 誤差は 1.943 → 1.940 cm でほぼ変わらず、予測の移動は平均 (−0.11, −0.07) cm だけでした。
  - pitch を ±8° ずらすと、誤差 +0.11 cm、予測の移動は最大 0.3 cm で、yaw よりは効いています。
- **結論:**
  - M8 は、Kabsch R と重複する rich16d の yaw をほとんど使っていません。
  - そのため、実機の +12° のずれは問題ではありますが、EYE_DOMINANT が 4〜5 cm に崩れることの主な原因である可能性は低いと考えます。
  - 主因は引き続き、Phase 7 の結論どおり目の向き（eye-in-head）の推定だと見ています。
  - ただし実機の rich16d pitch のずれはまだ見ていません。下の再推論の CSV に rich16d と Kabsch の pitch が両方出るので、そこで確認できます。
  - 角度なし（9次元）の M8 は、精度が同等以上で外れた入力がなくなるため、置き換える価値はあります。

## 3. Mac での使い方（保存済みランドマークからの再推論）

```bash
cd ~/tgaze && git fetch origin && git switch experiment/m8-no-rich16d-angles
cd experiments/phase6_kit && . .venv/bin/activate     # Phase 6 と同じ venv（追加の依存なし）
python reinfer_offline.py outputs/real_mac_validation                      # 凍結 M8 + models_alt/* 全部
python reinfer_offline.py outputs/real_mac_validation --models models_alt/m8_no_rich16d_angles --subjects subject_00
```

- **出力先:** `outputs/real_mac_validation/reinfer/`。
  - `trials.csv`: 評価試行 × モデルごとの生の予測、0/1/3/5/9 点補正後の誤差、Kabsch と rich16d の pitch/yaw/roll。
  - `summary.md` / `summary.json`: 姿勢条件 × 補正点数の平均誤差。
- **補正のやり方:** analyze.py と同じ primary 補正（config.json の λ 25）です。各モデル自身のキャリブ試行の予測で補正を当てはめます。
- **確認項目:** 凍結 M8 については、オフラインの予測とライブの予測（`uv_m8_raw`）の最大差を `summary.json` に記録します（期待値 0）。
- **既存データへの影響:** `subject_*` 以下は一切書き換えません。
- **動作確認:** 偽カメラ（合成画像の 1 人）で一通り通しました。数字はパイプラインの確認用です。

関数として使う場合:

```python
import json, numpy as np, m8_cpu
info = json.load(open("outputs/real_mac_validation/subject_00/subject.json"))
m = m8_cpu.OfflineM8("models_alt/m8_no_rich16d_angles", info["camera_matrix"]["K"], "/Users/<you>/tgaze")
z = np.load("outputs/real_mac_validation/subject_00/evaluation/trial_0123.npz")
rec = (z["phase"] == "record") & z["ok"]
uv, ok = m.predict_lms(z["lms"][rec])     # uv: (N,2) 画面 uv（右 +u、上 +v）。顔が無いフレームは NaN
pred = np.nanmedian(uv, 0)                 # analyze.py と同じ試行代表値
```

- **モデルの指定:**
  - `OfflineM8("model", K, tgaze)` は凍結 M8 です。
  - `final_model_preprocessing.json` の `head_angles` で特徴量が自動的に切り替わります。キーが無ければ rich16d です。
- **ライブで使う場合:** `M8Runtime(cfg, tgaze, K, model_dir="models_alt/m8_no_rich16d_angles")` で差し替えられます。本番の撮影では使わないでください。

## 4. analyze.py の変更提案（このブランチで実装済み、phase6-kit には未反映）

- **追加した列（既存の列は変更なし）:**
  - `trials.csv`:
    - `kabsch_head_pitch_deg` / `kabsch_head_yaw_deg` / `kabsch_head_roll_deg`。npz に保存済みの `R_head` から求めた、記録フレームの中央値です。
    - `kabsch_head_delta_from_cal_deg`。キャリブ試行の Kabsch 姿勢からの差です。
    - zero_shot.csv にも同じ列を出します。
  - `pose_invariance.csv`: `kabsch_head_yaw_diff_deg` / `kabsch_head_pitch_diff_deg` / `kabsch_head_pose_diff_deg`（EYE_DOMINANT と HEAD_ALLOWED の頭の向きの差）。
- **触っていないもの:**
  - `config.json` には一切触れていません。frozen_protocol_hash は config の timing/grids/calibration/evaluation/quality/analysis だけから計算されるので、変わりません。
  - 既存の `head_*_deg` の列（rich16d）はそのまま残しています。
- **注意（変えていないもの）:**
  - collect.py の試行品質判定（`head_motion_retake_deg` 4° / `invalid` 10°）は rich16d の角度で判定しています。
  - yaw は約 2.7 倍に増幅されているので、実際の頭の動きでは約 1.5° / 3.7° 相当で RETAKE / INVALID になります。
  - これは凍結プロトコルの一部なので変えていません。本番 5 人の RETAKE 率を読むときに考慮してください。

## 5. Phase 7E の全 120 人で学習した M8（副次解析用）

Phase 7E（人数の実験、`final_report.md` §27–28）で学習済みの重みを、キットの形に書き出しただけです。**学習し直していません。**

- **副次解析用です。** 本番の `model/`（sha256 `db19af0b…c25e`）と `config.json`（frozen_protocol_hash）は変えていません。書き出しの前後で sha256 が一致することを確認しました。
- 本番の撮影では使わないでください。

| ディレクトリ | 入力 | 元の run | ONNX sha256 | ONNX サイズ |
|---|---|---|---|---:|
| `m8_all120/` | ランドマークのみ（head_pose 12次元 = rich16d 角度あり、eye_norm_r16 168次元） | `identity_diversity/Tall_120ids/fold0_seed0` | `8ac78b41c6efa0d7826aee686ae6ddc2ff998032859d6e40b9a5dfe6c7ee32ab` | 87,109 B |
| `m8_eyepca16_all120/` | 上記 + 目の画像の PCA16（EyePCA、npz の `eye_patch` が必要） | `identity_diversity/eyepca_Tall_120ids/fold0_seed0` | `13c43a00fa5336ce13d2e43d82cfc4d8804033f2d4a52166493cfbce10ffa326` | 91,342 B |

学習データ（2つとも同じ分割）:

- **学習:** 120 人、50,000 枚。v2 の id 0–19（1人 1000 枚）+ p7_ids の新しい id 20–69（400 枚）と 70–119（200 枚）。62 と 81 は生成で除外され、144 と 145 で置き換えています。
- **検証:** id 140–143（800 枚）。早期終了と seed 選びに使用。
- **テスト:** id 120–139（4,000 枚）。
- **注意:** 学習に v2 の 0–19 番が入っているので、Phase 5 のテスト [7, 11] はこのモデルにとって未知の人ではありません。他の models_alt と [7, 11] で比べないでください。
- 合成人物の目の見た目（虹彩・肌・照明）はみな同じです。EyePCA 版は、実機で目の見た目が外れると崩れる可能性があります（下の残差の比率で確認できます）。

seed の選び方:

- `m8_all120`: 3 seed のうち、**検証（id 140–143）の平均誤差が最小**の seed 0 を採用しました。テストと実機の数字は使っていません。
  - 検証 cm: seed 0 = 2.025、seed 1 = 2.032、seed 2 = 2.042
- `m8_eyepca16_all120`: 学習されているのは seed 0 だけです（検証 1.201 cm）。

合成テスト（id 120–139、4,000 枚、個人補正なし）:

| モデル | 平均 cm | P95 | pose invariance |
|---|---:|---:|---:|
| `m8_all120`（seed 0） | 1.946 | 4.11 | 2.73 |
| `m8_eyepca16_all120` | 1.183 | 2.45 | 1.40 |

- いずれも合成データの数字で、実機の精度ではありません。

書き出しと検証（`experiments/phase7/code/run_kit_all120.sh`）:

- `p7export.py` → `p7onnx_bench.py` → `p7bundle_kit.py`。kit_m8_eyepca16 と同じ手順です。
  - `p7export.py` / `p7onnx_bench.py` は、run の `--root` に合わせて v2 + p7_ids を結合したデータを読むように直しました。v2 だけの run は今までと同じ動作です。
- **ONNX と torch の最大差**（`p7onnx_bench.py`、合成テスト 64 枚）: `m8_all120` 5.96e-8、`m8_eyepca16_all120` 1.79e-7。NumPy の特徴量と学習時の入力の差は 5.2e-7 / 9.4e-7。
- **キットの `OfflineM8` で lms から予測し直した場合**: `m8_all120` は torch と差 0。`m8_eyepca16_all120` は 2.1e-7（float32 の目の画像）、`reference_io.npz` に保存した float16 の目の画像では 7.2e-5 です。
- **CPU**（Xeon Gold 6134、CPUExecutionProvider のみ、torch なし、GPU なし）: 28.4 FPS / 26.9 FPS（`experiments/phase7/results/cpu_runtime/benchmark_kit_all120.json`）。
- `final_model_preprocessing.json` に `source_run`、`training_data`（id・人数・枚数）、`seed_selection`（検証誤差）、`synthetic_test_metrics` を書いています。EyePCA 版には `patch_pca_resid_frac_synthetic_test`（合成テストの目の画像の PCA 残差の比率、中央値 0.249）も入っています。

Mac で再推論する場合:

```bash
cd ~/tgaze && git fetch origin && git switch experiment/m8-all120
cd experiments/phase6_kit && . .venv/bin/activate
python reinfer_offline.py outputs/real_mac_validation \
    --models models_alt/m8_all120 models_alt/m8_eyepca16_all120 \
    --subjects subject_00 --out outputs/real_mac_validation/reinfer_all120
```

- 凍結 M8（`model/`）は基準として必ず一緒に計算されます。
- `--out` を付けているのは、前回の `reinfer/` を上書きしないためです。
- EyePCA 版は、trial の npz に `eye_patch` が入っている必要があります（`config.json` の `save_eye_patch: true` で撮ったもの）。
