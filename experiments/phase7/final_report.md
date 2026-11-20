# Tgaze Phase 7: Synthetic-only accuracy improvement (final report)

Date: 2026-10-11. **All numbers are synthetic** (MPFB2 / Blender renders as seen by the MacBook Air 13" M4 webcam with the measured K). They are not real-webcam accuracy.

## Running and outputs

- **Code:** `~/mpfb2-gaze-generator-poc/phase7/` (new files only).
- **Results:** `outputs/experiments/phase7/`. Each run (fold, seed) has its own directory: `result.json`, `curves.json`, `pred.npz`, `model.pt`. A run with an existing `result.json` is skipped, so experiments resume where they stopped.
- **Untouched:** the Tgaze runtime (`~/tgaze`, git clean), the Phase 5 dataset and models, and the Phase 6 kit. The new data lives in `outputs/gaze_dataset_p7_ids/`.
- **Training:** Quadro RTX 6000 (CUDA).
- **Inference / benchmark:** ONNX Runtime with CPUExecutionProvider only, without torch, with `CUDA_VISIBLE_DEVICES=` (empty).

---

## 0. Summary of conclusions

| Model | Mean cm | P95 cm | Pose invariance cm | 5-pt calibration cm | CPU FPS (all cores / 4 cores) |
|---|---|---|---|---|---|
| B2 Huber (rich16d + PCA16) | 1.96 | 4.62 | 2.79 | 1.53 | – (linear) |
| **M8** (reproduced, 3 seeds) | 1.94 ± 0.03 | 4.10 | 2.75 | 1.54 | 27.2 / 27.1 |
| M8 + Cons (scaled λ 0.25) | 1.92 ± 0.02 | 4.06 | 2.72 | 1.54 | 27.6 / 29.4 |
| M9 (physical head–eye) | 2.19 ± 0.01 | 4.55 | 3.18 | 1.69 | 28.2 / 28.2 |
| M9 + Cons (scaled λ 0.25) | 2.18 ± 0.03 | 4.51 | 3.14 | 1.69 | 28.1 / 29.4 |
| (ref) M8 + NoiseAug σ 0.25 px | 1.98 ± 0.04 | 4.12 | 2.77 | 1.57 | same as M8 |
| (ref) **M8 + EyePCA16** * | **1.58 ± 0.02** | **3.53** | **1.69** | **0.99** | 27.0 / 24.0 |

Reference conditions: Phase 5 M8 was 1.90 ± 0.29 cm, P95 4.00, pose invariance 2.77. CV is identity-disjoint, 5 folds; the ± is the standard deviation over 3 seeds.

\* **EyePCA uses eye-patch appearance.** That is the tgaze B2 eye-patch PCA16, added as an input to M8's eye branch. In the synthetic data, iris, skin and lighting are identical for every identity, so the appearance features are unrealistically clean. How well this transfers to real data is the biggest uncertainty.

### Main findings

1. **The pose-consistency loss barely improves the pose problem.**
   - The raw pose invariance drops at λ ≥ 1 (2.75 → 2.29 → 1.17), but the outputs shrink toward the centre in step: the predicted / true spread goes 0.91 → 0.80 → 0.47.
   - Mean error worsens at the same time: 1.96 → 3.86 cm.
   - The **scale-normalized** pose invariance improves by only a few percent.
   - The best setting (scaled λ = 0.25) gives 1.94 → 1.92 cm and 2.75 → 2.72 cm, within seed noise.
2. **The physical model M9 is worse than M8:** 2.19 vs 1.94 cm, pose invariance 3.18 vs 2.75.
   - Feeding the oracle Blender head rotation and eye centres only recovers it to about M8 level, and pose invariance stays around 3.3.
   - So head-pose estimation is not the bottleneck. **Estimating the eye-in-head direction from MediaPipe eye landmarks** is: about 2° on unseen people.
3. **The landmarks lack information.**
   - Even a large model has a training error of 1.05 cm or more.
   - Simply adding eye-patch PCA16 (appearance) cuts the error 19% and the pose invariance 39%. It is the only model that reaches the < 2.0 cm pose-invariance goal.
4. **The number of people matters more than the number of images.**
   - At a fixed 20,000 samples: 20 × 1000 → 2.10, 50 × 400 → 1.97, 100 × 200 → 1.92 cm. The gap shrinks 0.77 → 0.19.
   - 100 people × 40 images (4,000) beat 20 people × 1000 images (20,000).
   - Landmark-only M8 levels off at about 1.92–1.94 cm; EyePCA keeps improving with more people (1.60 → 1.27 → 1.18).
5. **Calibration works.**
   - M8: 0 → 1 → 3 → 5 → 9 points gives 1.94 → 1.75 → 1.61 → 1.54 → 1.50 cm.
   - Most of the inter-identity spread (0.61 cm) is removed; it falls to 0.13.
   - **5 points is the practical sweet spot.**
6. **Camera K errors and lens distortion hardly matter. Landmark jitter does.**
   - K errors of ±5% focal and ±20 px principal point, and distortion up to 1.5×, change the mean error by 0.05 cm or less.
   - Gaussian noise of 0.5 / 1 px on the eye landmarks raises it to 2.54 / 3.70 cm.
   - A 0.5 px sub-pixel shift of the image alone moves per-frame predictions by 0.7 cm.
   - Noise augmentation (σ 0.25 px) costs +0.04 cm on clean data and holds the error at 1 px noise to 2.50 cm instead of 3.63.
7. **CPU only:** every candidate runs at about 27–29 FPS on all cores and 24–29 FPS on 4 cores. MediaPipe takes about 30 ms, ONNX 0.16–0.27 ms; the ONNX files are 55–91 KB.

**Recommendation for the hackathon:** M8 (the frozen Phase 5 weights, statistically equivalent to M8 + Cons) with 5-point personal calibration. Run on the CPU, as already wired up in the Phase 6 kit. Details in §12.

---

## 1. M8 reproduction (Checkpoint 1)

- **Phase 5 script, unchanged, same 5 folds / aux weight 0.3 / 1 seed:** fold test 1.84 / 1.77 / 1.73 / 1.68 / 2.46. That is **exactly the same as Phase 5** (mean 1.90 ± 0.29, P95 4.00, pose invariance 2.77).
  - B0 6.71, B1 2.71, B2 1.96 are also exact (`baseline/phase5_script_repro`).
- **Phase 7 reimplementation** (`phase7/p7train.py`, λ = 0): mean 1.943 ± 0.03 over 3 seeds, P95 4.10, pose invariance 2.75. Within seed noise.
  - Phase 7 uses group-level mini-batches so that consistency can be added. All Phase 7 comparisons use this same code path.

## 2. Evaluation conditions (fixed)

- **Split:** identity-disjoint 5-fold CV, with the same splits as Phase 5. Each fold is 14 train / 2 validation / 4 test identities, so all 20 people are the test set once.
- **Leakage:** groups (counterfactual sets) never cross train, validation and test; this is asserted for every pair.
- **Model selection and early stopping:** validation identities only.
- **Metrics:** mean, median, RMSE, P50/P90/P95, mean angular error, and pose invariance.
  - **Pose invariance:** for each same-target group, the maximum pairwise distance between the 4 predictions (Phase 5 definition). Also recorded: the mean deviation from the group mean, the predicted / true spread ratio, and the spread-normalized value.
  - Also: train / validation / test error and the generalization gap, plus breakdowns by head angle, screen region and pose mode.
- **Seeds:** 3 for the main models.

## 3–4. Phase 7A: Pose consistency loss

**Design (§2):**
- Mini-batches are built from whole groups, so every same-target group (one target, 4 poses: EYE_ONLY / HEAD_FOLLOW / NECK_HEAD_FOLLOW / COUNTER) is complete inside a batch.
- L_total = Huber(δ = 1 cm, display cm) + aux (as in M8) + λ·L_cons.
- **mean_dev:** mean_i √(‖p_i − mean_g p‖² + ε), in cm.
- **pairwise:** mean distance over all pairs.
- **scaled:** deviations divided by the batch's predicted spread and multiplied by the true spread, so shrinking the outputs cannot lower it.
- Applied to the screen point only. Different gaze vectors under different poses are not forced to match.

**λ pilot** (folds 0, 1 × 2 seeds; selected on validation by cm + 0.5·pose invariance, the Phase 5 weighting; `consistency/lambda_selection.md`):

| Variant | Val cm | Val pose inv | Test cm | Test pose inv |
|---|---|---|---|---|
| λ 0 | 1.566 | 2.64 | 1.72 | 2.84 |
| mean_dev 0.05 / 0.1 / 0.25 | 1.57 / 1.57 / 1.63 | 2.63 / 2.67 / 2.51 | 1.73 / 1.72 / 1.75 | 2.78 / 2.78 / 2.68 |
| mean_dev 0.5 / 1 | 1.66 / 1.87 | 2.48 / 2.20 | 1.78 / 1.96 | 2.58 / 2.29 |
| mean_dev 2 / 5 | 3.88 / 6.66 | 1.18 / 0.03 | 3.86 / 6.70 | 1.17 / 0.01 (collapse) |
| pairwise 0.5 / 2 | 1.80 / 5.99 | 2.27 / 0.24 | 1.93 / 6.01 | 2.42 / 0.22 |
| **scaled 0.25** (selected) | **1.542** | 2.66 | 1.705 | 2.77 |
| scaled 0.5 / 1 / 2 | 3.62 / 5.11 / 6.67 | – | collapse | |

**Selected λ (§3):** scaled consistency, λ = 0.25. For comparison, mean_dev λ = 0.5 was also run on the full CV.

**Results (§4), full CV × 3 seeds:**
- **M8 + Cons (scaled 0.25):** 1.92 ± 0.02 cm, P95 4.06, pose invariance 2.72.
- **mean_dev 0.5:** 1.96 cm, pose invariance 2.52, with outputs shrunk (spread 0.87).

**Pareto front:** `plots/pose_invariance.png`, left panel. Pose invariance of 2.0 or less is only reachable with mean error of 3.5 cm or more, so the loss fails the success criterion of §13.

**Cause:**
- The consistency loss can lower its value by "shrinking all outputs toward the centre".
- The within-group inconsistency is not even fully resolved on **the training identities** (train pose invariance 2.33 at λ = 0). That points to missing information in the input landmarks, which no loss can create.
- After 9-point calibration (which undoes the shrinking), every M8-family model lands at a pose invariance of 3.0–3.1.

**Failure case also recorded:** with EyePCA + scaled λ 0.25, 2 of the 5 folds of seed 2 partly collapsed (3.5 cm), giving 1.79 ± 0.29. The scale-invariant version has a degenerate solution, so it is unsafe at production scale.

## 5–9. Phase 7B / 7C: M9, a physical head–eye model

### Architecture (§5)

```
MediaPipe 478 landmarks (mirrored frame, measured K mirrored)
  ├─ Head pose (analytic, no learning): Kabsch (SVD) of the rigid 40-point subset onto the training-identity template
  │     -> R_kabsch;  R_head = R_kabsch · R_off  (R_off = constant Procrustes average over training identities,
  │        mapping the template frame to the Blender head-bone frame)
  ├─ Eye MLP: input = eye_norm_r16 (52 eye/iris/brow points de-rotated into the head frame, centroid / scale removed,
  │     156 values + rich16d eye features 12 = 168) -> 64 -> 32 -> 6 -> + g0 (mean local gaze) -> per-eye L2 normalization
  │     -> g_local_L, g_local_R (3D unit vectors, eye-in-head)
  ├─ g_cam_e = R_head @ g_local_e                 (matrix product, outside the network)
  ├─ Eye centre O_e = z·(x_e, y_e, 1) + R_head·d_e
  │     (x_e, y_e) = ncam midpoint of the eye corners (33/133, 263/362); z = exp(w·[log s, log IPD, log iris] + b)
  │     w, b, d_e = 10 physical constants (initialised by least squares on training-identity GT eye centres;
  │     fine-tuned with lr 1e-4 and no weight decay; aux loss on the origin)
  ├─ P_e = ray–plane intersection(O_e, g_cam_e, display plane)  (geometry)
  └─ screen = (P_L + P_R)/2 -> (u, v)
```

- **Loss:** Huber(δ = 1 cm) on the final screen cm + 0.3 × angular loss on the eye-in-head vectors (Blender head frame, degrees) + 1.0 × SmoothL1 on the eye centres (cm).
- **Parameters:** 13.5k. No Blender ground truth is used at inference.
- **Variant choice** (pilot on folds 0, 1, validation cm): 3D vector + screen loss 1.53; yaw/pitch output 1.56; angular loss 1.56; local loss only (pure physics, no end-to-end) 1.94; no local aux 1.56; learned R_off 1.54; fixed physical constants 1.54; hidden 128-64-32 1.56.
  - **3D unit vector + screen loss** (the spec's first choice) was selected.

### Coordinate frames (§6)

- **cam:** the mirrored OpenCV camera frame, p_cam = MIR·(R_w2c·p_world + t_w2c) with MIR = diag(−1, 1, 1). It matches the mirrored=True image that tgaze uses.
- **head:** the Blender `head` bone frame. Because of the mirroring, it is conjugated as H_cam = MIR·R_w2c·R_bone·MIR to stay a proper rotation (det = +1, checked).
- **display:** a fixed plane (lid 105°) in the cam frame. Within 1e-4 m it is identical for every identity (asserted).

### Eye-in-head ground truth (§7)

- g_local = H_camᵀ·g_cam_true, computed per eye. g_cam_true is the eye-centre-to-target vector from the metadata.
- **Convention check:** the generator's `eye_in_head_yaw_pitch_deg` is h_worldᵀ·f in world axes (h_world = the pose rotation). The bone-frame vectors above match it up to **one constant rotation per identity and eye, with a maximum residual of 0.09°**. The axes were confirmed from the code and data, not assumed.
- Vergence is kept, because the two eyes are predicted and intersected separately.

### Results (§8)

- **M9:** 2.19 ± 0.01 cm, P95 4.55, pose invariance 3.18, angular error 2.11°, gap 0.78.

**Decomposition (diagnostic only, not deployable; folds 0, 1):**

| | Test cm | Pose inv |
|---|---|---|
| M9 | 1.99 | 3.29 |
| oracle head rotation | 1.97 | 3.23 |
| oracle eye centres | 2.00 | 3.28 |
| oracle head + eye centres | 1.91 | 3.31 |
| + learned head residual | 1.95 | 3.20 |

- **Head pose:** the Kabsch rotation differs from the ground truth by 1.57° mean (P95 2.85°) on unseen people, and its correlation with eye motion is −0.22. tgaze's solvePnP gives 11° under my Euler conversion; the conversion is not validated, so it was not adopted.
- **Eye centres:** depth is estimated from iris size with a 1.0 cm error.
- **Conclusion:** even with both of these made perfect, M9 does not reach M8 (fold 0/1: 1.72–1.76). **The bottleneck is the accuracy of eye-in-head direction from MediaPipe eye landmarks.**
  - M8's learned fusion absorbs some of that systematic error.
  - M9's strict geometric composition turns every eye-in-head angle error directly into screen error (1° ≈ 1 cm).

### M9 + consistency (§9)

| | Mean cm | Pose inv |
|---|---|---|
| scaled 0.25 | 2.18 ± 0.03 | 3.14 |
| mean_dev 0.5 | 2.14 ± 0.01 | 3.00 |

Same as with M8: no meaningful improvement.

## 10–21. All metrics (identity 5-fold CV)

| Model | Seeds | Mean | Median | RMSE | P50 | P90 | P95 | Deg | PoseInv | Train | Val | Test | Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 mean | 1 | 6.71 | 6.41 | 7.61 | 6.41 | 12.00 | 12.99 | 6.52 | 0.00 | 6.71 | 6.72 | 6.71 | 0.00 |
| B1 rich16d+Huber | 1 | 2.71 | 2.45 | 3.13 | 2.45 | 4.86 | 5.66 | 2.62 | 3.93 | 2.32 | 2.28 | 2.71 | 0.39 |
| B2 rich16d+PCA16+Huber | 1 | 1.96 | 1.60 | 2.40 | 1.60 | 3.94 | 4.62 | 1.89 | 2.79 | 1.47 | 1.59 | 1.96 | 0.50 |
| M8 | 3 | 1.94 ± 0.03 | 1.75 | 2.25 | 1.75 | 3.51 | 4.10 | 1.88 | 2.75 | 1.31 | 1.57 | 1.94 | 0.63 |
| M8+Cons (scaled 0.25) | 3 | 1.92 ± 0.02 | 1.73 | 2.22 | 1.73 | 3.45 | 4.06 | 1.85 | 2.72 | 1.31 | 1.58 | 1.92 | 0.61 |
| M8+Cons (mean-dev 0.5) | 3 | 1.96 ± 0.04 | 1.79 | 2.27 | 1.79 | 3.52 | 4.09 | 1.90 | 2.52 | 1.44 | 1.69 | 1.96 | 0.52 |
| M9 | 3 | 2.19 ± 0.01 | 2.00 | 2.53 | 2.00 | 3.93 | 4.55 | 2.11 | 3.18 | 1.41 | 1.54 | 2.19 | 0.78 |
| M9+Cons (scaled 0.25) | 3 | 2.18 ± 0.03 | 1.99 | 2.52 | 1.99 | 3.91 | 4.51 | 2.10 | 3.14 | 1.42 | 1.55 | 2.18 | 0.77 |
| M9+Cons (mean-dev 0.5) | 3 | 2.14 ± 0.01 | 1.95 | 2.47 | 1.95 | 3.84 | 4.43 | 2.06 | 3.00 | 1.44 | 1.59 | 2.14 | 0.70 |
| M8+NoiseAug σ 0.25 px | 3 | 1.98 ± 0.04 | 1.80 | 2.28 | 1.80 | 3.54 | 4.12 | 1.91 | 2.77 | 1.39 | 1.63 | 1.98 | 0.60 |
| M8+EyePCA16 * | 3 | 1.58 ± 0.02 | 1.35 | 1.88 | 1.35 | 3.08 | 3.53 | 1.53 | 1.69 | 0.83 | 1.09 | 1.58 | 0.75 |
| M8+EyePCA16+Cons * | 3 | 1.79 ± 0.29 | 1.54 | 2.14 | 1.54 | 3.52 | 4.02 | 1.73 | 1.57 | 1.20 | 1.40 | 1.79 | 0.59 |

All values in cm except Deg.

**Breakdowns** (§16 by head angle, §17 by screen region, plus pose mode; test mean cm):

| Model | Head <3° | Head 3–7° | Head >7° | Centre | Middle | Edge | EYE_ONLY | HEAD_FOLLOW | NECK_HEAD | COUNTER |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 | 1.73 | 2.00 | 2.53 | 1.80 | 1.89 | 2.17 | 1.92 | 1.75 | 2.21 | 1.89 |
| M8 | 1.82 | 1.98 | 2.13 | 1.77 | 1.87 | 2.18 | 2.04 | 1.81 | 1.89 | 1.89 |
| M8+Cons | 1.81 | 1.95 | 2.12 | 1.74 | 1.86 | 2.15 | 2.00 | 1.80 | 1.86 | 1.86 |
| M9 | 2.10 | 2.21 | 2.40 | 2.07 | 2.12 | 2.37 | 2.27 | 2.06 | 2.07 | 2.23 |
| M9+Cons | 2.10 | 2.20 | 2.36 | 2.09 | 2.12 | 2.33 | 2.28 | 2.04 | 2.03 | 2.23 |
| M8+EyePCA * | 1.49 | 1.61 | 1.76 | 1.47 | 1.51 | 1.77 | 1.57 | 1.50 | 1.57 | 1.60 |

- The M8 family is more robust than B2 at large head angles (2.13 vs 2.53).
- Errors are largest at the screen edges, and for EYE_ONLY (large eye rotation).

## 22–26. Phase 7D: Synthetic personal calibration (Checkpoint 7)

**Method:**
- Each test identity of each fold is treated as an "unseen virtual subject".
- **Calibration pool:** that identity's natural-pose single (`random`) samples. For each point of the Phase 6 point sets, 2 samples are drawn from within 0.15 uv:
  - 1 point = centre;
  - 3 points = triangle;
  - 5 points = centre + corners at ±0.7;
  - 9 points = 3×3.
- **Correction:** `real_mac_validation/common.py` `Correction`, used as is:
  - 1 point = bias;
  - 3/5/9 points = ridge affine with λ = 25 cm²;
  - secondaries: bias + scale, poly2.
- **Evaluation:** all remaining samples, excluding every group used for calibration at any level. So **no frame and no counterfactual group is in both calibration and evaluation**, and the evaluation set is the same for every level.
- Averaged over 10 random draws.

| Model | 0 pt | 1 pt | 3 pt affine | 3 pt bias+scale | 5 pt | 9 pt | 9 pt poly2 | P95 @ 5 pt | PoseInv @ 9 pt |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 2.70 | 2.42 | 2.40 | 2.28 | 2.32 | 2.20 | 2.25 | 4.74 | 4.61 |
| B2 | 1.96 | 1.62 | 1.55 | 1.51 | 1.53 | 1.44 | 1.45 | 3.23 | 3.03 |
| **M8** | 1.94 | 1.75 | 1.61 | 1.57 | **1.54** | 1.50 | 1.52 | 3.11 | 3.07 |
| M8+Cons | 1.92 | 1.74 | 1.64 | 1.60 | 1.54 | 1.50 | 1.53 | 3.12 | 3.04 |
| M9 | 2.19 | 1.86 | 1.74 | 1.70 | 1.69 | 1.63 | 1.68 | 3.48 | 3.41 |
| M9+Cons | 2.18 | 1.84 | 1.74 | 1.70 | 1.69 | 1.63 | 1.67 | 3.48 | 3.37 |
| M8+NoiseAug | 1.98 | 1.80 | 1.67 | 1.63 | 1.57 | 1.53 | 1.55 | 3.20 | 3.13 |
| M8+EyePCA * | 1.58 | 1.18 | 1.04 | 1.02 | **0.99** | 0.94 | 0.94 | 2.02 | 1.86 |

- **Inter-identity spread** (standard deviation of per-identity means), M8: 0.61 cm at 0 points → 0.17 at 1 point → 0.13 at 5 points. Calibration removes the per-person bias almost completely.
- **The residual is within-person error that depends on pose and sample**, about 1.5 cm for M8.
- **3 points:** the low-DOF bias + scale beats the affine (1.57 vs 1.61). For 9 points, poly2 brings no improvement.
- **Pose invariance after calibration** is 3.0–3.1 for the M8 family. This is the "shrink-free" pose invariance; the zero-shot 2.75 is flattered by the shrinking.
- **Calibration curve:** `plots/calibration_curve.png`. From 0 to 5 points M8 improves by −21% (1.94 → 1.54); 5 → 9 points adds only −0.04.

## 27–28. Phase 7E: Identity diversity (Checkpoint 8)

**Data:** `outputs/gaze_dataset_p7_ids`.
- Same generator, config, seed and identity distribution as v2 (moderate macro ranges, 12 facial morph targets per person, N(0, 0.28) clipped to ±0.55). Only the identity indices are new.
- Lighting, skin, iris and background are unchanged.
- **Pilot:** 30 identities × 20 → 600/600 MediaPipe detections, faces checked by eye.
- **Full:** ids 20–69 × 400, 70–119 × 200, test 120–139 × 200, validation 140–143 × 200.
- **QA:**
  - ids 62 and 81 were rejected by the generator (no valid same-target group). They are replaced by 144 (× 400) and 145 (× 200).
  - Shape statistics: 9 identities have a |z| > 3.5 deviation, but the reference standard deviation comes from only 20 people. Visual inspection on the contact sheet found nothing abnormal, so all were kept (`identity_diversity/qa_identities.json`, `identities_contact_sheet.jpg`).
  - 124 people, 34,800 samples, 100% MediaPipe detection.
  - Landmarks are extracted from the lossless PNG; afterwards the images are stored as JPEG q92 to save disk.
  - Rendering took about 3 hours.

**Results:** the test set is the same 20 new people for every condition, the M8 recipe is unchanged, 3 seeds (`identity_diversity/summary.json`, `plots/identity_curve.png`):

| Total samples | Identities × per identity | Test cm | P95 | Train | Gap |
|---|---|---|---|---|---|
| 20,000 | 20 × 1000 (v2) | 2.10 ± 0.02 | 4.55 | 1.34 | 0.77 |
| 20,000 | 50 × 400 | 1.97 ± 0.01 | 4.20 | 1.74 | 0.24 |
| 20,000 | **100 × 200** | **1.92 ± 0.03** | 4.08 | 1.73 | **0.19** |
| 4,000 | 20 × 200 (v2 / new ids) | 2.13 / 2.19 | 4.60 | 1.28 / 1.63 | 0.85 / 0.55 |
| 4,000 | 40 × 100 | 2.02 ± 0.02 | 4.34 | 1.62 | 0.40 |
| 4,000 | 100 × 40 | 1.95 ± 0.02 | 4.06 | 1.69 | 0.26 |
| ~50,000 | all 120 people | 1.94 ± 0.01 | 4.08 | 1.68 | 0.26 |
| 20,000, EyePCA * | 20 × 1000 / 50 × 400 / 100 × 200 | 1.60 / 1.31 / 1.27 (1 seed) | | | |
| ~50,000, EyePCA * | all 120 people | 1.18 (P95 2.45, pose invariance 1.40) | | | |

- **Conclusion:** at a fixed sample count, more people is consistently better, and 100 × 40 (4k) beats 20 × 1000 (20k). Overfitting to individuals (the gap) shrinks sharply.
- **Plateau:** landmark-only M8 saturates at about 1.92–1.94 cm with 100+ people. The landmark information limit stays, which matches §5–9 and §54.
- **Appearance:** EyePCA keeps improving with more people, so it will benefit from adding identities in the future (especially with appearance diversity).
- **Not done:** eyeball and iris size are not varied by the current generator; this is a remaining item.

## 29–31. Phase 7F: Robustness (Checkpoint 9)

### K perturbation (§29)

**(a) Wrong K in feature extraction only** (all 20 people as unseen; ±1/3/5% focal, ±2/5/10/20 px principal point):
- M8 / M8 + Cons / EyePCA change by **0.02 cm or less**.
- M9 rises by **0.15 cm** at ±20 px principal point, because it uses K geometrically.

**(b) The camera really has K'.** This is simulated exactly: for a pinhole camera, changing K is an affine warp K'K⁻¹ of the render. MediaPipe is re-run on fold-0 test identities (4,000 images).
- Mean error stays within **1.67–1.76 cm** (clean 1.70).
- Using the true K' vs the nominal K makes almost no difference.
- The per-frame prediction changes (1.3–1.5 cm) are dominated by MediaPipe's sensitivity to resampling. Control: a 0.5 px shift gives 0.70 cm, 1 px gives 1.09 cm; the unmodified re-run reproduces exactly (0.00).

### Landmark noise (§30)

Mean error in cm; σ in pixels at 1920 × 1080.

| Noise | σ | M8 | M9 | M8+EyePCA |
|---|---|---|---|---|
| all points | 0.1 / 0.25 / 0.5 / 1 / 2 | 1.93 / 2.09 / 2.54 / 3.70 / 5.99 | 2.22 / 2.36 / 2.80 / 3.90 / 5.99 | 1.58 / 1.64 / 1.83 / 2.42 / 3.89 |
| iris only | 0.5 / 1 / 2 | 2.12 / 2.58 / 3.51 | 2.45 / 2.99 / 4.05 | 1.65 / 1.84 / 2.28 |
| eyelids / eye contours only | 0.5 / 1 / 2 | 2.26 / 3.02 / 4.84 | 2.48 / 3.15 / 4.77 | 1.70 / 2.04 / 3.04 |
| rigid (head-pose) points only | 0.5 / 1 / 2 | 1.91 / 1.94 / 2.04 | 2.20 / 2.24 / 2.36 | 1.58 / 1.59 / 1.65 |
| whole-face shift | 0.5–5 px | 1.90 (no change) | 2.19–2.21 | 1.57 |

- **What drives the sensitivity:** the eye contours and iris. Head pose and whole-face translation hardly matter.
- **Noise augmentation (§55-type remedy):** M8 trained with landmark noise σ 0.25 px, 2 replicas. On folds 0, 1:
  - clean input 1.74 (baseline 1.70);
  - 0.5 px noise: 1.97 (baseline 2.41);
  - 1 px noise: 2.50 (baseline 3.63).
  - Full CV: 1.98 ± 0.04.

### Lens distortion (§31)

The measured Mac distortion coefficients ([−0.031, 0.048, −0.0027, 0.0030, −0.062]) are applied to the raw render at 0 / 0.5 / 1 / 1.5×, and MediaPipe is re-run.

| | Distorted, no correction | Distorted, cv2.undistort |
|---|---|---|
| M8 clean 1.70 | 1.69 / 1.69 / 1.71 | 1.70 / 1.71 / 1.71 |
| M9 clean 1.84 | 1.81 / 1.81 / 1.82 | 1.84 / 1.87 / 1.86 |
| EyePCA clean 1.65 | 1.66 / 1.64 / 1.60 | 1.72 / 1.76 / 1.76 |

- The distortion of this camera is small and the face is near the image centre, so **the effect on accuracy is negligible**.
- Undistortion halves the per-frame change (0.94 → 0.46 cm) but brings no accuracy gain.
- That supports Phase 6's decision to make no undistortion the primary condition.

### Mirroring (§50)

- **Automatic test** (`robustness/mirror_unit_test.json`): for every identity and every fold model, right targets − left targets > 2 cm and up − down > 1 cm. **PASS for all models**; the minimum right − left is 15.7 cm.
- If the flip is omitted (an unmirrored image goes in), the error becomes 10.4 cm and the correlation of predicted u with true u is −0.98. The mistake is detected immediately.
- The runtime uses `cv2.flip(frame, 1)` and `rich16d.set_camera_matrix(K, mirrored=True)`, the same as tgaze.

## Phase 7G: Overfitting and generalization

- **Train/validation curves:** for every run, `curves.json` stores per-epoch train / val loss, train / val cm and val pose invariance. Plot: `plots/train_val_curves.png`.
- **Gaps:** M8 0.63, M9 0.78, B2 0.50, EyePCA 0.75. Diversifying identities cuts the gap from 0.77 to 0.19.

**Learning curve** (§53; `plots/learning_curve.png`; 2 seeds, whole-group subsampling):

| Train samples | 1000 | 2500 | 5000 | 10000 | 14000 (all 14 identities) |
|---|---|---|---|---|---|
| test cm | 2.16 | 2.06 | 1.99 | 1.90 | 1.94 |
| gap | 1.00 | 0.80 | 0.69 | 0.62 | 0.63 |

It is flat from 10k onward. 16,000 cannot be reached because 5-fold CV has at most 14 people × 1000 for training.

**Capacity** (§54, 5 folds × 1 seed):

| Model | Test | Train | Gap |
|---|---|---|---|
| small (64-32, current) | 1.94 | 1.31 | 0.63 |
| medium (128-64-32) | 1.90 | 1.27 | 0.63 |
| large (256-128-64) | 1.90 | 1.07 | 0.83 |
| large, no dropout / no weight decay | 2.12 | 1.05 | **1.08** (overfit) |
| large, dropout 0.3 / weight decay 1e-2 | 1.89 | 1.19 | 0.70 |

- Only the training error falls, and test is unchanged, so larger models were **not adopted**.
- Training error never goes below about 1.05 cm, which again indicates the information limit of the landmarks.

## 32–37. Phase 7H: CPU runtime (Checkpoint 10)

**Export:** `cpu_runtime/<name>/`.
- `p7export.py` (tgaze venv) → weights and preprocessing JSON.
- `p7onnx_bench.py` (`.venv-tools`, no torch, `CUDA_VISIBLE_DEVICES=`) → hand-built ONNX graph (opset 18, IR 9).
- M9 includes the composition, origin and intersection steps inside the ONNX graph.
- **Deployment models:** trained on 18 people (validation identity 17, sanity-check test identity 6). Accuracy figures come from CV.

**Benchmark:** Intel Xeon Gold 6134. Pipeline: decoded 1920 × 1080 frame → flip → MediaPipe (CPU delegate) → rich16d → NumPy features → ONNX. 300 frames.

| Model | ONNX size | ONNX ms | Features ms | MediaPipe ms | FPS (all cores) | FPS (taskset 4 cores, ORT 4 threads) | CPU % of 1 core (all / 4 cores) |
|---|---|---|---|---|---|---|---|
| M8 | 85.1 KB | 0.16 | 3.7 | 32.9 | 27.2 | 27.1 | 221 / 131 |
| M8+Cons | 85.1 KB | 0.17 | 3.6 | 32.3 | 27.6 | 29.4 | 238 / 132 |
| M9 | 54.3 KB | 0.27 | 4.0 | 31.2 | 28.2 | 28.2 | 233 / 131 |
| M9+Cons | 54.3 KB | 0.25 | 3.9 | 31.4 | 28.1 | 29.4 | 233 / 132 |
| M8+EyePCA * | 89.2 KB | 0.19 | 6.8 (eye patch + CLAHE) | 29.9 | 27.0 | 24.0 | 441 / 211 |

- **About 90% of the latency is MediaPipe.** Model differences are under 1 ms, apart from EyePCA's patch extraction.
- **Memory:** RSS is about 2.4 GB, but about 1.9 GB of that is the 300 frames preloaded for the benchmark plus the evaluation data cache. The model itself is under 100 KB.
- **Phase 5 vs now:** Phase 5 measured about 28 FPS; this run is equivalent.

**How "no GPU" was verified (§37):**
1. The ONNX Runtime session's providers are exactly `["CPUExecutionProvider"]` (asserted).
2. `torch` is not in `sys.modules` (recorded as `torch_imported: false`).
3. With `CUDA_VISIBLE_DEVICES=` (empty), the process does not appear in `nvidia-smi --query-compute-apps` (`this_process_on_gpu: false`).
4. MediaPipe is set to `Delegate.CPU`.
5. ONNX vs torch outputs match to 4e-7 or better, and the runtime NumPy features match the training inputs to 7e-7 or better (asserted).

## 38–39. Hackathon recommendation

**Model (§38): M8**, the frozen Phase 5 weights already integrated into the Phase 6 kit.
- **Accuracy:** M8 + Cons is 1.92 vs 1.94 (P95 4.06 vs 4.10, pose invariance 2.72 vs 2.75). That is within one seed standard deviation, so they are statistically equivalent. M8 needs no new training and has already passed the Phase 6 pipeline checks, so it is less risky.
- **Why not M9:** worse on every metric (+0.25 cm), more sensitive to K, and no faster.
- **CPU:** the same 27–29 FPS, 85 KB.
- **Fallback:** if Phase 6 real data shows large landmark jitter, swap in **M8 + NoiseAug (σ 0.25 px)**. It costs +0.04 cm on clean input and roughly halves the degradation under 1 px noise.
- **Most promising candidate: M8 + EyePCA16.** It beats everything (5-point calibration 0.99 cm, pose invariance 1.69), but every synthetic person shares the same eye appearance. Adopt it only after confirming it on Phase 6 real data. The kit stores the same tgaze eye patch per frame, so it can be evaluated **offline without retaking**.

**Calibration points (§39): 5 points** (centre + corners at ±0.7, 2 samples each, about 10 seconds with the current timing).
- M8: 1.94 → 1.54 cm (−21%), P95 3.11.
- 9 points only goes to 1.50.
- 1 point (a few seconds, bias only) still gives 1.75, an emergency option.
- With 3 points, use the low-DOF bias + scale (1.57) rather than the affine (1.61).

## 40. Next steps after the real-device evaluation

1. Analyse the Phase 6 real data (zero-shot / 1 / 3 / 5 / 9 points) to measure the synthetic → real gap. **Compute the frame-to-frame jitter of real MediaPipe landmarks** and place it against the robustness table above (σ 0.25 / 0.5 / 1 px). That decides whether noise augmentation is needed.
2. **Evaluate M8 + EyePCA16 offline** on the eye patches stored by the kit. If it holds on real data, it becomes the first candidate. If it doesn't, the cause is a lack of synthetic appearance diversity: randomize iris colour, skin, lighting, glasses and eyelashes, then retrain.
3. **Tiny eye CNN + M8 geometry (§64 proposal).** Landmarks have clearly hit their information limit:
   - training error ≥ 1.05 cm;
   - M9 shows eye-in-head direction is the bottleneck;
   - EyePCA improves things a lot with a linear projection alone.

   Pass a 48 × 32 × 2 eye patch through a small CNN (tens of thousands of parameters, about 1 ms on CPU) and fuse it into M8's eye branch. A synthetic dataset with appearance diversity is required first.
4. **More identities:** 100 or more people (with eyeball and iris size variation). That helps the appearance model much more than the landmark model (§27–28).
5. **Temporal smoothing:** within each trial, median or EMA over the 800 ms window. Per-frame MediaPipe jitter (0.7 cm for a 0.5 px shift) should be averaged out there.
6. **Not recommended:** pose-consistency loss (it shrinks outputs and the scale-invariant version collapses) and the strict physical model M9. Both are kept here as recorded results.

---

## File map

| Content | Location |
|---|---|
| Data, features, evaluation | `phase7/p7data.py` |
| Training (M8 / Cons / M9 / NoiseAug / EyePCA) | `phase7/p7train.py` |
| Aggregation | `phase7/p7agg.py` → `<exp>/summary.json` |
| Synthetic calibration | `phase7/p7calib.py` → `<exp>/calibration.json`, `calibration/summary.json` |
| Robustness | `phase7/p7robust.py`, `phase7/p7noise_eval.py` → `robustness/` |
| Baselines | `phase7/p7baselines.py` → `baseline/B*` |
| Identity generation and QA | `phase7/run_idgen.sh`, `extract_one.py`, `qa_identities.py` → `identity_diversity/` |
| CPU export and benchmark | `phase7/p7export.py`, `phase7/p7onnx_bench.py` → `cpu_runtime/` |
| Plots | `phase7/p7plots.py` → `plots/`: model_comparison, pose_invariance, calibration_curve, identity_curve, learning_curve, train_val_curves, robustness_K, robustness_landmark_noise, distortion |
| Checkpoint run scripts | `phase7/run_ckpt*.sh`, `run_capacity.sh`, `run_idexp.sh` |
| Logs | `logs/` |
| Decision record | `consistency/lambda_selection.md`; head-frame diagnostics `diagnostics/head_frame.json` |

**Failed or ineffective experiments are kept, not deleted:**
- the consistency collapse runs (`consistency/pilot_*`);
- the M9 variants and oracles (`m9/`);
- the EyePCA + Cons collapse;
- the large no-dropout overfit.
