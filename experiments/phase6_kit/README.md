# Phase 6: real MacBook Air M4 validation kit

This kit measures the frozen Phase 5 M8 model (`model/final_cpu_model.onnx`) on the real built-in webcam of the MacBook Air 13" M4. It compares no personal calibration (zero-shot) with 1, 3, 5 and 9-point personal calibration, and with the current tgaze model (16D + eye-patch PCA + Huber) on the same frames.

Rules the kit follows:
- **M8 is never trained.** The ONNX sha256 is stored in every subject's `subject.json`.
- **CPU only:**
  - ONNX Runtime uses `providers=["CPUExecutionProvider"]`;
  - MediaPipe uses `Delegate.CPU`;
  - torch is never imported (asserted).
- **tgaze is only read.** Its source files are imported, never modified:
  - `rich16d` and `set_camera_matrix(K, mirrored=True)`, as in `main_camK_mac.py`;
  - `appearance.py`, for the baseline;
  - `macos.py`, for Center Stage off and the machine id.

## Setup on the Mac (once)

```bash
cd ~/tgaze && git fetch origin && git switch experiment/phase6-kit   # kit lives in experiments/phase6_kit
cd ~/tgaze/experiments/phase6_kit
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements_mac.txt
```

`config.json` → `paths.tgaze_dir` points at `~/tgaze`; the `TGAZE_DIR` environment variable overrides it.

The measured K is used only when `scutil --get LocalHostName` matches the `host` field of `camera_matrix.json`. This is the same rule `main_camK_mac.py` uses. If it doesn't match, the app stops.

Before you start:
- **Display:** use the built-in display at the default "Looks like 1470 x 956" scaling, with no external monitor as the main display. The app records the actual NSScreen size, so another scaling still converts correctly, but stick to the default.
- **Camera access:** macOS asks for camera permission for the terminal the first time.

## Pilot (1 person, before anyone else)

```bash
python collect.py --subject subject_00 --pilot
```

Steps, in order:
1. **G, geometry check.** Measure with a ruler:
   - the 100 mm and 50 mm lines;
   - the top-left cross position;
   - the spacing between the two top crosses.

   Press Y if all three are correct.
2. **D, direction check.** Look at the left, right, up and down targets. Zero-shot M8 must move right > left (by more than 2 cm) and up > down (by more than 1 cm). Calibration and evaluation refuse to start until this passes. **L** (live cursor) lets the operator check this by eye as well.
3. **C, calibration session:** 9 points (±0.7) × 2 repeats in random order, NATURAL posture. The corrections are then fitted on the Mac and their fit time is recorded in `calibration_model.json`.
4. **E, evaluation session (pilot):** 3 blocks (NATURAL, EYE_DOMINANT, HEAD_ALLOWED) × 25 targets.
5. **P, CPU benchmark:** 300 live frames. Records camera FPS, per-stage latency, effective FPS, CPU and memory.

Then:

```bash
python analyze.py outputs/real_mac_validation --pilot-check subject_00
```

This writes `results/pilot_check_subject_00.md`. It is a checklist covering:
- resolution, K, mirroring, providers;
- the geometry and direction checks, cm conversion;
- record timing, MediaPipe and ONNX latency;
- the calibration fit, saved files.

It also includes a λ sweep that uses calibration trials only. Change `config.json` (timing, quality thresholds, `lambda_cm2`) **now, if at all**. After the pilot the protocol is frozen: every main subject stores `frozen_protocol_hash`, and `analyze.py` refuses to mix hashes.

## Main subjects (5, expandable)

```bash
python collect.py --next-subject      # creates subject_01, subject_02, ...
```

Per subject, run G (only if the setup changed), D, C, E and P.

Evaluation is 4 blocks × 25 targets (5×5 grid, ±0.8 / ±0.4 / 0), each block in random order:
- odd subject numbers: NATURAL, EYE_DOMINANT, HEAD_ALLOWED, NATURAL;
- even subject numbers: NATURAL, HEAD_ALLOWED, EYE_DOMINANT, NATURAL.

Every target is therefore seen under all three instructions. The repeated NATURAL block gives the test-retest noise floor for the pose-invariance metric.

Timing per trial is configurable in `config.timing`. By default:
- the target shrinks into place for 500 ms;
- 700 ms of settling follows;
- then 800 ms of recording.

The trial prediction is the median over the recording window. Frames are never treated as independent samples.

**Interrupting a session:**
- Esc pauses at any time;
- in the pause screen: Space continues, R retakes the previous trial, B returns to the menu;
- sessions resume where they stopped.

**Retakes:**
- INVALID trials are re-queued automatically at the end of their block, at most twice;
- RETAKE_RECOMMENDED trials are kept and flagged.

The quality flags never look at the target or the error:
- face or landmark loss;
- head movement inside the recording window;
- prediction jump;
- prediction spread.

Only anonymous ids (`subject_XX`) are stored. Each trial saves:
- one representative JPEG;
- an npz with per-frame landmarks, rich16d, head rotation, eye patch, M8 prediction and latencies.

Set `storage.save_all_record_frames` to also keep every recording-window JPEG for re-extraction later.

## Analysis

```bash
python analyze.py outputs/real_mac_validation               # primary
python analyze.py outputs/real_mac_validation --secondary   # + distortion ON/OFF, EMA/median smoothing,
                                                            #   bias / bias+scale / poly2 corrections, tgaze 5-pt, rich16d-Huber
```

Outputs go to `results/`:
- `report.md`, `summary.json`;
- `calibration_comparison.csv`, `subject_summary.csv`, `zero_shot.csv`;
- `trials.csv`, which holds the raw and every calibrated prediction per trial;
- `pose_invariance.csv`, `jitter.csv`.

Plots go to `plots/`:
- `calibration_curve`, `error_by_subject`, `error_by_screen_region`;
- `error_by_head_pose`, `error_by_pose_instruction`, `pose_invariance`;
- `synthetic_vs_real`, `prediction_scatter`, `jitter`.

How the analysis computes each correction:
- **Calibration models:** pre-registered in `config.calibration.models`.
  - 1-pt: 2D bias.
  - 3, 5 and 9-pt: affine in display cm, ridge-shrunk toward the identity (λ = 25 cm²).
  - Secondary variants: bias+scale, bias, and 9-pt quadratic.
- **What the corrections are fitted on:** the subject's own calibration session trials only.
- **Current tgaze baseline:** tgaze's `AppearancePipeline` is fitted on all recording frames of the 9-point calibration trials and predicts the same evaluation frames.

## Testing without a Mac

```bash
python collect.py --subject subject_90 --pilot --fake-camera ../outputs/gaze_dataset_v2 --fake-identity 7 \
       --headless --auto G,Y,D,C,E,P,Q --allow-k-host-mismatch --out /tmp/rv
python analyze.py /tmp/rv --secondary
```

This mode feeds Phase 5 synthetic images of an unseen identity that match the target on screen. Results from it are only a pipeline test and are marked FAKE CAMERA in the report.

## Packing for the Mac

In tgaze the kit is used directly from `experiments/phase6_kit` (no tarball). The model files in `model/` are the
frozen Phase 5 M8 (ONNX sha256 db19af0b...c25e); `lib/landmark_features.py` is a copy of the generator repo's
`scripts/landmark_features.py`. Recordings go to `outputs/` inside the kit, which is git-ignored.
