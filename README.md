> **提出用（`submission` ブランチ）**：Web アプリの起動方法と内容は [SUBMISSION.md](SUBMISSION.md) を参照してください。

<h1 align="center">Tgaze</h1>

<p align="center">
  <b>True Gaze</b> — webcam eye tracking that tells you <b>where on the screen</b> you are looking.
</p>

https://github.com/user-attachments/assets/b87e13b7-bd6f-421d-bcb7-b413bd32e0a4

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/backbone-MediaPipe-orange.svg" alt="mediapipe">
  <img src="https://img.shields.io/badge/runtime-CPU%20realtime-green.svg" alt="cpu">
  <img src="https://img.shields.io/badge/no-GPU%20required-success.svg" alt="no gpu">
  <img src="https://img.shields.io/badge/license-source--available%20(permission%20required)-red.svg" alt="license">
</p>

---

## TL;DR

Most open webcam gaze projects stop at *"you're looking left"* or at a **gaze angle**. What people
actually want is **the point on the screen** — and that is what Tgaze outputs.

```python
from tgaze import GazeTracker

tracker = GazeTracker()
tracker.calibrate()                 # fullscreen 9-point calibration, once
x, y = tracker.predict(frame)       # normalized screen coordinates in [0, 1]
```

- 🎯 **~2.6 cm** on screen after a 9-point personal calibration
- 👥 **Works for someone it has never seen** — ~4.2 cm / 6.2° with **zero** calibration
- 🪶 **CPU, real-time** — MediaPipe + scikit-learn. No GPU, no deep net at runtime
- 🔒 **Private by design** — everything runs locally, nothing is uploaded
- 🧪 **Honestly measured** — every number below states its protocol, and the failed
  experiments are published too

## 🚀 Quick start

> ⚠️ **Read the [licence](#license) first.** Tgaze is source-available, not open source:
> **running it requires permission**, including for personal, educational and research use.
> [Open an issue](https://github.com/taru104/tgaze/issues) to ask — noncommercial requests
> are normally granted.

```bash
git clone https://github.com/taru104/tgaze.git
cd tgaze
pip install -e .                  # or: pip install -r requirements.txt

python examples/quickstart.py     # calibrate, then watch the gaze dot follow you
```

<sub>`pip install tgaze` from PyPI is not available yet — the runtime needs the landmark model
file that ships with this repo. Installing from a clone is the supported path today.</sub>

Or run the full app with its HUD, head-pose preview and live accuracy readout:

```bash
python main.py            # default: geometry + eye-image (steadier)
python main_16d.py        # geometry only (the leaner baseline)
```

| Key | Action |
|---|---|
| `C` | Calibrate — follow the dot through **9 points, clockwise** |
| `M` | Multi-pose calibrate (look at each dot while slowly rotating your head) |
| `R` / `Q` | Reset / Quit |

### Library API

```python
from tgaze import GazeTracker

tracker = GazeTracker()                    # video=True by default (webcam stream)

tracker.calibrate()                        # or build your own UI:
tracker.add_calibration_sample(frame, (0.5, 0.5))   #   "this frame was looking at the center"
tracker.fit()

gaze = tracker.predict(frame)
gaze.x, gaze.y                             # [0, 1] screen coordinates
gaze.face_detected, gaze.head_pose_ok
gaze.as_pixels(1920, 1080)

tracker.record_click((0.3, 0.7))           # a click is free ground truth — corrects drift
tracker.save("me.tgaze")                   # skip calibration next time
```

> Scoring **still images** rather than a video stream? Use `GazeTracker(video=False)`.
> MediaPipe's video mode carries tracking state between frames, which silently corrupts
> unordered images — this flag switched our own batch evaluation from 10.4 cm to 2.6 cm.

## 📊 Accuracy

Every row states its protocol, because in gaze estimation the protocol *is* the number.

**On MPIIFaceGaze** (15 subjects, 37,595 frames, offline — [`experiments/REPORT8_metrics.md`](experiments/REPORT8_metrics.md)):

| Setting | Screen error | Angular error |
|---|---:|---:|
| Same person, calibrated *(time-split: train early, test later)* | **3.19 cm** | **3.65°** |
| **New person, zero calibration** *(leave-one-person-out)* | 4.19 cm | 6.24° RMSE |
| New person, **16 calibration samples** | **3.51 cm** | **5.73° RMSE** |

**Live webcam**, personal 9-point calibration, leave-one-point-out (the on-screen HUD number):

| Model | Median error | Off-screen jump rate |
|---|---:|---:|
| Geometry only (16-D) | 1.6 cm | ~12 % |
| **Geometry + eye-image (default)** | 1.3 – 2.1 cm | **~4–6 %** |

<sub>Live numbers are single-operator sessions on one machine — treat them as indicative, not as a
benchmark. A friend using it for the first time reached ~2.0 cm.</sub>

### How that compares

The closest published work in the same regime (**landmarks only, no GPU, few-shot per-session
calibration**) is **EMC-Gaze** ([arXiv 2603.12388](https://arxiv.org/html/2603.12388), 2026), which
reports MPIIFaceGaze leave-one-person-out with 16 calibration samples:

| Method | 16-shot LOPO (subject-macro angular RMSE) |
|---|---:|
| Elastic Net on raw landmarks *(reported in that paper)* | 10.83° |
| EMC-Gaze — E(3)-equivariant GNN + meta-calibration | 8.82° ± 1.21 |
| **Tgaze — 16-D + eye patch, linear Huber, CPU** | **5.73° ± 0.98** |

> ⚠️ **Read this before quoting the table.** The paper's MPII numbers may come from an encoder
> trained on their own dataset and transferred to MPII, whereas Tgaze trains on the other 14 MPII
> subjects — that would make our setting *easier*, and the comparison not apples-to-apples. Frame
> counts and preprocessing are not fully specified either. The defensible claim is
> **"a 16-D linear model on a CPU reaches this range on the same public benchmark"** — not
> "Tgaze beats the state of the art."

## 🔍 How it works

```
webcam frame
  └─ MediaPipe FaceLandmarker (478 landmarks + iris)
       ├─ 16-D geometric feature
       │    iris(x,y) L/R, head pitch/yaw/roll, inter-eye distance,
       │    eye-openness, iris vertical position, iris size, iris aspect
       │    (iris normalized to each eye's corners → distance/translation invariant)
       └─ (default) eye-image patch
            each eye warped to a canonical 48×32 crop by its corners,
            CLAHE contrast-normalized, compressed to 16 PCA components
       └─ Huber regression (robust) → on-screen point
       └─ One-Euro filter → smooth gaze cursor
```

**Insight 1 — the bug that unlocked accuracy.** An early feature normalized the iris against the
**image center**, quietly turning it into a *face-position sensor*: its correlation with head
location was **0.98**, with actual gaze only **0.48**. Switching to **eye-corner normalization**
made it distance- and translation-invariant and cut center error from **~9 cm → ~1.4 cm**.
*The iris carries the gaze signal; head pose is only a correction term.*

**Insight 2 — the metric and the feel disagreed, and the feel was right.** Adding the eye-image
patch *slightly hurt* the point-accuracy number in some sessions, so by the number alone you would
reject it. But in hands-on use it was clearly steadier, so I logged both versions and measured the
real culprit — **wild off-screen jumps** — and found the eye-image model cut them roughly in half.
One offline metric can miss what makes a tool usable. It ships as the default because it *behaves*
better, and the logs backed that up.

**Insight 3 — my own metric was wrong, in my favour to fix.** Every MPII number here was computed
by scaling normalized error by a flat 30 cm — on screens that are 28.6 × **17.9** cm. That inflated
vertical error by 1.68×. Fixing it moved person-specific error from a reported 4.21 cm to an actual
**3.19 cm**. The lesson is not "the numbers got better"; it is that a metric nobody audits will
quietly decide what you believe.

## ⚖️ Limitations (please read before trusting it)

- **Calibration is per-session-ish.** Move a lot, or change your distance from the screen a lot,
  and accuracy degrades. `record_click()` fights the drift, a re-calibration fixes it.
- **Large head rotation is the weak spot.** Beyond roughly ±30° yaw the error grows substantially.
- **Sub-centimetre is not reachable with this design.** We tried: 9 additional methods
  (`experiments/REPORT5`), limbus ellipse fitting (`REPORT7`), and a resolution sweep that flattens
  at 48×32 (`REPORT6/exp61`). Hand-crafted features + linear regression saturate here; going lower
  needs a learned CNN representation.
- Numbers come from one operator, one friend, and MPIIFaceGaze. That is not a user study.

## 🧪 Research notes

- **Features** — `rich16d.py` (16-D geometry, single source of truth), `appearance.py`
  (canonical eye patch → PCA-16). `calibration.py` fits robust Huber regression on the fusion.
- **Why not more dimensions?** Ablations in `experiments/` show hand-crafted features saturate,
  and raising eye-image resolution helps *only* under tight PCA compression — more raw dimensions
  overfit a sparse calibration. Overfitting was the constant adversary, so gains are checked
  person-independently.
- **Negative results are published too** — they are the most useful part of a research repo:
  limbus ellipse fitting loses to MediaPipe's learned landmarks (`REPORT7`), pose-interaction and
  vergence features add nothing (`REPORT6`), and eye-image resolution stops paying off past 48×32.
- Reproduce the benchmark: `benchmarks/extract_mpii_full.py` then `experiments/exp63_metrics.py`,
  `exp64_lopo_kshot.py`.

## 🗺️ Roadmap

- [x] Real-webcam validation on a second person (cold first-time user)
- [x] Literature-comparable evaluation in degrees on MPIIFaceGaze
- [x] Library API (`tgaze`) + save/load calibration
- [ ] PyPI release (`pip install tgaze`)
- [ ] Shrinkage-guarded few-shot adaptation shipped into the live app
- [ ] Denser / continuous calibration to also win the point-accuracy number
- [ ] Eye-crop logging (privacy-scoped) to tune the appearance model on real sessions

## 📚 Citation

```bibtex
@software{tgaze,
  title  = {Tgaze: True Gaze — calibration-light webcam eye tracking},
  author = {taru104},
  year   = {2026},
  url    = {https://github.com/taru104/tgaze}
}
```

## License

© 2026 taru104. **All rights reserved.** Tgaze is **source-available, not open source** —
see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

You may **read and study** the code freely. **Any other use — running it, copying it,
modifying it, building on it — requires prior written permission**, and that includes
personal, educational, academic and research use. No category is pre-authorised.

**To ask, [open an issue](https://github.com/taru104/tgaze/issues)** saying who you are and
what you want to do. If it is noncommercial and reasonable, expect a yes — the point is that
I want to know where Tgaze is used, not to make it difficult.

**Commercial use is not licensed** and commercial licences are not offered.

If permission is granted, crediting Tgaze is mandatory: any resulting software, paper, demo,
article or talk must say *"This work uses Tgaze by taru104 —
https://github.com/taru104/tgaze"*. Presenting results obtained with Tgaze without naming it
is a breach of the licence.

Third-party components keep their own licences — MediaPipe and the bundled
`face_landmarker.task` model are © Google LLC under Apache 2.0.

## Acknowledgments

- Built on [MediaPipe](https://developers.google.com/mediapipe) FaceLandmarker.
- Compared against [EyeTrax](https://github.com/ck-zhang/eyetrax) and
  [GazeTracking](https://github.com/antoinelame/GazeTracking).
- Research used **MPIIFaceGaze** (Zhang et al., CVPRW 2017; research license) and draws on
  **GazeCapture** and **ETH-XGaze**. These research datasets are **not** used in any commercial
  build and are not redistributed here.
