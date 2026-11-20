# Pilot check: subject_00

- [x] webcam resolution [1920, 1080] (backend AVFOUNDATION, Center Stage disabled: True)
- [x] measured K applied (host taruyaryoutarounoMacBook-Air vs json taruyaryoutarounoMacBook-Air; |K - training K| = 0.0 px)
- [x] mirrored=True; set_camera_matrix(K_raw, (1920,1080), mirrored=True)
- [x] ONNX Runtime providers ['CPUExecutionProvider'] (available: ['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']); MediaPipe delegate CPU; torch loaded: False
- [x] geometry / target coordinate check confirmed with a ruler: True layout {'screen_pt': [1470, 956], 'window_pt': [0, 32, 1470, 924], 'canvas_px': [1470, 924], 'display_mm': [290.6, 189.0], 'mm_per_pt': [0.19768707482993197, 0.19769874476987448]}
- [x] direction sanity check: right-left 24.346690247505904 cm, up-down 15.961696303635835 cm
- [x] cm conversion: uv (0.8, 0.8) -> (11.624, 7.560) cm from display centre (290.6 x 189.0 mm)
- [x] record timing: frames per window median move/settle/record = [15.0, 21.0, 24.0], camera FPS median 30.01694652674061
- [x] M8 ONNX CPU inference ran: mean 0.06387656846046036 ms
- [x] MediaPipe ran: mean 5.805296161210791 ms
- [x] calibration fitted on device: {'1': 0.103, '3': 0.12, '5': 0.08, '9': 0.075} ms; tgaze 9-pt: {'fit_ms_including_tgaze_LOO': 44530.50491702743, 'n_frames': 432}
- [x] result saving: metadata.jsonl, per-trial npz, representative frames

status counts: {'calibration': {'VALID': 18}, 'evaluation': {'RETAKE_RECOMMENDED': 9, 'VALID': 66, 'INVALID': 1}}

## Calibration-only LOO lambda sweep (pilot only; evaluation trials not used)

| model | lambda cm^2 | LOO error cm (9 cal points) |
|---|---|---|
| affine | 0 | 2.64 |
| affine | 5 | 2.63 |
| affine | 25 | 2.59 |
| affine | 100 | 2.49 |
| affine | 400 | 2.29 |
| bias_scale | 0 | 2.22 |
| bias_scale | 5 | 2.21 |
| bias_scale | 25 | 2.21 |
| bias_scale | 100 | 2.19 |
| bias_scale | 400 | 2.13 |

zero-shot error on the pilot calibration trials: 2.51 cm

Change config.json (calibration.models.*.lambda_cm2, timing, quality) only now, before main subjects. After that the frozen_protocol_hash must stay the same for every main subject.
