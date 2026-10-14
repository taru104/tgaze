# Calibration-method comparison (secondary / exploratory; primary = config.json)

Mean error [cm] over evaluation trials, averaged over subjects. Exploratory rows use evaluation trials as extra calibration and score the rest.

## live (frozen M8) (1 subjects)

| variant | NATURAL | EYE_DOMINANT | HEAD_ALLOWED | ALL |
|---|---:|---:|---:|---:|
| 0pt none | 2.74 | 4.10 | 2.83 | 3.22 |
| 1pt primary | 2.65 | 5.24 | 3.61 | 3.83 |
| 1pt bias | 2.65 | 5.24 | 3.61 | 3.83 |
| 1pt bias x-only | 2.87 | 4.29 | 3.16 | 3.44 |
| 1pt bias x0.5 | 2.61 | 4.65 | 3.17 | 3.48 |
| 3pt primary | 3.06 | 6.59 | 4.23 | 4.63 |
| 3pt bias | 2.61 | 5.86 | 3.74 | 4.07 |
| 3pt bias_scale l25 | 2.94 | 6.58 | 4.21 | 4.58 |
| 3pt affine l100 | 2.94 | 6.40 | 4.10 | 4.48 |
| 3pt affine l400 | 2.77 | 6.12 | 3.92 | 4.27 |
| 3pt bias x-only | 2.69 | 3.98 | 2.56 | 3.07 |
| 3pt bias x0.5 | 2.47 | 4.90 | 3.09 | 3.49 |
| 5pt primary | 2.42 | 5.39 | 3.41 | 3.74 |
| 5pt bias | 2.50 | 5.43 | 3.42 | 3.78 |
| 5pt bias_scale l25 | 2.49 | 5.44 | 3.47 | 3.80 |
| 5pt affine l100 | 2.43 | 5.40 | 3.42 | 3.75 |
| 5pt affine l400 | 2.45 | 5.42 | 3.43 | 3.77 |
| 5pt bias x-only | 2.69 | 3.99 | 2.61 | 3.10 |
| 5pt bias x0.5 | 2.49 | 4.71 | 3.00 | 3.40 |
| 9pt primary | 2.25 | 5.15 | 3.25 | 3.55 |
| 9pt bias | 2.47 | 5.19 | 3.24 | 3.63 |
| 9pt bias_scale l25 | 2.47 | 5.23 | 3.38 | 3.69 |
| 9pt affine l100 | 2.27 | 5.15 | 3.25 | 3.56 |
| 9pt affine l400 | 2.30 | 5.16 | 3.26 | 3.57 |
| 9pt poly2 l25 | 2.32 | 5.44 | 3.55 | 3.77 |
| 9pt bias x-only | 2.69 | 3.99 | 2.60 | 3.09 |
| 9pt bias x0.5 | 2.51 | 4.60 | 2.94 | 3.35 |
| 9pt + 1 EYE_DOMINANT bias | 2.47 | 5.00 | 3.10 | 3.50 |
| 9pt + 1 EYE_DOMINANT affine l25 | 2.22 | 4.88 | 3.10 | 3.38 |
| 9pt + 3 EYE_DOMINANT bias | 2.50 | 4.67 | 2.88 | 3.30 |
| 9pt + 3 EYE_DOMINANT affine l25 | 2.18 | 4.51 | 2.91 | 3.14 |
| 9pt + 5 EYE_DOMINANT bias | 2.55 | 4.43 | 2.74 | 3.16 |
| 9pt + 5 EYE_DOMINANT affine l25 | 2.19 | 4.23 | 2.75 | 2.97 |
| 9pt + 1 HEAD_ALLOWED bias | 2.47 | 5.09 | 3.16 | 3.58 |
| 9pt + 1 HEAD_ALLOWED affine l25 | 2.24 | 5.05 | 3.17 | 3.49 |
| 9pt + 3 HEAD_ALLOWED bias | 2.48 | 4.92 | 3.03 | 3.50 |
| 9pt + 3 HEAD_ALLOWED affine l25 | 2.22 | 4.84 | 3.04 | 3.38 |
| 9pt + 5 HEAD_ALLOWED bias | 2.50 | 4.80 | 2.95 | 3.45 |
| 9pt + 5 HEAD_ALLOWED affine l25 | 2.21 | 4.67 | 2.93 | 3.30 |

Linear response pred = offset + gain * target (cm), per subject:

- subject_00: NATURAL x gain 0.97 off +0.51, y gain 1.09 off +0.91; EYE_DOMINANT x gain 0.79 off +1.40, y gain 0.86 off -2.78; HEAD_ALLOWED x gain 0.81 off +1.73, y gain 0.91 off -0.55; cal NATURAL x gain 1.04 off +0.61, y gain 0.91 off +1.55
