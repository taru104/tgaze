# Phase 7A lambda selection (pilot: folds 0,1 x seeds 0,1; decision on VALIDATION identities only)

Rule (fixed before looking at test): minimise validation (cm_mean + 0.5 * pose_invariance), the Phase 5 weighting.
Collapse (lambda >= 2 mean_dev, >= 0.5 scaled, 2 pairwise) = predictions shrink to a constant; rejected.

| variant | val cm | val poseInv | score |
|---|---|---|---|
| mean_dev 0    | 1.566 | 2.636 | 2.884 |
| mean_dev 0.05 | 1.570 | 2.627 | 2.884 |
| mean_dev 0.1  | 1.573 | 2.674 | 2.910 |
| mean_dev 0.25 | 1.627 | 2.510 | 2.882 |
| mean_dev 0.5  | 1.658 | 2.479 | 2.898 |
| mean_dev 1    | 1.872 | 2.198 | 2.971 |
| pairwise 0.5  | 1.799 | 2.265 | 2.932 |
| scaled 0.25   | 1.542 | 2.655 | **2.870** |

Selected: scaled consistency, lambda 0.25 (M8+Cons). Also run in full CV for the trade-off: mean_dev lambda 0.5.
Finding: raw pose-invariance gains of the plain loss come mainly from output compression (pred/true spread
0.91 -> 0.80 at lambda 1 -> 0.47 at lambda 2); spread-normalised invariance changes by only a few %.
