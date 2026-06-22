# Supplementary Results

Per-subset AFC evaluation results for external baselines. All values are
seed0 unless noted otherwise. Metrics use Top-M=20, ε=0.5.

## ETH-UCY Per-Subset Breakdown

### Social-STGCNN

| Subset | minADE | minFDE | ADE_avg | FDE_avg | WMR | Precision | Unsupported | Chamfer |
|---|---|---|---|---|---|---|---|---|
| eth | 0.723 | 1.201 | 1.041 | 1.869 | 0.541 | 0.425 | 0.528 | 0.612 |
| hotel | 0.416 | 0.679 | 0.808 | 1.511 | 0.702 | 0.593 | 0.367 | 0.450 |
| univ | 0.489 | 0.912 | 0.741 | 1.416 | 0.658 | 0.565 | 0.328 | 0.470 |
| zara1 | 0.332 | 0.521 | 0.643 | 1.212 | 0.802 | 0.559 | 0.306 | 0.400 |
| zara2 | 0.303 | 0.481 | 0.565 | 1.052 | 0.840 | 0.698 | 0.225 | 0.342 |

### TUTR

| Subset | minADE | minFDE | ADE_avg | FDE_avg | WMR | Precision | Unsupported | Chamfer |
|---|---|---|---|---|---|---|---|---|
| eth | 0.415 | 0.613 | 1.578 | 3.218 | 0.652 | 0.261 | 0.705 | 0.663 |
| hotel | 0.127 | 0.178 | 0.707 | 1.528 | 0.766 | 0.408 | 0.503 | 0.485 |
| univ | 0.235 | 0.432 | 1.025 | 2.255 | 0.789 | 0.286 | 0.620 | 0.546 |
| zara1 | 0.190 | 0.353 | 0.750 | 1.653 | 0.828 | 0.470 | 0.408 | 0.420 |
| zara2 | 0.144 | 0.260 | 0.928 | 2.075 | 0.917 | 0.336 | 0.616 | 0.415 |

### GraphTERN

| Subset | minADE | minFDE | ADE_avg | FDE_avg | WMR | Precision | Unsupported | Chamfer |
|---|---|---|---|---|---|---|---|---|
| eth | 0.382 | 0.485 | 1.381 | 2.564 | 0.805 | 0.210 | 0.751 | 0.646 |
| hotel | 0.142 | 0.231 | 0.837 | 1.667 | 0.852 | 0.342 | 0.569 | 0.453 |
| univ | 0.237 | 0.378 | 0.947 | 1.906 | 0.874 | 0.297 | 0.597 | 0.473 |
| zara1 | 0.195 | 0.332 | 0.780 | 1.703 | 0.906 | 0.469 | 0.410 | 0.370 |
| zara2 | 0.156 | 0.251 | 0.774 | 1.570 | 0.934 | 0.446 | 0.488 | 0.341 |

### EigenTrajectory-LBEBM

| Subset | minADE | minFDE | ADE_avg | FDE_avg | WMR | Precision | Unsupported | Chamfer |
|---|---|---|---|---|---|---|---|---|
| eth | 0.364 | 0.579 | 1.528 | 3.279 | 0.582 | 0.220 | 0.741 | 0.724 |
| hotel | 0.127 | 0.202 | 0.866 | 1.928 | 0.845 | 0.357 | 0.560 | 0.443 |
| univ | 0.261 | 0.470 | 1.123 | 2.429 | 0.827 | 0.247 | 0.659 | 0.561 |
| zara1 | 0.194 | 0.353 | 0.924 | 2.014 | 0.932 | 0.373 | 0.534 | 0.402 |
| zara2 | 0.138 | 0.246 | 0.955 | 2.232 | 0.941 | 0.354 | 0.595 | 0.411 |

### MID

| Subset | minADE | minFDE | ADE_avg | FDE_avg | WMR | Precision | Unsupported | Chamfer |
|---|---|---|---|---|---|---|---|---|
| eth | 0.270 | 0.491 | 0.656 | 1.530 | 0.599 | 0.432 | 0.389 | 0.544 |
| hotel | 0.182 | 0.306 | 0.460 | 1.025 | 0.719 | 0.642 | 0.246 | 0.419 |
| univ | 0.237 | 0.507 | 0.483 | 1.160 | 0.572 | 0.463 | 0.261 | 0.523 |
| zara1 | 0.210 | 0.425 | 0.595 | 1.440 | 0.739 | 0.487 | 0.334 | 0.440 |
| zara2 | 0.152 | 0.322 | 0.465 | 1.152 | 0.833 | 0.683 | 0.229 | 0.314 |

### SingularTrajectory

| Subset | minADE | minFDE | ADE_avg | FDE_avg | WMR | Precision | Unsupported | Chamfer |
|---|---|---|---|---|---|---|---|---|
| eth | 0.354 | 0.425 | 1.472 | 2.772 | 0.605 | 0.244 | 0.702 | 0.843 |
| hotel | 0.129 | 0.196 | 0.877 | 1.879 | 0.829 | 0.358 | 0.546 | 0.460 |
| univ | 0.261 | 0.451 | 1.195 | 2.480 | 0.837 | 0.264 | 0.658 | 0.572 |
| zara1 | 0.188 | 0.318 | 0.918 | 1.960 | 0.921 | 0.393 | 0.508 | 0.409 |
| zara2 | 0.148 | 0.253 | 1.018 | 2.239 | 0.920 | 0.332 | 0.613 | 0.458 |

## SDD External Models — Per-Seed Breakdown

### TUTR (SDD, seed0/1/2, same-runner)

TUTR is deterministic at this checkpoint, producing identical values across seeds.

| Metric | Value |
|---|---|
| ADE_min | 0.273 |
| FDE_min | 0.458 |
| ADE_avg | 1.089 |
| FDE_avg | 2.210 |
| WMR | 0.607 |
| Precision | 0.398 |
| Unsupported | 0.516 |
| Chamfer | 2.868 |

### MID (SDD, seed1/2, scale=1.0)

Values are mean ± std. MID seed0 WMR was 0.288 (notably higher than
seed1/2), so per-seed variation is material.

| Metric | Value |
|---|---|
| ADE_min | 0.177 ± 0.002 |
| FDE_min | 0.340 ± 0.006 |
| ADE_avg | 0.428 ± 0.002 |
| FDE_avg | 0.918 ± 0.005 |
| WMR | 0.173 ± 0.000 |
| Precision | 0.369 ± 0.001 |
| Unsupported | 0.206 ± 0.002 |
| Chamfer | 8.697 ± 0.000 |

### TrajEvo (SDD, seed0/1/2, official generalization protocol)

TrajEvo uses the official ETH→SDD generalization heuristic (scale factor
100), not an SDD-trained model. Small per-seed variance confirms stable
heuristic behavior.

| Metric | Value |
|---|---|
| ADE_min | 10.011 ± 0.023 |
| FDE_min | 17.306 ± 0.099 |
| ADE_avg | 24.446 ± 0.047 |
| FDE_avg | 46.899 ± 0.050 |
| WMR | 0.052 ± 0.000 |
| Precision | 0.149 ± 0.000 |
| Unsupported | 0.850 ± 0.000 |
| Chamfer | 29.395 ± 0.010 |

## UNSUPPORTED-Pure Ablation

The paper's `tab:corr` footnote states that a pure-AFC version of Unsupported
(without the GT exemption) yields directionally consistent trends with the
default version. The GT exemption prevents candidates close to the observed GT
from being counted as unsupported even when no analogical mode covers them.
The pure version (`unsupported = ~pred_matched_mode` without the `gt_ade >
eps` check) produces higher absolute unsupported ratios but preserves the
same relative ordering across models and diagnostic branches. The exemption
is a conservative design choice; the pure variant is available in the
evaluation code for users who prefer it.

## Retrieval Feature: Social-Risk Cues

The default `full_past_social` feature includes the following 10 social-risk
cues, computed from observed past positions and constant-velocity
extrapolation. Each cue is normalized per-agent before concatenation.

| # | Cue | Description |
|---|---|---|
| 1 | `nearest_distance` | Distance to closest neighbor, normalized to [0,1] |
| 2 | `nearest_distance_inv` | Inverse-distance weight of the closest neighbor |
| 3 | `mean_distance_inv` | Average inverse-distance weight across all neighbors |
| 4 | `close_count_0p5` | Proportion of neighbors within 0.5-unit radius |
| 5 | `close_count_1p0` | Proportion of neighbors within 1.0-unit radius |
| 6 | `soft_density_sigma1` | Gaussian-kernel soft density (σ=1.0) |
| 7 | `max_approaching_speed` | Maximum closing speed across neighbors |
| 8 | `mean_approaching_speed` | Mean closing speed across neighbors |
| 9 | `min_tca_inv` | 1/(1 + min TCA), TCA = time to closest approach |
| 10 | `min_constvel_distance_inv` | 1/(1 + minimum projected distance under constant velocity) |

These features are defined in `trustmoe_traj/data/transforms.py`
(`PAST_SOCIAL_RISK_FEATURE_NAMES`) and consumed by
`trustmoe_traj/scripts/analogical_future_coverage.py`.

## Raw Data

Full experiment CSV files are available under
`data_results/afc_experiments_1_7_20260612/`. The directory includes
per-experiment subdirectories with per-dataset, per-seed, and per-branch
breakdowns.
