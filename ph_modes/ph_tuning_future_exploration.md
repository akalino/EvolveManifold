# Future Exploration: Tuning Scalable Persistent-Homology Monitoring

The runtime--fidelity comparison suggests that `event_driven` and `online_landmark_dynamic_support` are the most promising candidates for scalable persistent-homology (PH) monitoring. However, neither should be treated as a final production setting without further tuning. The next step is to perform a small, controlled parameter sweep on the existing full-VR calibration panel, holding the checkpoint set fixed and varying only the PH approximation parameters.

The goal of tuning is not to make the shortcut numerically identical to full Vietoris--Rips persistence. Instead, the goal is to identify the cheapest approximation that preserves the decision-relevant structure of the full-VR signal: rank ordering, epoch-wise trends, transition timing, mechanism separation, and relationships to cheaper geometric or spectral metrics.

## Tuning Objectives

Each candidate setting should be evaluated against the existing `full_vr` reference using:

- Median Spearman correlation with full VR across PH summary metrics.
- Median Spearman correlation of epoch-to-epoch changes.
- Median normalized mean absolute error relative to full VR.
- Median transition epoch absolute error.
- Runtime speedup relative to full VR.
- Failure behavior on topology-sensitive cases, especially `torus + hole_fill`.

A useful approximation may have biased absolute persistence values while still preserving the ordering and temporal structure needed for downstream prediction. Therefore, fidelity should be judged primarily by rank preservation, trend preservation, transition timing, and downstream usefulness rather than exact numerical equality.

## Key Parameters

The current PH workflow exposes several tuning parameters:

```python
_sparse = 0.2
_n_landmarks = 500
_skip_every = 2
_knn_k = 24
_event_thresh = 0.01
_event_max_skip = 5
_force_every = 5
```

For `event_driven`, the most important parameters are:

```text
event_thresh
event_max_skip
force_every
```

These determine when PH is recomputed.

For `online_landmark_dynamic_support`, the most important parameters are:

```text
n_landmarks
knn_k
event_thresh
event_max_skip
force_every
```

These determine both the support used to approximate PH and the refresh policy over time.

## Tranche A: Event-Driven Recompute Tuning

The purpose of this tranche is to determine whether event-triggered recomputation can preserve the full-VR signal while reducing computation. The current result suggests excellent fidelity but only modest speedup, meaning the trigger may be too conservative.

Recommended initial settings:

| Setting | `event_thresh` | `event_max_skip` | `force_every` | Interpretation |
|---|---:|---:|---:|---|
| A1 | 0.005 | 2 | 3 | Very conservative |
| A2 | 0.010 | 5 | 5 | Current baseline |
| A3 | 0.020 | 5 | 5 | Slightly cheaper |
| A4 | 0.050 | 5 | 5 | Aggressive threshold |
| A5 | 0.010 | 10 | 10 | Cheaper temporal reuse |
| A6 | 0.020 | 10 | 10 | Aggressive cheap mode |

Interpretation:

```text
If A1 is much better than A2:
    The current event threshold is too loose.

If A2 and A3 are similar:
    Prefer A3 for scaling.

If A4 fails:
    event_thresh = 0.05 is too aggressive.

If A5 or A6 preserve fidelity:
    PH can be recomputed less often on larger runs.
```

The expected useful region is likely around:

```text
event_thresh = 0.01 or 0.02
event_max_skip = 5
force_every = 5
```

## Tranche B: Online Dynamic Landmark Support Tuning

The purpose of this tranche is to determine whether dynamic landmark support can provide the best runtime--fidelity tradeoff for large-scale monitoring. The current result suggests substantial speedup and acceptable rank fidelity, but weaker temporal fidelity and transition timing.

Recommended initial settings:

| Setting | `n_landmarks` | `knn_k` | `event_thresh` | `event_max_skip` | `force_every` | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| B1 | 300 | 16 | 0.010 | 5 | 5 | Cheap support |
| B2 | 500 | 24 | 0.010 | 5 | 5 | Current baseline |
| B3 | 750 | 32 | 0.010 | 5 | 5 | Conservative support |
| B4 | 500 | 16 | 0.010 | 5 | 5 | Test sparse graph support |
| B5 | 500 | 32 | 0.010 | 5 | 5 | Test denser graph support |
| B6 | 500 | 24 | 0.020 | 10 | 10 | Aggressive dynamic reuse |

Interpretation:

```text
If B1 works:
    300 landmarks and k = 16 may be sufficient for large-scale exploratory monitoring.

If B2 works but B1 fails:
    500 landmarks and k = 24 is a reasonable minimum.

If B3 is much better:
    The topology is landmark-sensitive, and scaling will require more support.

If B4 fails but B5 works:
    The kNN support is too sparse at k = 16.

If B6 works:
    The dynamic support policy is robust enough for cheaper large-grid monitoring.
```

For the current benchmark size (`n = 1000`, `d = 50`), `n_landmarks = 500` and `knn_k = 24` is a reasonable baseline. The main scaling question is whether cheaper settings such as `n_landmarks = 300` or `knn_k = 16` preserve enough fidelity.

## Implementation Plan

The measurement script should be patched so that the PH workflow parameters are configurable from the command line rather than hard-coded. Useful CLI arguments would include:

```bash
--n-landmarks
--knn-k
--sparse
--event-thresh
--event-max-skip
--force-every
--skip-every
--run-tag
```

Each tuned setting should write to a distinct output directory so that the comparison script can treat it as a separate mode. For example:

```text
metric_outputs/event_driven__eth0p01_maxskip5_force5/
metric_outputs/event_driven__eth0p02_maxskip10_force10/
metric_outputs/online_landmark_dynamic_support__lm500_k24_eth0p01_maxskip5_force5/
```

This avoids overwriting previous measurements and makes downstream comparison straightforward.

## Decision Criteria

After tuning, each setting should be assigned to one of three categories.

### Green: scale candidate

A setting is a strong scale candidate if it satisfies approximately:

```text
median_spearman_vs_full_vr >= 0.90
median_delta_spearman_vs_full_vr >= 0.75
median_normalized_mae_vs_full_vr <= 0.25
median_transition_epoch_abs_error <= 1 checkpoint interval
runtime speedup is materially better than full VR
```

### Yellow: exploratory scale candidate

A setting is useful for exploratory scaling if it satisfies approximately:

```text
median_spearman_vs_full_vr >= 0.80
trend fidelity is acceptable
normalized error is not catastrophic
failure cases are understood
```

Yellow settings can be used for broad exploratory scans, but a small full-VR or landmark-VR audit panel should be retained.

### Red: do not scale without changes

A setting should not be used for scaling if it:

```text
misses torus + hole_fill behavior
reverses epoch-wise trend direction
has weak rank correlation with full VR
has large transition timing errors
has high normalized error despite high speedup
```

Fast but distorted shortcuts should be treated as negative results, not production methods.

## Recommended Initial Tuning Set

The first tuning pass should be small:

```text
event_driven:
  eth0p005_maxskip2_force3
  eth0p01_maxskip5_force5
  eth0p02_maxskip5_force5
  eth0p02_maxskip10_force10

online_landmark_dynamic_support:
  lm300_k16_eth0p01_maxskip5_force5
  lm500_k24_eth0p01_maxskip5_force5
  lm750_k32_eth0p01_maxskip5_force5
  lm500_k24_eth0p02_maxskip10_force10
```

This gives eight additional tuned settings over the same selected checkpoint panel. It should be enough to determine whether the eventual scaling mode should prioritize event-driven recomputation, dynamic landmark support, or a hybrid with more conservative refresh behavior.

## Expected Outcomes

The tuning results should distinguish between several possible failure modes:

```text
If landmark-style modes fail:
    The landmark approximation itself is the bottleneck.

If event-driven works but online dynamic support fails:
    The support approximation is the issue, not temporal recomputation.

If online dynamic support preserves rank but misses transitions:
    The refresh policy needs to be more conservative.

If aggressive event thresholds fail:
    The trigger is skipping important topological changes.

If cheap landmark counts fail but larger counts work:
    Scaling will require more support points or adaptive landmark selection.
```

The final scaling choice should be the cheapest setting that preserves the decision-relevant full-VR structure. The broader methodological point is that scalable PH monitoring should be calibrated before deployment: speedup alone is not evidence that the approximation is scientifically useful.
