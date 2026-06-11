# PH mode runtime/fidelity comparison

## Modes loaded

- `event_driven`
- `fixed_knn_vr`
- `fixed_support_vr`
- `full_vr`
- `landmark_vr`
- `online_landmark_dynamic_support`
- `online_landmark_event`
- `skip_vr`

## Runtime summary

| ph_mode                         |   rows |   num_runs |   total_ph_time_sec |   mean_ph_time_sec |   speedup_vs_full_vr |   ph_recomputed_rate |
|:--------------------------------|-------:|-----------:|--------------------:|-------------------:|---------------------:|---------------------:|
| full_vr                         |   1664 |         13 |           74022.6   |          44.4847   |             1        |             0        |
| landmark_vr                     |   1664 |         13 |            6541.44  |           3.93115  |            11.316    |             0        |
| fixed_support_vr                |   1664 |         13 |             880.75  |           0.529297 |            84.0449   |             0        |
| fixed_knn_vr                    |   1664 |         13 |             432.195 |           0.259733 |           171.271    |             0        |
| skip_vr                         |   1664 |         13 |           77937.5   |          46.8374   |             0.949769 |             0        |
| event_driven                    |   1664 |         13 |           57997.8   |          34.8544   |             1.2763   |             0.774038 |
| online_landmark_event           |   1664 |         13 |             309.684 |           0.186108 |           239.026    |             0.710337 |
| online_landmark_dynamic_support |   1664 |         13 |            1150.19  |           0.69122  |            64.3568   |             0.771635 |

## Fidelity summary

| ph_mode                         |   median_spearman_vs_full_vr |   median_delta_spearman_vs_full_vr |   median_normalized_mae_vs_full_vr |   median_transition_epoch_abs_error |   speedup_vs_full_vr |   scaling_score |
|:--------------------------------|-----------------------------:|-----------------------------------:|-----------------------------------:|------------------------------------:|---------------------:|----------------:|
| skip_vr                         |                     0.99995  |                           0.997574 |                        8.57409e-05 |                                 0   |             0.949769 |       0.804204  |
| online_landmark_dynamic_support |                     0.8034   |                           0.650071 |                        0.134766    |                                 5   |            64.3568   |       0.777293  |
| event_driven                    |                     0.999164 |                           0.883426 |                        0.0017089   |                                 0   |             1.2763   |       0.759915  |
| landmark_vr                     |                     0.751543 |                           0.713024 |                        0.140963    |                                 4   |            11.316    |       0.750767  |
| fixed_knn_vr                    |                     0.257922 |                          -0.644945 |                       24.0733      |                                 4.5 |           171.271    |       0.03484   |
| fixed_support_vr                |                     0.225619 |                          -0.624128 |                       34.4671      |                                 5   |            84.0449   |       0.0290296 |
| online_landmark_event           |                     0.124164 |                          -0.657831 |                        9.98875     |                                 5   |           239.026    |      -0.0328741 |

## Scaling recommendations

| ph_mode                         | decision                     | reason                                                                                                             |   scaling_score |
|:--------------------------------|:-----------------------------|:-------------------------------------------------------------------------------------------------------------------|----------------:|
| skip_vr                         | diagnostic_only_or_tune      | excellent rank fidelity; trend changes preserved; low normalized error; no speedup; transition timing preserved    |       0.804204  |
| online_landmark_dynamic_support | diagnostic_only_or_tune      | usable rank fidelity; trend changes questionable; low normalized error; material speedup; transition timing drift  |       0.777293  |
| event_driven                    | diagnostic_only_or_tune      | excellent rank fidelity; trend changes preserved; low normalized error; small speedup; transition timing preserved |       0.759915  |
| landmark_vr                     | diagnostic_only_or_tune      | usable rank fidelity; trend changes preserved; low normalized error; material speedup; transition timing drift     |       0.750767  |
| fixed_knn_vr                    | do_not_scale_without_changes | weak rank fidelity; trend changes questionable; high normalized error; material speedup; transition timing drift   |       0.03484   |
| fixed_support_vr                | do_not_scale_without_changes | weak rank fidelity; trend changes questionable; high normalized error; material speedup; transition timing drift   |       0.0290296 |
| online_landmark_event           | do_not_scale_without_changes | weak rank fidelity; trend changes questionable; high normalized error; material speedup; transition timing drift   |      -0.0328741 |

## How to read this

Use `full_vr` as the reference, but do not demand exact equality. For scaling, prefer modes that preserve rank/order and epoch-wise trends while producing material speedup. A shortcut with biased absolute persistence values can still be acceptable if its Spearman correlation, trend sign agreement, mechanism ordering, and downstream relationships are stable.
