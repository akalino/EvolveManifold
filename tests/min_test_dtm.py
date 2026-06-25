import numpy as np
from ph_workflow_alt import PHWorkflow


rng = np.random.default_rng(17)

x0 = rng.normal(size=(60, 6))
xs = []
for t in range(3):
    x = x0.copy()
    x[:, 3:] *= 1.0 - 0.2 * t
    xs.append(x)


modes = [
    "full_dtm",
    "landmark_dtm",
    "fixed_knn_dtm",
    "online_landmark_dtm_event",
    "online_landmark_dtm_dynamic_support",
]

for mode in modes:
    print("\nMODE:", mode)

    wf = PHWorkflow(
        _mode=mode,
        _max_dim=1,
        _n_landmarks=20,
        _knn_k=6,
        _dtm_k=6,
        _event_thresh=0.01,
        _event_max_skip=2,
        _force_every=2,
    )

    for epoch, x in enumerate(xs):
        dgms = wf.diagrams(x, epoch)
        print(
            "epoch", epoch,
            "recomputed", wf.last_recomputed,
            "event_score", wf.last_event_score,
            "diagram_shapes", [d.shape for d in dgms],
        )

print("\nBaby DTM test finished.")
