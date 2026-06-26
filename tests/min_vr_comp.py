import numpy as np
from ph_workflow_alt import PHWorkflow


rng = np.random.default_rng(17)
x = rng.normal(size=(50, 5))

modes = [
    "full_vr",
    "landmark_vr",
    "fixed_knn_vr",
    "online_landmark_event",
    "online_landmark_dynamic_support",
]

for mode in modes:
    print("\nMODE:", mode)

    wf = PHWorkflow(
        _mode=mode,
        _max_dim=1,
        _n_landmarks=20,
        _knn_k=6,
        _event_thresh=0.01,
        _event_max_skip=2,
        _force_every=2,
    )

    dgms = wf.diagrams(x, 0)
    print("diagram_shapes", [d.shape for d in dgms])

print("\nBaby VR compatibility test finished.")
