import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    path = Path("/media/alex/WD_BLACK/evolve_collapse/metric_outputs/online_landmark_dynamic_support_all_metrics.csv")
    df = pd.read_csv(path)

    print(df.shape)
    print(df.columns.tolist())

    must_have = [
        "experiment",
        "geometry",
        "mechanism",
        "model",
        "epoch",
        "effective_rank",
        "top_k_variance_fraction",
        "mean_pairwise_distance",
        "median_pairwise_distance",
        "q10_pairwise_distance",
        "q90_pairwise_distance",
        "total_persistence_h1",
        "max_persistence_h1",
        "betti_curve_area_h1",
        "betti_curve_peak_h1",
        "betti_curve_change_h1",
        "ph_mode",
        "ph_recomputed",
        "ph_time_sec",
        "ph_mem",
        "ph_event_score",
    ]

    missing = [c for c in must_have if c not in df.columns]
    print("missing:", missing)

    print(df[must_have[:5] + [
        "effective_rank",
        "mean_pairwise_distance",
        "total_persistence_h1",
        "ph_time_sec",
    ]].head())
