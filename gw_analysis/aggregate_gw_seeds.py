#!/usr/bin/env python3
"""
Aggregate GW outputs across landmark seeds.
"""

import argparse
from pathlib import Path

import pandas as pd


GROUP_COLS = [
    "run_id",
    "mechanism",
    "geometry",
    "schedule",
    "severity",
    "n",
    "d",
    "k",
    "mover_frac",
    "noise",
    "source_epoch",
    "target_epoch",
    "comparison_type",
    "normalization",
    "n_landmarks",
    "epsilon",
    "ot_method",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gw_root = Path(args.gw_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(gw_root.rglob("*.parquet")):
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"[WARN] failed to read {path}: {exc}")

    if not frames:
        raise RuntimeError("No readable parquet files found.")

    df = pd.concat(frames, ignore_index=True)

    group_cols = [c for c in GROUP_COLS if c in df.columns]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            gw_distance_mean=("gw_distance", "mean"),
            gw_distance_std=("gw_distance", "std"),
            gw_distance_min=("gw_distance", "min"),
            gw_distance_max=("gw_distance", "max"),
            gw_distance_q25=("gw_distance", lambda x: x.quantile(0.25)),
            gw_distance_q75=("gw_distance", lambda x: x.quantile(0.75)),
            n_seeds=("landmark_seed", "nunique"),
        )
        .reset_index()
    )

    agg["gw_distance_cv"] = agg["gw_distance_std"] / agg["gw_distance_mean"].abs()

    if output.suffix == ".parquet":
        agg.to_parquet(output, index=False)
    elif output.suffix == ".csv":
        agg.to_csv(output, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")

    print(f"[DONE] wrote {output}")
    print(f"[DONE] rows={len(agg)}")


if __name__ == "__main__":
    main()
