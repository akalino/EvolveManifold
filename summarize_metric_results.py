import os
import re
import glob
import argparse
import traceback

import numpy as np
import pandas as pd


FILE_RE = re.compile(
    r"^(?P<ph_mode>.+?)-(?P<mechanism>.+?)__"
    r"(?P<geometry>.+?)_n(?P<n>\d+)_d(?P<d>\d+)"
    r"(?:_k(?P<k>\d+))?"
    r"__(?P<schedule>.+?)"
    r"__(?P<severity>.+?)"
    r"__mp(?P<mover_frac>[\d.]+)"
    r"__noise(?P<noise>[\d.]+)"
    r"__seed(?P<seed>\d+)\.csv$"
)


def parse_metadata_from_filename(path):
    name = os.path.basename(path)
    m = FILE_RE.match(name)
    if m is None:
        return None

    out = m.groupdict()
    if out.get("k") is not None:
        out["k"] = int(out["k"])
    out["n"] = int(out["n"])
    out["d"] = int(out["d"])
    out["mover_frac"] = float(out["mover_frac"])
    out["noise"] = float(out["noise"])
    out["seed"] = int(out["seed"])
    return out


def normalize_metric_direction(df):
    """
    Adds collapse-oriented normalized scores so larger means more collapse.
    """
    df = df.sort_values("epoch").copy()

    if "effective_rank" in df.columns:
        x0 = df["effective_rank"].iloc[0]
        if x0 != 0 and pd.notna(x0):
            df["collapse_effective_rank"] = 1.0 - df["effective_rank"] / x0

    if "mean_pairwise_distance" in df.columns:
        x0 = df["mean_pairwise_distance"].iloc[0]
        if x0 != 0 and pd.notna(x0):
            df["collapse_mean_pairwise_distance"] = (
                1.0 - df["mean_pairwise_distance"] / x0
            )

    if "projection_residual" in df.columns:
        x0 = df["projection_residual"].iloc[0]
        if x0 != 0 and pd.notna(x0):
            df["collapse_projection_residual"] = (
                1.0 - df["projection_residual"] / x0
            )

    if "total_persistence_h1" in df.columns:
        x0 = df["total_persistence_h1"].iloc[0]
        if x0 != 0 and pd.notna(x0):
            df["collapse_total_persistence_h1"] = (
                1.0 - df["total_persistence_h1"] / x0
            )

    if "top_k_variance_fraction" in df.columns:
        x0 = df["top_k_variance_fraction"].iloc[0]
        denom = 1.0 - x0
        if abs(denom) > 1e-12 and pd.notna(x0):
            df["collapse_top_k_variance_fraction"] = (
                (df["top_k_variance_fraction"] - x0) / denom
            )
    if "betti_curve_area_h1" in df.columns:
        x0 = df["betti_curve_area_h1"].iloc[0]
        if pd.notna(x0) and x0 != 0:
            df["collapse_betti_curve_area_h1"] = (
                1.0 - df["betti_curve_area_h1"] / x0
            )
    if "betti_curve_peak_h1" in df.columns:
        x0 = df["betti_curve_peak_h1"].iloc[0]
        if pd.notna(x0) and x0 != 0:
            df["collapse_betti_curve_peak_h1"] = (
                1.0 - df["betti_curve_peak_h1"] / x0
            )
    if "betti_curve_change_h1" in df.columns:
        x0 = df["betti_curve_change_h1"].iloc[0]
        if pd.notna(x0):
            if x0 == 0:
                df["collapse_betti_curve_change_h1"] = df["betti_curve_change_h1"]
            else:
                df["collapse_betti_curve_change_h1"] = df["betti_curve_change_h1"] / x0

    return df


def summarize_one_run(df):
    """
    Creates a one-row summary for one result file.
    """
    df = df.sort_values("epoch").copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = normalize_metric_direction(df)

    row0 = df.iloc[0]
    rowf = df.iloc[-1]

    out = {
        "ph_mode": row0["ph_mode"],
        "n": row0["n"],
        "d": row0["d"],
        "mechanism": row0["mechanism"],
        "geometry": row0["geometry"],
        "schedule": row0["schedule"],
        "severity": row0["severity"],
        "mover_frac": row0["mover_frac"],
        "noise": row0["noise"],
        "seed": row0["seed"],
        "epoch_start": row0["epoch"],
        "epoch_end": rowf["epoch"],
        "n_checkpoints": len(df),
    }

    metric_cols = [
        "effective_rank",
        "top_k_variance_fraction",
        "mean_pairwise_distance",
        "projection_residual",
        "total_persistence_h1",
        "betti_curve_area_h1",
        "betti_curve_peak_h1",
        "betti_curve_change_h1"
    ]

    for col in metric_cols:
        if col not in df.columns:
            continue

        vals = df[col]
        vals_nonan = vals.dropna()
        if len(vals_nonan) == 0:
            continue

        out[f"{col}_start"] = vals.iloc[0]
        out[f"{col}_end"] = vals.iloc[-1]
        out[f"{col}_delta"] = vals.iloc[-1] - vals.iloc[0]
        out[f"{col}_min"] = vals_nonan.min()
        out[f"{col}_max"] = vals_nonan.max()
        out[f"{col}_argmin_epoch"] = df.loc[vals_nonan.idxmin(), "epoch"]
        out[f"{col}_argmax_epoch"] = df.loc[vals_nonan.idxmax(), "epoch"]

    collapse_cols = [
        "collapse_effective_rank",
        "collapse_top_k_variance_fraction",
        "collapse_mean_pairwise_distance",
        "collapse_projection_residual",
        "collapse_total_persistence_h1",
        "collapse_betti_curve_area_h1",
        "collapse_betti_curve_peak_h1",
        "collapse_betti_curve_change_h1"
    ]

    for col in collapse_cols:
        if col not in df.columns:
            continue

        vals = df[col]
        vals_nonan = vals.dropna()
        if len(vals_nonan) == 0:
            continue

        tmp = df.loc[vals.notna(), ["epoch", col]].copy()

        out[f"{col}_end"] = vals.iloc[-1]
        out[f"{col}_max"] = vals_nonan.max()
        out[f"{col}_argmax_epoch"] = df.loc[vals_nonan.idxmax(), "epoch"]
        out[f"{col}_auc"] = np.trapz(tmp[col].to_numpy(),
                                     x=tmp["epoch"].to_numpy())

    return pd.DataFrame([out])


def add_metadata(df, meta):
    for k, v in meta.items():
        df[k] = v
    return df


def load_all_results(input_dir):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    dfs = []

    for path in paths:
        meta = parse_metadata_from_filename(path)
        if meta is None:
            continue

        df = pd.read_csv(path)

        metric_cols = [
            "epoch",
            "effective_rank",
            "top_k_variance_fraction",
            "mean_pairwise_distance",
            "projection_residual",
            "total_persistence_h1",
        ]

        for col in metric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = add_metadata(df, meta)
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No matching CSV files found in {input_dir}")

    return dfs


def aggregate_over_seeds(summary_df):
    group_cols = [
        "ph_mode",
        "n",
        "d",
        "mechanism",
        "geometry",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
    ]

    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in group_cols + ["seed"]]

    agg = (
        summary_df.groupby(group_cols)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg.columns = [
        "__".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    return agg


def make_epochwise_average(long_df):
    group_cols = [
        "ph_mode",
        "n",
        "d",
        "mechanism",
        "geometry",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "epoch",
    ]

    numeric_cols = [
        c for c in long_df.select_dtypes(include=[np.number]).columns
        if c not in group_cols + ["seed"]
    ]

    out = (
        long_df.groupby(group_cols)[numeric_cols]
        .mean()
        .reset_index()
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="metric_outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metric_summaries",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dfs = load_all_results(args.input_dir)
    print(f"loaded {len(dfs)} files")

    long_df = pd.concat(dfs, ignore_index=True)
    long_df = (
        long_df.sort_values(
            ["ph_mode", "n", "d",
             "mechanism", "geometry", "schedule",
             "severity", "mover_frac", "noise", "seed", "epoch"]
        )
        .reset_index(drop=True)
    )
    print(f"long shape: {long_df.shape}")

    long_df.to_csv(
        os.path.join(args.output_dir, "all_results_long.csv"),
        index=False,
    )

    run_summaries = []
    group_cols = [
        "ph_mode",
        "n",
        "d",
        "mechanism",
        "geometry",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
    ]

    for keys, df_run in long_df.groupby(group_cols, sort=False):
        try:
            run_summaries.append(summarize_one_run(df_run))
        except Exception as e:
            print(f"failed summarizing run {keys}: {e}")
            traceback.print_exc()

    if not run_summaries:
        raise ValueError("No run summaries were created")
    print(f"built {len(run_summaries)} run summaries")

    summary_df = pd.concat(run_summaries, ignore_index=True)
    summary_df.to_csv(
        os.path.join(args.output_dir, "run_level_summary.csv"),
        index=False,
    )

    seed_agg_df = aggregate_over_seeds(summary_df)
    seed_agg_df.to_csv(
        os.path.join(args.output_dir, "summary_aggregated_over_seeds.csv"),
        index=False,
    )

    epoch_avg_df = make_epochwise_average(long_df)
    epoch_avg_df.to_csv(
        os.path.join(args.output_dir, "epochwise_mean_over_seeds.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
