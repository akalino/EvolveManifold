"""Analyze noise robustness over seeds for measured EvolveManifold metrics.

This script assumes metrics have already been computed for a noise-robustness
tranche using the ``online_landmark_dynamic_support`` PH mode.

It reads a combined metrics parquet file, computes seed-aggregated summaries,
and writes tables and plots describing how metric behavior changes as noise
increases.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "effective_rank",
    "top_k_variance_fraction",
    "mean_pairwise_distance",
    "median_pairwise_distance",
    "std_pairwise_distance",
    "projection_residual",
    "total_persistence_h1",
    "max_persistence_h1",
    "top5_persistence_h1",
    "betti_curve_area_h1",
    "betti_curve_peak_h1",
    "betti_curve_change_h1",
]

PH_DIAGNOSTICS = [
    "ph_time_sec",
    "ph_mem",
    "ph_support_edges",
    "ph_event_score",
]


def ensure_dir(path):
    """Create a directory if needed.

    :param path: Directory path.
    :return: Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def available_columns(df, names):
    """Return requested columns that are present in a dataframe.

    :param df: Input dataframe.
    :param names: Candidate column names.
    :return: List of available column names.
    """
    return [name for name in names if name in df.columns]


def trajectory_auc(group, metric):
    """Compute normalized trajectory area under the curve for one run.

    :param group: One metric trajectory for a single seed/run.
    :param metric: Metric column name.
    :return: Normalized area under the metric curve.
    """
    g = group.sort_values("epoch")
    x = g["epoch"].to_numpy(dtype=float)
    y = g[metric].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    width = x.max() - x.min()
    if width <= 0:
        return np.nan

    return float(np.trapz(y, x) / width)


def add_run_summaries(df, metrics):
    """Compute per-run final, initial, delta, and AUC summaries.

    :param df: Long checkpoint-level metric dataframe.
    :param metrics: Metric columns to summarize.
    :return: Long dataframe with one row per run and metric.
    """
    run_keys = [
        "geometry",
        "mechanism",
        "schedule",
        "severity",
        "n",
        "d",
        "mover_frac",
        "noise",
        "seed",
        "ph_mode",
    ]
    run_keys = [key for key in run_keys if key in df.columns]

    rows = []
    for keys, group in df.groupby(run_keys, dropna=False):
        key_payload = dict(zip(run_keys, keys if isinstance(keys, tuple) else (keys,)))
        group = group.sort_values("epoch")

        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                continue

            initial_value = float(values.iloc[0])
            final_value = float(values.iloc[-1])
            delta_value = final_value - initial_value

            rows.append({
                **key_payload,
                "metric": metric,
                "initial_value": initial_value,
                "final_value": final_value,
                "delta_value": delta_value,
                "auc": trajectory_auc(group, metric),
                "num_checkpoints": int(group["epoch"].nunique()),
            })

    return pd.DataFrame(rows)


def summarize_over_seeds(run_summary):
    """Aggregate run summaries over seeds.

    :param run_summary: Per-run summary dataframe.
    :return: Seed-aggregated summary dataframe.
    """
    group_cols = [
        "geometry",
        "mechanism",
        "schedule",
        "severity",
        "n",
        "d",
        "mover_frac",
        "noise",
        "ph_mode",
        "metric",
    ]
    group_cols = [col for col in group_cols if col in run_summary.columns]

    summary = (
        run_summary
        .groupby(group_cols, dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            initial_mean=("initial_value", "mean"),
            initial_std=("initial_value", "std"),
            final_mean=("final_value", "mean"),
            final_std=("final_value", "std"),
            delta_mean=("delta_value", "mean"),
            delta_std=("delta_value", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
        )
        .reset_index()
    )

    summary["final_cv"] = summary["final_std"] / summary["final_mean"].abs().replace(0, np.nan)
    summary["auc_cv"] = summary["auc_std"] / summary["auc_mean"].abs().replace(0, np.nan)

    return summary


def add_noise_baseline_comparison(summary):
    """Compare each noise level to the matching zero-noise baseline.

    :param summary: Seed-aggregated summary dataframe.
    :return: Summary dataframe with baseline comparison columns.
    """
    match_cols = [
        "geometry",
        "mechanism",
        "schedule",
        "severity",
        "n",
        "d",
        "mover_frac",
        "ph_mode",
        "metric",
    ]
    match_cols = [col for col in match_cols if col in summary.columns]

    baseline = summary[summary["noise"] == 0.0].copy()
    baseline = baseline[
        match_cols + ["final_mean", "delta_mean", "auc_mean"]
    ].rename(
        columns={
            "final_mean": "final_mean_noise0",
            "delta_mean": "delta_mean_noise0",
            "auc_mean": "auc_mean_noise0",
        }
    )

    out = summary.merge(baseline, on=match_cols, how="left")

    for col in ["final_mean", "delta_mean", "auc_mean"]:
        base_col = f"{col}_noise0"
        diff_col = f"{col}_minus_noise0"
        pct_col = f"{col}_pct_change_from_noise0"

        out[diff_col] = out[col] - out[base_col]
        denom = out[base_col].abs().replace(0, np.nan)
        out[pct_col] = 100.0 * out[diff_col] / denom

    return out


def summarize_ph_diagnostics(df):
    """Summarize PH runtime and support diagnostics by noise.

    :param df: Checkpoint-level metric dataframe.
    :return: PH diagnostic summary dataframe.
    """
    diagnostics = available_columns(df, PH_DIAGNOSTICS)
    if not diagnostics:
        return pd.DataFrame()

    group_cols = [
        "geometry",
        "mechanism",
        "schedule",
        "severity",
        "n",
        "d",
        "mover_frac",
        "noise",
        "ph_mode",
    ]
    group_cols = [col for col in group_cols if col in df.columns]

    agg_spec = {}
    for col in diagnostics:
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_std"] = (col, "std")

    if "ph_recomputed" in df.columns:
        agg_spec["ph_recomputed_rate"] = ("ph_recomputed", "mean")

    return df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()


def plot_metric_trajectories(df, metric, out_dir):
    """Plot mean metric trajectory by noise for each geometry/mechanism.

    :param df: Checkpoint-level metric dataframe.
    :param metric: Metric column to plot.
    :param out_dir: Output directory.
    :return: None.
    """
    group_cols = ["geometry", "mechanism"]
    for (geometry, mechanism), sub in df.groupby(group_cols, dropna=False):
        if metric not in sub.columns:
            continue

        summary = (
            sub
            .groupby(["noise", "epoch"], dropna=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(["noise", "epoch"])
        )

        if summary.empty:
            continue

        plt.figure(figsize=(7.2, 4.2))

        for noise, g in summary.groupby("noise", dropna=False):
            x = g["epoch"].to_numpy(dtype=float)
            y = g["mean"].to_numpy(dtype=float)
            s = g["std"].fillna(0.0).to_numpy(dtype=float)

            plt.plot(x, y, marker="o", linewidth=1.5, label=f"noise={noise}")
            plt.fill_between(x, y - s, y + s, alpha=0.15)

        plt.title(f"{metric}: {geometry} / {mechanism}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()

        path = out_dir / f"trajectory__{metric}__{geometry}__{mechanism}.png"
        plt.savefig(path, dpi=200)
        plt.close()


def plot_final_vs_noise(summary, metric, out_dir):
    """Plot final metric value against noise.

    :param summary: Seed-aggregated summary dataframe.
    :param metric: Metric column to plot.
    :param out_dir: Output directory.
    :return: None.
    """
    sub = summary[summary["metric"] == metric].copy()
    if sub.empty:
        return

    for (geometry, mechanism), g in sub.groupby(["geometry", "mechanism"], dropna=False):
        g = g.sort_values("noise")

        plt.figure(figsize=(6.4, 4.0))
        plt.errorbar(
            g["noise"],
            g["final_mean"],
            yerr=g["final_std"].fillna(0.0),
            marker="o",
            capsize=3,
        )
        plt.title(f"Final {metric}: {geometry} / {mechanism}")
        plt.xlabel("Noise")
        plt.ylabel(f"Final {metric}")
        plt.tight_layout()

        path = out_dir / f"final_vs_noise__{metric}__{geometry}__{mechanism}.png"
        plt.savefig(path, dpi=200)
        plt.close()


def plot_seed_variability(summary, metric, out_dir):
    """Plot seed-level coefficient of variation against noise.

    :param summary: Seed-aggregated summary dataframe.
    :param metric: Metric column to plot.
    :param out_dir: Output directory.
    :return: None.
    """
    sub = summary[summary["metric"] == metric].copy()
    if sub.empty or "final_cv" not in sub.columns:
        return

    for (geometry, mechanism), g in sub.groupby(["geometry", "mechanism"], dropna=False):
        g = g.sort_values("noise")

        plt.figure(figsize=(6.4, 4.0))
        plt.plot(g["noise"], g["final_cv"], marker="o")
        plt.title(f"Seed variability in final {metric}: {geometry} / {mechanism}")
        plt.xlabel("Noise")
        plt.ylabel("Coefficient of variation")
        plt.tight_layout()

        path = out_dir / f"seed_cv__{metric}__{geometry}__{mechanism}.png"
        plt.savefig(path, dpi=200)
        plt.close()


def main():
    """Run noise-effect analysis.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-path",
        required=True,
        help="Path to combined metrics parquet, usually all_metrics.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        default="noise_effects",
        help="Directory where tables and plots will be written.",
    )
    parser.add_argument(
        "--ph-mode",
        default="online_landmark_dynamic_support",
        help="PH mode to analyze.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric list. Defaults to common geometric/topological metrics.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    table_dir = ensure_dir(out_dir / "tables")
    plot_dir = ensure_dir(out_dir / "plots")

    df = pd.read_parquet(args.metrics_path)

    if "ph_mode" in df.columns:
        df = df[df["ph_mode"] == args.ph_mode].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check --metrics-path and --ph-mode.")

    metrics = args.metrics or DEFAULT_METRICS
    metrics = available_columns(df, metrics)

    if not metrics:
        raise ValueError("No requested metric columns were found.")

    run_summary = add_run_summaries(df, metrics)
    seed_summary = summarize_over_seeds(run_summary)
    noise_comparison = add_noise_baseline_comparison(seed_summary)
    ph_summary = summarize_ph_diagnostics(df)

    run_summary.to_csv(table_dir / "noise_run_summary_by_seed.csv", index=False)
    seed_summary.to_csv(table_dir / "noise_seed_summary.csv", index=False)
    noise_comparison.to_csv(table_dir / "noise_baseline_comparison.csv", index=False)

    run_summary.to_parquet(table_dir / "noise_run_summary_by_seed.parquet", index=False)
    seed_summary.to_parquet(table_dir / "noise_seed_summary.parquet", index=False)
    noise_comparison.to_parquet(table_dir / "noise_baseline_comparison.parquet", index=False)

    if not ph_summary.empty:
        ph_summary.to_csv(table_dir / "noise_ph_diagnostics.csv", index=False)
        ph_summary.to_parquet(table_dir / "noise_ph_diagnostics.parquet", index=False)

    for metric in metrics:
        plot_metric_trajectories(df, metric, plot_dir)
        plot_final_vs_noise(seed_summary, metric, plot_dir)
        plot_seed_variability(seed_summary, metric, plot_dir)

    print(f"[DONE] wrote tables to {table_dir}")
    print(f"[DONE] wrote plots to {plot_dir}")
    print(f"[INFO] metrics analyzed: {metrics}")
    print(f"[INFO] rows analyzed: {len(df)}")
    print(f"[INFO] ph_mode: {args.ph_mode}")


if __name__ == "__main__":
    main()
