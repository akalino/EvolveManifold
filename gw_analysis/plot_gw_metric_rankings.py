#!/usr/bin/env python3
"""
Plot ranked GW metrics across runs.

This script consumes a run-level GW summary table and creates top-N ranking
plots for selected GW trajectory metrics. It is useful for quickly identifying
which runs had the largest OT path length, largest from-start departure, largest
local spike, or strongest acceleration.

Examples
--------
python plot_gw_metric_rankings.py \\
    --summary "$EVOLVE_ROOT/gw_outputs/tables/table_gw_primary_adjacent_L256.parquet" \\
    --output-dir "$EVOLVE_ROOT/gw_outputs/plots/rankings"

python plot_gw_metric_rankings.py \\
    --summary "$EVOLVE_ROOT/gw_outputs/tables/table_gw_primary_from_start_L256.parquet" \\
    --metric gw_final_from_start \\
    --metric gw_auc_from_start \\
    --top-n 30 \\
    --output-dir "$EVOLVE_ROOT/gw_outputs/plots/rankings"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "gw_path_length_proxy",
    "gw_mean_distance",
    "gw_max_distance",
    "gw_mean_abs_acceleration",
    "gw_front_loadedness",
    "gw_late_instability",
    "gw_auc_from_start",
    "gw_final_from_start",
]


def read_summary(path):
    """
    Read a GW summary table.

    :param path: Path to parquet or CSV summary table.
    :return: Summary dataframe.
    """
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError("Summary file must end in .parquet or .csv")


def safe_label(row):
    """
    Build a compact human-readable run label.

    :param row: Dataframe row.
    :return: Label string.
    """
    parts = []

    for col in ["mechanism", "geometry", "schedule", "severity"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))

    extras = []

    if "noise" in row.index and pd.notna(row["noise"]):
        extras.append(f"noise={row['noise']}")

    if "mover_frac" in row.index and pd.notna(row["mover_frac"]):
        extras.append(f"mp={row['mover_frac']}")

    if "seed" in row.index and pd.notna(row["seed"]):
        extras.append(f"seed={row['seed']}")

    if extras:
        parts.append(",".join(extras))

    if parts:
        return " | ".join(parts)

    if "run_id" in row.index and pd.notna(row["run_id"]):
        return str(row["run_id"])

    if "model" in row.index and pd.notna(row["model"]):
        return str(row["model"])

    return "run"


def available_metrics(df, requested):
    """
    Return requested metrics that exist and are numeric.

    :param df: Summary dataframe.
    :param requested: Requested metric names.
    :return: Usable metric list.
    """
    metrics = []

    for metric in requested:
        if metric not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[metric]):
            continue

        if df[metric].notna().sum() == 0:
            continue

        metrics.append(metric)

    return metrics


def ranked_rows(df, metric, top_n, ascending):
    """
    Select top-N ranked rows for one metric.

    :param df: Summary dataframe.
    :param metric: Metric column.
    :param top_n: Number of rows to keep.
    :param ascending: Sort ascending if true.
    :return: Ranked dataframe.
    """
    usable = df.dropna(subset=[metric]).copy()
    usable = usable[np.isfinite(usable[metric])]

    if len(usable) == 0:
        return usable

    usable["plot_label"] = usable.apply(safe_label, axis=1)

    ranked = usable.sort_values(metric, ascending=ascending).head(top_n)
    ranked = ranked.iloc[::-1].copy()

    return ranked


def plot_ranking(df, metric, output_dir, top_n, ascending):
    """
    Plot a horizontal top-N ranking for one metric.

    :param df: Summary dataframe.
    :param metric: Metric column.
    :param output_dir: Output directory.
    :param top_n: Number of rows to keep.
    :param ascending: Sort ascending if true.
    """
    ranked = ranked_rows(df, metric, top_n, ascending)

    if len(ranked) == 0:
        print(f"[SKIP] no usable rows for {metric}")
        return

    fig_height = max(4, 0.35 * len(ranked))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    y = np.arange(len(ranked))
    x = ranked[metric].to_numpy(dtype=float)
    labels = ranked["plot_label"].tolist()

    ax.barh(y, x)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(metric)

    direction = "Smallest" if ascending else "Largest"
    ax.set_title(f"{direction} {min(top_n, len(ranked))} runs by {metric}")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    suffix = "bottom" if ascending else "top"
    out = output_dir / f"{metric}__{suffix}_{top_n}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"[PLOT] {out}")


def write_rank_table(df, metrics, output_dir, top_n, ascending):
    """
    Write a CSV table containing ranked rows for all metrics.

    :param df: Summary dataframe.
    :param metrics: Metric columns.
    :param output_dir: Output directory.
    :param top_n: Number of rows per metric.
    :param ascending: Sort ascending if true.
    """
    rows = []

    for metric in metrics:
        ranked = ranked_rows(df, metric, top_n, ascending)

        for rank, (_, row) in enumerate(ranked.iloc[::-1].iterrows(), start=1):
            out = {
                "metric": metric,
                "rank": rank,
                "value": row[metric],
                "plot_label": row["plot_label"],
            }

            for col in [
                "run_id",
                "model",
                "mechanism",
                "geometry",
                "schedule",
                "severity",
                "noise",
                "mover_frac",
                "seed",
                "n",
                "d",
                "k",
            ]:
                if col in row.index:
                    out[col] = row[col]

            rows.append(out)

    table = pd.DataFrame(rows)

    suffix = "bottom" if ascending else "top"
    out = output_dir / f"gw_metric_rankings__{suffix}_{top_n}.csv"
    table.to_csv(out, index=False)

    print(f"[TABLE] {out}")


def main():
    """
    Run the GW metric ranking plotting CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_summary(summary_path)

    requested = args.metric if args.metric else DEFAULT_METRICS
    metrics = available_metrics(df, requested)

    if not metrics:
        raise ValueError("No requested numeric metrics were available in the summary table.")

    for metric in metrics:
        plot_ranking(df, metric, output_dir, args.top_n, args.ascending)

    if not args.no_table:
        write_rank_table(df, metrics, output_dir, args.top_n, args.ascending)


if __name__ == "__main__":
    main()
