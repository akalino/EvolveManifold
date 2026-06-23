#!/usr/bin/env python3
"""
Plot GW summary metrics grouped by collapse mechanism.

This script consumes a run-level GW summary table, such as the output from
``summarize_gw_trajectory.py``. It is intended for mechanism-level comparison
plots after the raw pairwise GW outputs have already been summarized.

Examples
--------
python plot_gw_summary_by_mechanism.py \\
    --summary "$EVOLVE_ROOT/gw_outputs/tables/table_gw_primary_adjacent_L256.parquet" \\
    --output-dir "$EVOLVE_ROOT/gw_outputs/plots/summary_by_mechanism"

python plot_gw_summary_by_mechanism.py \\
    --summary "$EVOLVE_ROOT/gw_outputs/tables/table_gw_primary_from_start_L256.parquet" \\
    --metric gw_final_from_start \\
    --output-dir "$EVOLVE_ROOT/gw_outputs/plots/summary_by_mechanism"
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

    :param path: Path to summary parquet or CSV file.
    :return: Summary dataframe.
    """
    path = Path(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError("Summary file must end in .parquet or .csv")


def available_metrics(df, requested):
    """
    Return metrics that are available and numeric.

    :param df: Summary dataframe.
    :param requested: Requested metric names.
    :return: List of usable metric names.
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


def clean_for_metric(df, metric, group_col):
    """
    Filter rows to those usable for a given metric.

    :param df: Summary dataframe.
    :param metric: Metric column.
    :param group_col: Grouping column.
    :return: Filtered dataframe.
    """
    out = df.copy()

    if group_col not in out.columns:
        raise ValueError(f"Missing group column: {group_col}")

    out = out.dropna(subset=[group_col, metric])
    out = out[np.isfinite(out[metric])]

    return out


def summarize_groups(df, metric, group_col):
    """
    Aggregate a metric by group.

    :param df: Summary dataframe.
    :param metric: Metric column.
    :param group_col: Grouping column.
    :return: Aggregated dataframe.
    """
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            metric_mean=(metric, "mean"),
            metric_median=(metric, "median"),
            metric_std=(metric, "std"),
            metric_count=(metric, "count"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values("metric_median", ascending=False)

    return grouped


def plot_group_bar(summary, metric, group_col, output_dir, use_median):
    """
    Plot a grouped bar chart for one metric.

    :param summary: Aggregated dataframe.
    :param metric: Metric name.
    :param group_col: Grouping column.
    :param output_dir: Output directory.
    :param use_median: If true, plot medians rather than means.
    """
    value_col = "metric_median" if use_median else "metric_mean"
    label = "Median" if use_median else "Mean"

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(summary))
    y = summary[value_col].to_numpy(dtype=float)
    labels = summary[group_col].astype(str).tolist()

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(f"{label} {metric}")
    ax.set_title(f"{label} {metric} by {group_col}")
    ax.grid(axis="y", alpha=0.3)

    for i, row in summary.iterrows():
        ax.text(
            i,
            row[value_col],
            f"n={int(row['metric_count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    out = output_dir / f"{metric}__by_{group_col}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"[PLOT] {out}")


def plot_group_box(df, metric, group_col, output_dir):
    """
    Plot a grouped boxplot for one metric.

    :param df: Filtered summary dataframe.
    :param metric: Metric column.
    :param group_col: Grouping column.
    :param output_dir: Output directory.
    """
    order = (
        df.groupby(group_col)[metric]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    data = [df.loc[df[group_col] == group, metric].to_numpy(dtype=float) for group in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, labels=[str(x) for x in order], showfliers=False)
    ax.set_xticklabels([str(x) for x in order], rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} distribution by {group_col}")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    out = output_dir / f"{metric}__box_by_{group_col}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"[PLOT] {out}")


def main():
    """
    Run the mechanism-level GW summary plotting CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-col", default="mechanism")
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--boxplot", action="store_true")
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
        usable = clean_for_metric(df, metric, args.group_col)

        if len(usable) == 0:
            print(f"[SKIP] no usable rows for {metric}")
            continue

        grouped = summarize_groups(usable, metric, args.group_col)
        plot_group_bar(
            grouped,
            metric,
            args.group_col,
            output_dir,
            use_median=not args.mean,
        )

        if args.boxplot:
            plot_group_box(usable, metric, args.group_col, output_dir)


if __name__ == "__main__":
    main()
