"""Create a compact noise-effects report for EvolveManifold metrics.

This script summarizes checkpoint-level metric trajectories into scalar
noise-noise_response quantities, then generates a small set of tables and figures.

The intended use case is an extended ``noise_noise_response`` tranche measured with
``online_landmark_dynamic_support``.
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
    "ph_recomputed",
]


def ensure_dir(path):
    """Create a directory if needed.

    :param path: Directory path.
    :return: Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def present_columns(df, columns):
    """Return columns that are present in a dataframe.

    :param df: Input dataframe.
    :param columns: Candidate column names.
    :return: Present column names.
    """
    return [col for col in columns if col in df.columns]


def regime_label(row):
    """Build a compact geometry/mechanism label.

    :param row: Dataframe row.
    :return: Human-readable regime label.
    """
    return f"{row['geometry']} / {row['mechanism']}"


def normalized_auc(group, metric):
    """Compute normalized area under a metric trajectory.

    :param group: Checkpoint-level dataframe for one run.
    :param metric: Metric column name.
    :return: Normalized AUC.
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


def build_run_summary(df, metrics):
    """Collapse checkpoint trajectories to one row per run and metric.

    :param df: Checkpoint-level metric dataframe.
    :param metrics: Metric columns to summarize.
    :return: Long run-summary dataframe.
    """
    run_cols = [
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
    run_cols = present_columns(df, run_cols)

    rows = []
    for key, group in df.groupby(run_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        payload = dict(zip(run_cols, key))
        group = group.sort_values("epoch")

        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                continue

            initial_value = float(values.iloc[0])
            final_value = float(values.iloc[-1])

            rows.append({
                **payload,
                "metric": metric,
                "initial_value": initial_value,
                "final_value": final_value,
                "delta_value": final_value - initial_value,
                "auc": normalized_auc(group, metric),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["regime"] = out.apply(regime_label, axis=1)
    return out


def summarize_over_seeds(run_summary):
    """Aggregate per-run summaries over seeds.

    :param run_summary: Long run-summary dataframe.
    :return: Seed-aggregated dataframe.
    """
    group_cols = [
        "geometry",
        "mechanism",
        "regime",
        "schedule",
        "severity",
        "n",
        "d",
        "mover_frac",
        "noise",
        "ph_mode",
        "metric",
    ]
    group_cols = present_columns(run_summary, group_cols)

    out = (
        run_summary
        .groupby(group_cols, dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            final_mean=("final_value", "mean"),
            final_std=("final_value", "std"),
            delta_mean=("delta_value", "mean"),
            delta_std=("delta_value", "std"),
        )
        .reset_index()
    )

    out["auc_cv"] = out["auc_std"] / out["auc_mean"].abs().replace(0, np.nan)
    out["final_cv"] = out["final_std"] / out["final_mean"].abs().replace(0, np.nan)
    out["delta_cv"] = out["delta_std"] / out["delta_mean"].abs().replace(0, np.nan)
    return out


def add_noise0_comparison(seed_summary):
    """Add baseline-relative columns using noise=0 as the reference.

    :param seed_summary: Seed-aggregated dataframe.
    :return: Dataframe with noise-zero comparison columns.
    """
    match_cols = [
        "geometry",
        "mechanism",
        "regime",
        "schedule",
        "severity",
        "n",
        "d",
        "mover_frac",
        "ph_mode",
        "metric",
    ]
    match_cols = present_columns(seed_summary, match_cols)

    baseline = seed_summary[seed_summary["noise"] == 0.0].copy()
    baseline = baseline[
        match_cols + ["auc_mean", "final_mean", "delta_mean"]
    ].rename(
        columns={
            "auc_mean": "auc_mean_noise0",
            "final_mean": "final_mean_noise0",
            "delta_mean": "delta_mean_noise0",
        }
    )

    out = seed_summary.merge(baseline, on=match_cols, how="left")

    for col in ["auc_mean", "final_mean", "delta_mean"]:
        base = f"{col}_noise0"
        out[f"{col}_minus_noise0"] = out[col] - out[base]
        denom = out[base].abs().replace(0, np.nan)
        out[f"{col}_pct_change_from_noise0"] = 100.0 * out[f"{col}_minus_noise0"] / denom

    return out


def build_metric_ranking(comparison):
    """Rank metrics by aggregate noise noise_response.

    Lower score means more robust.

    :param comparison: Baseline-relative comparison dataframe.
    :return: Metric ranking dataframe.
    """
    nonzero = comparison[comparison["noise"] != 0.0].copy()

    ranking = (
        nonzero
        .groupby("metric", dropna=False)
        .agg(
            median_abs_auc_pct_change=(
                "auc_mean_pct_change_from_noise0",
                lambda x: float(np.nanmedian(np.abs(x))),
            ),
            median_abs_final_pct_change=(
                "final_mean_pct_change_from_noise0",
                lambda x: float(np.nanmedian(np.abs(x))),
            ),
            median_auc_cv=("auc_cv", "median"),
            max_auc_cv=("auc_cv", "max"),
            regimes=("regime", "nunique"),
            noise_levels=("noise", "nunique"),
        )
        .reset_index()
    )

    ranking["noise_response_score"] = (
        ranking["median_abs_auc_pct_change"].fillna(0.0)
        + 100.0 * ranking["median_auc_cv"].fillna(0.0)
    )

    return ranking.sort_values("noise_response_score")


def summarize_ph(df):
    """Summarize PH workflow diagnostics by noise and regime.

    :param df: Checkpoint-level dataframe.
    :return: PH diagnostic dataframe.
    """
    cols = present_columns(df, PH_DIAGNOSTICS)
    if not cols:
        return pd.DataFrame()

    work = df.copy()
    if "regime" not in work.columns:
        work["regime"] = work.apply(regime_label, axis=1)

    group_cols = ["regime", "geometry", "mechanism", "noise", "ph_mode"]
    group_cols = present_columns(work, group_cols)

    agg = {}
    for col in cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")

    return work.groupby(group_cols, dropna=False).agg(**agg).reset_index()


def save_heatmap(pivot, title, out_path, cmap="viridis", colorbar_label="Value"):
    """Save a compact heatmap from a pivot table.

    :param pivot: Pivot table with row and column labels.
    :param title: Figure title.
    :param out_path: Output image path.
    :param cmap: Matplotlib colormap name.
    :param colorbar_label: Label for colorbar.
    :return: None.
    """
    if pivot.empty:
        print(f"[WARN] skipping empty heatmap: {title}")
        return

    pivot = pivot.copy()

    # Drop rows/columns with no finite values.
    finite = np.isfinite(pivot.to_numpy(dtype=float))
    keep_rows = finite.any(axis=1)
    keep_cols = finite.any(axis=0)

    pivot = pivot.loc[pivot.index[keep_rows], pivot.columns[keep_cols]]

    if pivot.empty:
        print(f"[WARN] skipping all-NaN heatmap after filtering: {title}")
        return

    values = pivot.to_numpy(dtype=float)

    if not np.isfinite(values).any():
        print(f"[WARN] skipping heatmap with no finite values: {title}")
        return

    vmax = np.nanmax(np.abs(values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig_width = max(7.0, 0.55 * len(pivot.columns) + 3.0)
    fig_height = max(4.0, 0.35 * len(pivot.index) + 1.5)

    plt.figure(figsize=(fig_width, fig_height))
    masked_values = np.ma.masked_invalid(values)

    plt.imshow(masked_values, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    plt.colorbar(label=colorbar_label)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_noise_sensitivity_heatmap(comparison, out_dir):
    """Plot median absolute AUC change by metric and regime.

    :param comparison: Baseline-relative comparison dataframe.
    :param out_dir: Output plot directory.
    :return: None.
    """
    required = {"noise", "regime", "metric", "auc_mean_pct_change_from_noise0"}
    missing = required.difference(comparison.columns)
    if missing:
        print(f"[WARN] skipping noise sensitivity heatmap; missing columns: {sorted(missing)}")
        return

    nonzero = comparison[comparison["noise"] != 0.0].copy()
    nonzero = nonzero[np.isfinite(nonzero["auc_mean_pct_change_from_noise0"])]

    if nonzero.empty:
        print(
            "[WARN] skipping noise sensitivity heatmap; no finite baseline-relative "
            "AUC changes. Check that each regime has a matching noise=0 baseline."
        )
        return

    pivot = nonzero.pivot_table(
        index="metric",
        columns="regime",
        values="auc_mean_pct_change_from_noise0",
        aggfunc=lambda x: float(np.nanmedian(np.abs(x))),
    )

    save_heatmap(
        pivot,
        "Noise sensitivity: median absolute AUC % change from noise=0",
        out_dir / "fig1_noise_sensitivity_heatmap.png",
        cmap="viridis",
        colorbar_label="Median absolute AUC % change",
    )


def plot_seed_instability_heatmap(comparison, out_dir):
    """Plot seed instability by metric and regime.

    :param comparison: Baseline-relative comparison dataframe.
    :param out_dir: Output plot directory.
    :return: None.
    """
    required = {"noise", "regime", "metric", "auc_cv"}
    missing = required.difference(comparison.columns)
    if missing:
        print(f"[WARN] skipping seed instability heatmap; missing columns: {sorted(missing)}")
        return

    nonzero = comparison[comparison["noise"] != 0.0].copy()
    nonzero = nonzero[np.isfinite(nonzero["auc_cv"])]

    if nonzero.empty:
        print("[WARN] skipping seed instability heatmap; no finite AUC CV values.")
        return

    pivot = nonzero.pivot_table(
        index="metric",
        columns="regime",
        values="auc_cv",
        aggfunc="median",
    )

    save_heatmap(
        pivot,
        "Seed instability: median AUC coefficient of variation",
        out_dir / "fig2_seed_instability_heatmap.png",
        cmap="viridis",
        colorbar_label="Median AUC CV",
    )


def plot_ph_stability(ph_summary, out_dir):
    """Plot compact PH workflow stability summary.

    :param ph_summary: PH diagnostic dataframe.
    :param out_dir: Output plot directory.
    :return: None.
    """
    if ph_summary.empty:
        print("[WARN] skipping PH stability plot; PH summary is empty.")
        return

    y_cols = [
        col for col in [
            "ph_time_sec_mean",
            "ph_event_score_mean",
            "ph_recomputed_mean",
        ]
        if col in ph_summary.columns
    ]

    if not y_cols:
        print("[WARN] skipping PH stability plot; no supported PH diagnostic columns.")
        return

    agg = ph_summary.groupby("noise", dropna=False)[y_cols].mean().reset_index()
    agg = agg.sort_values("noise")

    plotted = False
    plt.figure(figsize=(7.0, 4.2))

    for col in y_cols:
        y = agg[col].to_numpy(dtype=float)

        if not np.isfinite(y).any():
            print(f"[WARN] skipping all-NaN PH diagnostic: {col}")
            continue

        scale = np.nanmax(np.abs(y))
        if np.isfinite(scale) and scale > 0:
            y = y / scale

        plt.plot(agg["noise"], y, marker="o", label=col.replace("_mean", ""))
        plotted = True

    if not plotted:
        plt.close()
        print("[WARN] skipping PH stability plot; all diagnostics were NaN.")
        return

    plt.title("PH workflow stability under noise")
    plt.xlabel("Noise")
    plt.ylabel("Normalized mean diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_ph_workflow_stability.png", dpi=220)
    plt.close()

def write_markdown_summary(ranking, out_path):
    """Write a short markdown summary table.

    :param ranking: Metric ranking dataframe.
    :param out_path: Output markdown path.
    :return: None.
    """
    keep = [
        "metric",
        "median_abs_auc_pct_change",
        "median_auc_cv",
        "max_auc_cv",
        "noise_response_score",
    ]
    keep = present_columns(ranking, keep)

    lines = [
        "# Noise response summary",
        "",
        "Lower noise response scores indicate less baseline-relative AUC drift and lower seed variability.",
        "",
        ranking[keep].head(15).to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines))


def main():
    """Create compact noise-effect tables and figures.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--out-dir", default="summary_assets/noise_effects_compact")
    parser.add_argument("--ph-mode", default="online_landmark_dynamic_support")
    parser.add_argument("--metrics", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    table_dir = ensure_dir(out_dir / "tables")
    plot_dir = ensure_dir(out_dir / "figures")

    df = pd.read_parquet(args.metrics_path)

    if "ph_mode" in df.columns:
        df = df[df["ph_mode"] == args.ph_mode].copy()

    if df.empty:
        raise ValueError("No rows after PH-mode filtering.")

    metrics = args.metrics or DEFAULT_METRICS
    metrics = present_columns(df, metrics)

    if not metrics:
        raise ValueError("No requested metrics are present in the dataframe.")

    run_summary = build_run_summary(df, metrics)
    seed_summary = summarize_over_seeds(run_summary)
    comparison = add_noise0_comparison(seed_summary)
    ranking = build_metric_ranking(comparison)
    ph_summary = summarize_ph(df)

    run_summary.to_csv(table_dir / "table_run_summary_by_seed.csv", index=False)
    seed_summary.to_csv(table_dir / "table_seed_summary_by_noise.csv", index=False)
    comparison.to_csv(table_dir / "table_noise0_comparison.csv", index=False)
    ranking.to_csv(table_dir / "table_metric_noise_response_ranking.csv", index=False)

    run_summary.to_parquet(table_dir / "table_run_summary_by_seed.parquet", index=False)
    seed_summary.to_parquet(table_dir / "table_seed_summary_by_noise.parquet", index=False)
    comparison.to_parquet(table_dir / "table_noise0_comparison.parquet", index=False)
    ranking.to_parquet(table_dir / "table_metric_noise_response_ranking.parquet", index=False)

    if not ph_summary.empty:
        ph_summary.to_csv(table_dir / "table_ph_workflow_stability.csv", index=False)
        ph_summary.to_parquet(table_dir / "table_ph_workflow_stability.parquet", index=False)

    plot_noise_sensitivity_heatmap(comparison, plot_dir)
    plot_seed_instability_heatmap(comparison, plot_dir)
    plot_ph_stability(ph_summary, plot_dir)

    write_markdown_summary(ranking, out_dir / "NOISE_noise_response_SUMMARY.md")

    print(f"[DONE] wrote compact report to {out_dir}")
    print("[DONE] figures:")
    print(f"  {plot_dir / 'fig1_noise_sensitivity_heatmap.png'}")
    print(f"  {plot_dir / 'fig2_seed_instability_heatmap.png'}")
    print(f"  {plot_dir / 'fig3_ph_workflow_stability.png'}")
    print("[DONE] key table:")
    print(f"  {table_dir / 'table_metric_noise_response_ranking.csv'}")


if __name__ == "__main__":
    main()
