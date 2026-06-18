"""Compare PH modes under noise for EvolveManifold metric outputs.

This script compares multiple persistent-homology modes over the same
noise-robustness tranche. It is designed for comparing, for example,

    online_landmark_dynamic_support
    landmark_vr
    full_vr

over the same generated point-cloud trajectories.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_TOPOLOGY_METRICS = [
    "total_persistence_h1",
    "max_persistence_h1",
    "top5_persistence_h1",
    "betti_curve_area_h1",
    "betti_curve_peak_h1",
    "betti_curve_change_h1",
]

DEFAULT_CONTEXT_METRICS = [
    "effective_rank",
    "top_k_variance_fraction",
    "mean_pairwise_distance",
    "std_pairwise_distance",
    "projection_residual",
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
    """Return candidate columns that are present in a dataframe.

    :param df: Input dataframe.
    :param columns: Candidate column names.
    :return: Present column names.
    """
    return [col for col in columns if col in df.columns]


def read_metrics_files(metrics_paths, metrics_root, ph_modes):
    """Read one or more combined metric files.

    :param metrics_paths: Explicit metric parquet paths.
    :param metrics_root: Root containing ``<ph_mode>/_combined/all_metrics.parquet``.
    :param ph_modes: PH modes to load from metrics_root.
    :return: Combined checkpoint-level dataframe.
    """
    frames = []

    if metrics_paths:
        for path in metrics_paths:
            path = Path(path).expanduser()
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_parquet(path)
            if "ph_mode" not in df.columns:
                df["ph_mode"] = path.parent.parent.name
            frames.append(df)

    if metrics_root:
        metrics_root = Path(metrics_root).expanduser()
        if not ph_modes:
            ph_modes = [
                p.name for p in metrics_root.iterdir()
                if p.is_dir() and (p / "_combined" / "all_metrics.parquet").exists()
            ]

        for mode in ph_modes:
            path = metrics_root / mode / "_combined" / "all_metrics.parquet"
            if not path.exists():
                print(f"[WARN] skipping missing metrics file: {path}")
                continue
            df = pd.read_parquet(path)
            df["ph_mode"] = mode
            frames.append(df)

    if not frames:
        raise ValueError("No metric files were loaded.")

    out = pd.concat(frames, ignore_index=True)

    if ph_modes and "ph_mode" in out.columns:
        out = out[out["ph_mode"].isin(ph_modes)].copy()

    if out.empty:
        raise ValueError("No metric rows remain after PH-mode filtering.")

    return out


def add_regime_label(df):
    """Add compact geometry/mechanism regime labels.

    :param df: Input dataframe.
    :return: Dataframe with a ``regime`` column.
    """
    df = df.copy()
    if "regime" not in df.columns:
        df["regime"] = df["geometry"].astype(str) + " / " + df["mechanism"].astype(str)
    return df


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
    """Collapse checkpoint trajectories to one row per run, mode, and metric.

    :param df: Checkpoint-level metric dataframe.
    :param metrics: Metric columns to summarize.
    :return: Long run-summary dataframe.
    """
    run_cols = [
        "geometry",
        "mechanism",
        "regime",
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
                "num_checkpoints": int(group["epoch"].nunique()),
            })

    return pd.DataFrame(rows)


def summarize_by_mode_noise(run_summary):
    """Aggregate run summaries across seeds for each PH mode/noise group.

    :param run_summary: Per-run summary dataframe.
    :return: Seed-aggregated summary dataframe.
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


def add_noise0_comparison(mode_summary):
    """Add within-mode baseline-relative columns using noise=0.

    :param mode_summary: Seed-aggregated mode/noise summary.
    :return: Summary with baseline-relative columns.
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
    match_cols = present_columns(mode_summary, match_cols)

    baseline = mode_summary[mode_summary["noise"] == 0.0].copy()
    baseline = baseline[
        match_cols + ["auc_mean", "final_mean", "delta_mean"]
    ].rename(
        columns={
            "auc_mean": "auc_mean_noise0",
            "final_mean": "final_mean_noise0",
            "delta_mean": "delta_mean_noise0",
        }
    )

    out = mode_summary.merge(baseline, on=match_cols, how="left")

    for col in ["auc_mean", "final_mean", "delta_mean"]:
        base = f"{col}_noise0"
        diff = f"{col}_minus_noise0"
        pct = f"{col}_pct_change_from_noise0"
        out[diff] = out[col] - out[base]
        denom = out[base].abs().replace(0, np.nan)
        out[pct] = 100.0 * out[diff] / denom

    return out


def build_reference_comparison(run_summary, reference_mode):
    """Compare each PH mode to a reference mode on matched runs.

    :param run_summary: Per-run summary dataframe.
    :param reference_mode: Higher-fidelity PH mode to use as reference.
    :return: Long dataframe with per-run mode-vs-reference differences.
    """
    if reference_mode not in set(run_summary["ph_mode"].dropna()):
        print(f"[WARN] reference mode not present: {reference_mode}")
        return pd.DataFrame()

    match_cols = [
        "geometry",
        "mechanism",
        "regime",
        "schedule",
        "severity",
        "n",
        "d",
        "mover_frac",
        "noise",
        "seed",
        "metric",
    ]
    match_cols = present_columns(run_summary, match_cols)

    ref = run_summary[run_summary["ph_mode"] == reference_mode].copy()
    ref = ref[
        match_cols + ["auc", "final_value", "delta_value"]
    ].rename(
        columns={
            "auc": "auc_reference",
            "final_value": "final_reference",
            "delta_value": "delta_reference",
        }
    )

    comp = run_summary.merge(ref, on=match_cols, how="left")
    comp = comp[comp["ph_mode"] != reference_mode].copy()

    for col, ref_col in [
        ("auc", "auc_reference"),
        ("final_value", "final_reference"),
        ("delta_value", "delta_reference"),
    ]:
        comp[f"{col}_minus_reference"] = comp[col] - comp[ref_col]
        denom = comp[ref_col].abs().replace(0, np.nan)
        comp[f"{col}_pct_error_vs_reference"] = 100.0 * comp[f"{col}_minus_reference"] / denom

    return comp


def summarize_reference_comparison(ref_comparison):
    """Aggregate reference-mode comparison over seeds.

    :param ref_comparison: Per-run mode-vs-reference comparison.
    :return: Aggregated reference-comparison dataframe.
    """
    if ref_comparison.empty:
        return pd.DataFrame()

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
    group_cols = present_columns(ref_comparison, group_cols)

    out = (
        ref_comparison
        .groupby(group_cols, dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            median_abs_auc_pct_error=(
                "auc_pct_error_vs_reference",
                lambda x: float(np.nanmedian(np.abs(x))),
            ),
            mean_abs_auc_pct_error=(
                "auc_pct_error_vs_reference",
                lambda x: float(np.nanmean(np.abs(x))),
            ),
            median_abs_final_pct_error=(
                "final_value_pct_error_vs_reference",
                lambda x: float(np.nanmedian(np.abs(x))),
            ),
            mean_abs_final_pct_error=(
                "final_value_pct_error_vs_reference",
                lambda x: float(np.nanmean(np.abs(x))),
            ),
            auc_bias=("auc_minus_reference", "mean"),
            final_bias=("final_value_minus_reference", "mean"),
        )
        .reset_index()
    )

    return out


def compute_rank_correlations(run_summary, reference_mode):
    """Compute within-group Spearman correlations against a reference mode.

    :param run_summary: Per-run summary dataframe.
    :param reference_mode: Reference PH mode.
    :return: Correlation summary dataframe.
    """
    ref_comp = build_reference_comparison(run_summary, reference_mode)
    if ref_comp.empty:
        return pd.DataFrame()

    group_cols = [
        "geometry",
        "mechanism",
        "regime",
        "noise",
        "ph_mode",
        "metric",
    ]
    group_cols = present_columns(ref_comp, group_cols)

    rows = []
    for key, group in ref_comp.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        payload = dict(zip(group_cols, key))

        g = group[["auc", "auc_reference", "final_value", "final_reference"]].dropna()

        if len(g) < 3:
            auc_spearman = np.nan
            final_spearman = np.nan
        else:
            auc_spearman = g["auc"].rank().corr(g["auc_reference"].rank())
            final_spearman = g["final_value"].rank().corr(g["final_reference"].rank())

        rows.append({
            **payload,
            "matched_seeds": int(len(g)),
            "auc_spearman_vs_reference": auc_spearman,
            "final_spearman_vs_reference": final_spearman,
        })

    return pd.DataFrame(rows)


def summarize_ph_diagnostics(df):
    """Summarize PH diagnostics by regime, noise, and PH mode.

    :param df: Checkpoint-level dataframe.
    :return: PH diagnostics summary dataframe.
    """
    cols = present_columns(df, PH_DIAGNOSTICS)
    if not cols:
        return pd.DataFrame()

    group_cols = ["regime", "geometry", "mechanism", "noise", "ph_mode"]
    group_cols = present_columns(df, group_cols)

    agg = {}
    for col in cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")

    return df.groupby(group_cols, dropna=False).agg(**agg).reset_index()


def build_mode_ranking(noise_comparison, reference_summary, rank_correlations):
    """Build a compact ranking table for PH modes and metrics.

    Lower ``mode_noise_fidelity_score`` is better.

    :param noise_comparison: Within-mode noise-zero comparison dataframe.
    :param reference_summary: Mode-vs-reference error summary.
    :param rank_correlations: Spearman agreement summary.
    :return: Ranking dataframe.
    """
    nonzero = noise_comparison[noise_comparison["noise"] != 0.0].copy()

    base = (
        nonzero
        .groupby(["ph_mode", "metric"], dropna=False)
        .agg(
            median_abs_auc_pct_change=(
                "auc_mean_pct_change_from_noise0",
                lambda x: float(np.nanmedian(np.abs(x))),
            ),
            median_auc_cv=("auc_cv", "median"),
            max_auc_cv=("auc_cv", "max"),
            regimes=("regime", "nunique"),
            noise_levels=("noise", "nunique"),
        )
        .reset_index()
    )

    if reference_summary is not None and not reference_summary.empty:
        ref = (
            reference_summary
            .groupby(["ph_mode", "metric"], dropna=False)
            .agg(
                median_abs_auc_pct_error_vs_reference=(
                    "median_abs_auc_pct_error",
                    "median",
                ),
                max_abs_auc_pct_error_vs_reference=(
                    "median_abs_auc_pct_error",
                    "max",
                ),
            )
            .reset_index()
        )
        base = base.merge(ref, on=["ph_mode", "metric"], how="left")
    else:
        base["median_abs_auc_pct_error_vs_reference"] = np.nan
        base["max_abs_auc_pct_error_vs_reference"] = np.nan

    if rank_correlations is not None and not rank_correlations.empty:
        corr = (
            rank_correlations
            .groupby(["ph_mode", "metric"], dropna=False)
            .agg(
                median_auc_spearman_vs_reference=(
                    "auc_spearman_vs_reference",
                    "median",
                ),
            )
            .reset_index()
        )
        base = base.merge(corr, on=["ph_mode", "metric"], how="left")
    else:
        base["median_auc_spearman_vs_reference"] = np.nan

    base["mode_noise_fidelity_score"] = (
        base["median_abs_auc_pct_change"].fillna(0.0)
        + 100.0 * base["median_auc_cv"].fillna(0.0)
        + base["median_abs_auc_pct_error_vs_reference"].fillna(0.0)
        + 100.0 * (1.0 - base["median_auc_spearman_vs_reference"]).clip(lower=0).fillna(0.0)
    )

    return base.sort_values(["mode_noise_fidelity_score", "ph_mode", "metric"])


def drop_empty_pivot(pivot):
    """Drop all-NaN rows and columns from a pivot table.

    :param pivot: Pivot table.
    :return: Filtered pivot table.
    """
    if pivot.empty:
        return pivot

    values = pivot.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return pivot.iloc[0:0, 0:0]

    keep_rows = finite.any(axis=1)
    keep_cols = finite.any(axis=0)
    return pivot.loc[pivot.index[keep_rows], pivot.columns[keep_cols]]


def save_heatmap(pivot, title, out_path, colorbar_label="Value", cmap="viridis"):
    """Save a NaN-tolerant heatmap.

    :param pivot: Pivot table.
    :param title: Figure title.
    :param out_path: Output image path.
    :param colorbar_label: Colorbar label.
    :param cmap: Matplotlib colormap.
    :return: None.
    """
    pivot = drop_empty_pivot(pivot)

    if pivot.empty:
        print(f"[WARN] skipping empty heatmap: {title}")
        return

    values = pivot.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        print(f"[WARN] skipping all-NaN heatmap: {title}")
        return

    vmax = np.nanmax(np.abs(values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig_width = max(7.0, 0.55 * len(pivot.columns) + 3.0)
    fig_height = max(4.0, 0.36 * len(pivot.index) + 1.5)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    plt.colorbar(label=colorbar_label)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_mode_noise_response(noise_comparison, out_dir):
    """Plot compact heatmap of noise response by metric and PH mode.

    :param noise_comparison: Within-mode baseline-relative dataframe.
    :param out_dir: Figure directory.
    :return: None.
    """
    required = {"noise", "metric", "ph_mode", "auc_mean_pct_change_from_noise0"}
    if not required.issubset(noise_comparison.columns):
        print("[WARN] skipping mode noise response heatmap; missing columns")
        return

    nonzero = noise_comparison[noise_comparison["noise"] != 0.0].copy()
    nonzero = nonzero[np.isfinite(nonzero["auc_mean_pct_change_from_noise0"])]

    if nonzero.empty:
        print("[WARN] skipping mode noise response heatmap; no finite values")
        return

    pivot = nonzero.pivot_table(
        index="metric",
        columns="ph_mode",
        values="auc_mean_pct_change_from_noise0",
        aggfunc=lambda x: float(np.nanmedian(np.abs(x))),
    )

    save_heatmap(
        pivot,
        "Noise response by PH mode",
        out_dir / "fig1_noise_response_by_ph_mode.png",
        colorbar_label="Median absolute AUC % change from noise=0",
    )


def plot_reference_error(reference_summary, out_dir):
    """Plot compact heatmap of mode error against reference.

    :param reference_summary: Aggregated reference-comparison dataframe.
    :param out_dir: Figure directory.
    :return: None.
    """
    if reference_summary.empty:
        print("[WARN] skipping reference error heatmap; reference summary is empty")
        return

    required = {"metric", "ph_mode", "median_abs_auc_pct_error"}
    if not required.issubset(reference_summary.columns):
        print("[WARN] skipping reference error heatmap; missing columns")
        return

    work = reference_summary[np.isfinite(reference_summary["median_abs_auc_pct_error"])].copy()
    if work.empty:
        print("[WARN] skipping reference error heatmap; no finite values")
        return

    pivot = work.pivot_table(
        index="metric",
        columns="ph_mode",
        values="median_abs_auc_pct_error",
        aggfunc="median",
    )

    save_heatmap(
        pivot,
        "Fidelity error against reference PH mode",
        out_dir / "fig2_reference_error_by_ph_mode.png",
        colorbar_label="Median absolute AUC % error vs reference",
    )


def plot_rank_agreement(rank_correlations, out_dir):
    """Plot compact heatmap of rank agreement against reference.

    :param rank_correlations: Spearman correlation dataframe.
    :param out_dir: Figure directory.
    :return: None.
    """
    if rank_correlations.empty:
        print("[WARN] skipping rank agreement heatmap; rank correlations are empty")
        return

    required = {"metric", "ph_mode", "auc_spearman_vs_reference"}
    if not required.issubset(rank_correlations.columns):
        print("[WARN] skipping rank agreement heatmap; missing columns")
        return

    work = rank_correlations[np.isfinite(rank_correlations["auc_spearman_vs_reference"])].copy()
    if work.empty:
        print("[WARN] skipping rank agreement heatmap; no finite values")
        return

    pivot = work.pivot_table(
        index="metric",
        columns="ph_mode",
        values="auc_spearman_vs_reference",
        aggfunc="median",
    )
    pivot = drop_empty_pivot(pivot)
    if pivot.empty:
        return

    values = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(max(7.0, 0.55 * len(pivot.columns) + 3.0),
                        max(4.0, 0.36 * len(pivot.index) + 1.5)))
    plt.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Median Spearman correlation")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Rank agreement with reference PH mode")
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_rank_agreement_vs_reference.png", dpi=220)
    plt.close()


def plot_ph_runtime(ph_summary, out_dir):
    """Plot PH runtime by mode and noise.

    :param ph_summary: PH diagnostics summary.
    :param out_dir: Figure directory.
    :return: None.
    """
    if ph_summary.empty or "noise" not in ph_summary.columns or "ph_mode" not in ph_summary.columns:
        print("[WARN] skipping PH runtime plot; missing diagnostics")
        return

    if "ph_time_sec_mean" not in ph_summary.columns:
        print("[WARN] skipping PH runtime plot; ph_time_sec_mean unavailable")
        return

    agg = ph_summary.groupby(["ph_mode", "noise"], dropna=False)["ph_time_sec_mean"].mean().reset_index()
    agg = agg.sort_values(["ph_mode", "noise"])

    plt.figure(figsize=(7.4, 4.4))
    plotted = False

    for mode, g in agg.groupby("ph_mode", dropna=False):
        y = g["ph_time_sec_mean"].to_numpy(dtype=float)
        if not np.isfinite(y).any():
            continue
        plt.plot(g["noise"], y, marker="o", label=str(mode))
        plotted = True

    if not plotted:
        plt.close()
        print("[WARN] skipping PH runtime plot; no finite runtime values")
        return

    plt.title("PH runtime by noise and mode")
    plt.xlabel("Noise")
    plt.ylabel("Mean PH time, seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_ph_runtime_by_mode.png", dpi=220)
    plt.close()


def write_markdown_summary(mode_ranking, reference_mode, out_path):
    """Write a compact markdown summary for Obsidian or reports.

    :param mode_ranking: PH mode/metric ranking dataframe.
    :param reference_mode: Reference PH mode.
    :param out_path: Output markdown path.
    :return: None.
    """
    keep = [
        "ph_mode",
        "metric",
        "median_abs_auc_pct_change",
        "median_auc_cv",
        "median_abs_auc_pct_error_vs_reference",
        "median_auc_spearman_vs_reference",
        "mode_noise_fidelity_score",
    ]
    keep = present_columns(mode_ranking, keep)

    lines = [
        "# PH Mode Noise/Fidelity Comparison",
        "",
        f"Reference mode: `{reference_mode}`",
        "",
        "Lower `mode_noise_fidelity_score` indicates less noise response, lower seed variability, smaller error against the reference mode, and better rank agreement.",
        "",
        "This score is a triage diagnostic, not a final scientific definition of utility.",
        "",
    ]

    if keep:
        lines.extend([
            "## Top rows",
            "",
            mode_ranking[keep].head(20).to_markdown(index=False),
            "",
        ])

    lines.extend([
        "## Interpretation notes",
        "",
        "- A high noise-response score does not automatically mean a topology metric is useless.",
        "- A high reference error with low rank agreement suggests an approximation/fidelity problem.",
        "- A high baseline shift with high rank agreement suggests magnitude drift but preserved ordering.",
        "- `full_vr`, if available, should be treated as the strongest reference for small audit panels.",
        "- `landmark_vr` is a useful intermediate reference when `full_vr` is too expensive.",
        "",
    ])

    out_path.write_text("\n".join(lines))


def main():
    """Run the multi-PH-mode noise/fidelity comparison.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-root",
        default=None,
        help="Root containing <ph_mode>/_combined/all_metrics.parquet.",
    )
    parser.add_argument(
        "--metrics-paths",
        nargs="*",
        default=None,
        help="Explicit all_metrics.parquet paths to compare.",
    )
    parser.add_argument(
        "--ph-modes",
        nargs="*",
        default=None,
        help="PH modes to compare. If omitted with --metrics-root, all available modes are used.",
    )
    parser.add_argument(
        "--reference-mode",
        default="landmark_vr",
        help="Reference PH mode. Use full_vr later when available.",
    )
    parser.add_argument(
        "--out-dir",
        default="summary_assets/ph_mode_noise_comparison",
        help="Output directory for compact tables and figures.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric columns to compare. Defaults to topology metrics plus context metrics.",
    )
    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="Only compare topology metrics.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    table_dir = ensure_dir(out_dir / "tables")
    figure_dir = ensure_dir(out_dir / "figures")

    df = read_metrics_files(
        metrics_paths=args.metrics_paths,
        metrics_root=args.metrics_root,
        ph_modes=args.ph_modes,
    )
    df = add_regime_label(df)

    if args.metrics:
        metrics = args.metrics
    elif args.topology_only:
        metrics = DEFAULT_TOPOLOGY_METRICS
    else:
        metrics = DEFAULT_TOPOLOGY_METRICS + DEFAULT_CONTEXT_METRICS

    metrics = present_columns(df, metrics)
    if not metrics:
        raise ValueError("None of the requested metrics are present.")

    print(f"[INFO] loaded rows: {len(df)}")
    print(f"[INFO] ph modes: {sorted(df['ph_mode'].dropna().unique())}")
    print(f"[INFO] metrics: {metrics}")

    run_summary = build_run_summary(df, metrics)
    mode_summary = summarize_by_mode_noise(run_summary)
    noise_comparison = add_noise0_comparison(mode_summary)

    reference_comparison = build_reference_comparison(run_summary, args.reference_mode)
    reference_summary = summarize_reference_comparison(reference_comparison)
    rank_correlations = compute_rank_correlations(run_summary, args.reference_mode)
    ph_summary = summarize_ph_diagnostics(df)
    mode_ranking = build_mode_ranking(noise_comparison, reference_summary, rank_correlations)

    run_summary.to_csv(table_dir / "table_run_summary_by_mode_seed.csv", index=False)
    mode_summary.to_csv(table_dir / "table_mode_noise_seed_summary.csv", index=False)
    noise_comparison.to_csv(table_dir / "table_mode_noise0_comparison.csv", index=False)
    mode_ranking.to_csv(table_dir / "table_ph_mode_metric_ranking.csv", index=False)

    run_summary.to_parquet(table_dir / "table_run_summary_by_mode_seed.parquet", index=False)
    mode_summary.to_parquet(table_dir / "table_mode_noise_seed_summary.parquet", index=False)
    noise_comparison.to_parquet(table_dir / "table_mode_noise0_comparison.parquet", index=False)
    mode_ranking.to_parquet(table_dir / "table_ph_mode_metric_ranking.parquet", index=False)

    if not reference_comparison.empty:
        reference_comparison.to_csv(table_dir / "table_reference_comparison_by_seed.csv", index=False)
        reference_summary.to_csv(table_dir / "table_reference_comparison_summary.csv", index=False)
        reference_comparison.to_parquet(table_dir / "table_reference_comparison_by_seed.parquet", index=False)
        reference_summary.to_parquet(table_dir / "table_reference_comparison_summary.parquet", index=False)

    if not rank_correlations.empty:
        rank_correlations.to_csv(table_dir / "table_rank_agreement_vs_reference.csv", index=False)
        rank_correlations.to_parquet(table_dir / "table_rank_agreement_vs_reference.parquet", index=False)

    if not ph_summary.empty:
        ph_summary.to_csv(table_dir / "table_ph_diagnostics_by_mode.csv", index=False)
        ph_summary.to_parquet(table_dir / "table_ph_diagnostics_by_mode.parquet", index=False)

    plot_mode_noise_response(noise_comparison, figure_dir)
    plot_reference_error(reference_summary, figure_dir)
    plot_rank_agreement(rank_correlations, figure_dir)
    plot_ph_runtime(ph_summary, figure_dir)

    write_markdown_summary(
        mode_ranking,
        args.reference_mode,
        out_dir / "PH_MODE_NOISE_FIDELITY_SUMMARY.md",
    )

    print(f"[DONE] wrote report to {out_dir}")
    print(f"[DONE] ranking table: {table_dir / 'table_ph_mode_metric_ranking.csv'}")
    print(f"[DONE] markdown summary: {out_dir / 'PH_MODE_NOISE_FIDELITY_SUMMARY.md'}")


if __name__ == "__main__":
    main()
