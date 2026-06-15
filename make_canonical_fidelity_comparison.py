#!/usr/bin/env python3
"""
Generate Fig. 5: PH runtime-fidelity comparison.

Compares each PH mode against full_vr using matched
run metadata + epoch + PH summary columns.

Outputs:
  figures/fig05_ph_runtime_fidelity.pdf
  figures/fig05_ph_runtime_fidelity.png
  tables/ph_mode_fidelity_summary.csv
  tables/ph_mode_fidelity_summary.md
  tables/ph_mode_fidelity_summary.tex
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PNG_DPI = 600

REFERENCE_MODE = "full_vr"

PH_MODES_DEFAULT = [
    "full_vr",
    "skip_vr",
    "event_driven",
    "landmark_vr",
    "online_landmark_dynamic_support",
    "fixed_knn_vr",
    "fixed_support_vr",
    "online_landmark_event",
]

PH_METRICS = [
    "total_persistence_h1",
    "max_persistence_h1",
    "top5_persistence_h1",
    "betti_curve_area_h1",
    "betti_curve_peak_h1",
    "betti_curve_change_h1",
]

MODE_LABELS = {
    "full_vr": "full VR",
    "skip_vr": "skip VR",
    "event_driven": "event driven",
    "landmark_vr": "landmark VR",
    "online_landmark_dynamic_support": "online landmark dynamic support",
    "fixed_knn_vr": "fixed kNN",
    "fixed_support_vr": "fixed support",
    "online_landmark_event": "landmark event",
}

MODE_COLORS = {
    "full_vr": "#000000",
    "skip_vr": "#666666",
    "event_driven": "#0072B2",
    "landmark_vr": "#E69F00",
    "online_landmark_dynamic_support": "#CC79A7",
    "fixed_knn_vr": "#999999",
    "fixed_support_vr": "#999999",
    "online_landmark_event": "#999999",
}


def default_evolve_root() -> Path:
    return Path(
        os.environ.get(
            "EVOLVE_ROOT",
            os.path.expanduser("~/evolve_local/evolve_collapse"),
        )
    ).expanduser().resolve()


def find_metric_files(metric_root: Path, ph_mode: str) -> List[Path]:
    mode_root = metric_root / ph_mode
    combined = mode_root / "_combined" / "all_metrics.parquet"

    if combined.exists():
        return [combined]

    files = sorted(mode_root.rglob("metrics.parquet"))
    return [
        p for p in files
        if "_combined" not in p.parts and "_status" not in p.parts
    ]


def load_mode(metric_root: Path, ph_mode: str) -> pd.DataFrame | None:
    files = find_metric_files(metric_root, ph_mode)
    if not files:
        print(f"[WARN] no metric files for {ph_mode}")
        return None

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"[WARN] skipping unreadable parquet: {path} ({exc})")

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df["ph_mode"] = ph_mode

    for col in ["n", "d", "seed", "epoch", "ph_time_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["mover_frac", "noise"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def canonical_filter(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "geometry",
        "mechanism",
        "n",
        "d",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] missing canonical metadata columns {missing}; skipping canonical filter")
        return df.copy()

    canonical_pairs = (
        ((df["geometry"] == "spiked_gaussian") & (df["mechanism"] == "linear_to_kplane"))
        | ((df["geometry"] == "isotropic") & (df["mechanism"] == "radial_collapse"))
        | ((df["geometry"] == "clustered_gaussian") & (df["mechanism"] == "cluster_tightening"))
        | ((df["geometry"] == "clustered_gaussian") & (df["mechanism"] == "cluster_merging"))
        | ((df["geometry"] == "torus") & (df["mechanism"] == "hole_fill"))
    )

    out = df[
        canonical_pairs
        & (df["n"] == 1000)
        & (df["d"] == 50)
        & (df["schedule"] == "linear")
        & (df["severity"] == "moderate")
        & np.isclose(df["mover_frac"], 1.0)
        & np.isclose(df["noise"], 0.0)
    ].copy()

    return out


def match_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "geometry",
        "mechanism",
        "n",
        "d",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
        "epoch",
    ]
    return [c for c in preferred if c in df.columns]


def run_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "geometry",
        "mechanism",
        "n",
        "d",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
    ]
    return [c for c in preferred if c in df.columns]


def spearman_safe(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.DataFrame({"a": a, "b": b}).dropna()
    if tmp.shape[0] < 3:
        return np.nan
    if tmp["a"].nunique() < 2 or tmp["b"].nunique() < 2:
        return np.nan
    return float(tmp["a"].corr(tmp["b"], method="spearman"))


def normalized_mae(a_ref: pd.Series, a_mode: pd.Series) -> float:
    tmp = pd.DataFrame({"ref": a_ref, "mode": a_mode}).dropna()
    if tmp.empty:
        return np.nan

    denom = float(tmp["ref"].max() - tmp["ref"].min())
    if denom <= 1e-12:
        denom = float(abs(tmp["ref"].mean())) + 1e-12

    return float(np.mean(np.abs(tmp["mode"] - tmp["ref"])) / denom)


def trace_delta_spearman(g: pd.DataFrame, ref_col: str, mode_col: str) -> float:
    g = g.sort_values("epoch")
    ref_delta = g[ref_col].diff()
    mode_delta = g[mode_col].diff()
    return spearman_safe(ref_delta, mode_delta)


def first_sustained_detection(
    epochs: pd.Series,
    values: pd.Series,
    threshold: float,
    window: int,
    direction: str = "decrease",
) -> float | None:
    trace = pd.DataFrame({"epoch": epochs, "value": values}).dropna().sort_values("epoch")
    if trace.shape[0] < 2:
        return None

    baseline = float(trace["value"].iloc[0])
    denom = abs(baseline) + 1e-12

    if direction == "decrease":
        score = (baseline - trace["value"]) / denom
    else:
        score = (trace["value"] - baseline) / denom

    passed = score.to_numpy() >= threshold
    ep = trace["epoch"].to_numpy()

    for i in range(len(passed)):
        j = i + window
        if j <= len(passed) and bool(np.all(passed[i:j])):
            return float(ep[i])

    return None


def transition_error_epoch(
    merged: pd.DataFrame,
    metric: str,
    key_cols: List[str],
    threshold: float,
    window: int,
) -> float:
    errs = []

    ref_col = f"{metric}_ref"
    mode_col = f"{metric}_mode"

    for _, g in merged.groupby(key_cols, dropna=False):
        ref_t = first_sustained_detection(
            g["epoch"],
            g[ref_col],
            threshold=threshold,
            window=window,
            direction="decrease",
        )
        mode_t = first_sustained_detection(
            g["epoch"],
            g[mode_col],
            threshold=threshold,
            window=window,
            direction="decrease",
        )

        if ref_t is None and mode_t is None:
            errs.append(0.0)
        elif ref_t is None or mode_t is None:
            max_epoch = float(pd.to_numeric(g["epoch"], errors="coerce").max())
            errs.append(max_epoch)
        else:
            errs.append(abs(mode_t - ref_t))

    if not errs:
        return np.nan

    return float(np.nanmedian(errs))


def summarize_mode(
    ref: pd.DataFrame,
    mode_df: pd.DataFrame,
    mode: str,
    threshold: float,
    window: int,
) -> Dict[str, float | str]:
    metrics = [m for m in PH_METRICS if m in ref.columns and m in mode_df.columns]

    if not metrics:
        raise RuntimeError(f"No shared PH metrics for mode {mode}")

    keys = match_columns(ref)
    run_keys = run_columns(ref)

    ref_cols = keys + metrics + ["ph_time_sec", "ph_recomputed"]
    mode_cols = keys + metrics + ["ph_time_sec", "ph_recomputed"]

    ref_small = ref[[c for c in ref_cols if c in ref.columns]].copy()
    mode_small = mode_df[[c for c in mode_cols if c in mode_df.columns]].copy()

    merged = ref_small.merge(
        mode_small,
        on=keys,
        how="inner",
        suffixes=("_ref", "_mode"),
    )

    if merged.empty:
        raise RuntimeError(f"No matched rows between full_vr and {mode}")

    rank_vals = []
    trend_vals = []
    mae_vals = []
    transition_vals = []

    for metric in metrics:
        ref_col = f"{metric}_ref"
        mode_col = f"{metric}_mode"

        rank_vals.append(spearman_safe(merged[ref_col], merged[mode_col]))
        mae_vals.append(normalized_mae(merged[ref_col], merged[mode_col]))

        for _, g in merged.groupby(run_keys, dropna=False):
            trend_vals.append(trace_delta_spearman(g, ref_col, mode_col))

        transition_vals.append(
            transition_error_epoch(
                merged,
                metric=metric,
                key_cols=run_keys,
                threshold=threshold,
                window=window,
            )
        )

    mean_ref_time = float(pd.to_numeric(merged["ph_time_sec_ref"], errors="coerce").mean())
    mean_mode_time = float(pd.to_numeric(merged["ph_time_sec_mode"], errors="coerce").mean())

    speedup = mean_ref_time / mean_mode_time if mean_mode_time > 0 else np.nan

    if "ph_recomputed_mode" in merged.columns:
        recompute_rate = float(pd.to_numeric(merged["ph_recomputed_mode"], errors="coerce").mean())
    else:
        recompute_rate = np.nan

    return {
        "ph_mode": mode,
        "label": MODE_LABELS.get(mode, mode),
        "matched_rows": int(merged.shape[0]),
        "metrics_compared": ",".join(metrics),
        "mean_full_vr_time_sec": mean_ref_time,
        "mean_mode_time_sec": mean_mode_time,
        "speedup": speedup,
        "rank_fidelity": float(np.nanmedian(rank_vals)),
        "trend_fidelity": float(np.nanmedian(trend_vals)),
        "normalized_mae": float(np.nanmedian(mae_vals)),
        "transition_error_epoch": float(np.nanmedian(transition_vals)),
        "recompute_rate": recompute_rate,
    }


def interpretation(row: pd.Series) -> str:
    mode = row["ph_mode"]

    if mode == "full_vr":
        return "Fidelity anchor"

    speedup = row["speedup"]
    rank = row["rank_fidelity"]
    mae = row["normalized_mae"]

    if speedup >= 20 and rank >= 0.70 and mae <= 0.25:
        return "Best scale candidate"
    if rank >= 0.90 and mae <= 0.05 and speedup < 5:
        return "High fidelity, modest speedup"
    if rank >= 0.65 and mae <= 0.30:
        return "Useful approximation"
    if speedup >= 20 and rank < 0.50:
        return "Fast but distorted"
    return "Diagnostic only"


def make_table(summary: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    out = summary.copy()
    out["Interpretation"] = out.apply(interpretation, axis=1)

    table = out[
        [
            "ph_mode",
            "speedup",
            "rank_fidelity",
            "trend_fidelity",
            "normalized_mae",
            "transition_error_epoch",
            "recompute_rate",
            "Interpretation",
        ]
    ].copy()

    table = table.rename(
        columns={
            "ph_mode": "PH mode",
            "speedup": "Speedup",
            "rank_fidelity": "Rank fidelity",
            "trend_fidelity": "Trend fidelity",
            "normalized_mae": "Norm. error",
            "transition_error_epoch": "Transition error",
            "recompute_rate": "Recompute rate",
        }
    )

    for c in ["Speedup", "Rank fidelity", "Trend fidelity", "Norm. error", "Transition error", "Recompute rate"]:
        table[c] = pd.to_numeric(table[c], errors="coerce").round(4)

    csv_path = tables_dir / "ph_mode_fidelity_summary.csv"
    md_path = tables_dir / "ph_mode_fidelity_summary.md"
    tex_path = tables_dir / "ph_mode_fidelity_summary.tex"

    table.to_csv(csv_path, index=False)
    table.to_markdown(md_path, index=False)

    tex_path.write_text(
        table.to_latex(
            index=False,
            escape=True,
            caption="Runtime--fidelity tradeoff for persistent-homology approximation modes.",
            label="tab:ph-runtime-fidelity",
        )
    )

    print(f"[WROTE] {csv_path}")
    print(f"[WROTE] {md_path}")
    print(f"[WROTE] {tex_path}")


def make_figure5(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = summary.copy()
    df = df[df["ph_mode"] != REFERENCE_MODE].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["speedup", "rank_fidelity", "normalized_mae"])

    if df.empty:
        raise RuntimeError("No non-reference modes available for Fig. 5")

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    plotted = []

    for _, row in df.iterrows():
        mode = row["ph_mode"]
        x = float(row["speedup"])
        y = float(row["rank_fidelity"])
        mae = float(row["normalized_mae"])

        size = 80 + 280 * min(mae, 1.0)

        ax.scatter(
            x,
            y,
            s=size,
            color=MODE_COLORS.get(mode, "#666666"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            zorder=3,
        )

        plotted.append(
            {
                "mode": mode,
                "label": MODE_LABELS.get(mode, mode),
                "x": x,
                "y": y,
            }
        )

    # Put labels below nodes. Sort left-to-right and stagger them slightly
    # to avoid collisions among nearby points on the log-scaled x-axis.
    plotted = sorted(plotted, key=lambda r: r["x"])

    label_levels = [-0.045, -0.085, -0.125]
    last_log_x_at_level = {level: -np.inf for level in label_levels}

    for item in plotted:
        x = item["x"]
        y = item["y"]
        log_x = np.log10(max(x, 1e-12))

        chosen_level = label_levels[0]
        for level in label_levels:
            if log_x - last_log_x_at_level[level] > 0.22:
                chosen_level = level
                break

        last_log_x_at_level[chosen_level] = log_x

        label_y = max(-0.03, y + chosen_level)

        # Keep the far-right "landmark event" label inside the plot box.
        if item["mode"] == "online_landmark_event":
            label_y = max(0.08, label_y)

        label_x = x
        ha = "center"

        # Keep the far-right "landmark event" label inside the plot box.
        if item["mode"] == "online_landmark_event":
            label_x = x / 1.03  # move left on log scale
            ha = "right"

        ax.text(
            label_x,
            label_y,
            item["label"],
            ha=ha,
            va="top",
            fontsize=8,
            zorder=4,
        )

        ax.plot(
            [x, label_x],
            [y - 0.015, label_y + 0.01],
            linewidth=0.8,
            alpha=0.75,
            color="#333333",
            zorder=2,
        )

    ax.axhline(0.75, linestyle=":", linewidth=1)
    ax.text(
        0.99,
        0.705,
        "rank fidelity = 0.75",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Speedup relative to full VR")
    ax.set_ylabel("Rank fidelity to full VR")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Persistent-homology runtime--fidelity tradeoff")
    ax.grid(alpha=0.25, which="both")

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=MODE_COLORS.get("event_driven", "#0072B2"),
            markeredgecolor="black",
            label="High-fidelity adaptive",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=MODE_COLORS.get("landmark_vr", "#E69F00"),
            markeredgecolor="black",
            label="Landmark approximation",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=MODE_COLORS.get("fixed_knn_vr", "#999999"),
            markeredgecolor="black",
            label="Shortcut method",
        ),
    ]

    size_handles = [
        plt.scatter([], [], s=80, color="white", edgecolor="black", label="Lower error"),
        plt.scatter([], [], s=220, color="white", edgecolor="black", label="Higher error"),
    ]

    mode_legend = ax.legend(
        handles=color_handles,
        title="Mode class",
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        facecolor="white",
        framealpha=0.85,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.17),
        bbox_transform=ax.transAxes,
        fontsize=8,
        title_fontsize=8,
        ncol=1,
        handletextpad=0.8,
    )

    ax.add_artist(mode_legend)

    ax.legend(
        handles=size_handles,
        title="Normalized error",
        frameon=True,
        fancybox=False,
        edgecolor="#BBBBBB",
        facecolor="white",
        framealpha=0.88,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.05),
        bbox_transform=ax.transAxes,
        fontsize=8,
        title_fontsize=8,
        ncol=2,
        handletextpad=0.9,
        columnspacing=1.2,
        borderpad=0.7
    )

    fig.tight_layout()

    out_pdf = figures_dir / "fig05_ph_runtime_fidelity.pdf"
    out_png = figures_dir / "fig05_ph_runtime_fidelity.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[WROTE] {out_pdf}")
    print(f"[WROTE] {out_png}")

def parse_modes(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric-root",
        type=Path,
        default=default_evolve_root() / "metric_outputs",
    )
    parser.add_argument(
        "--modes",
        default=",".join(PH_MODES_DEFAULT),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("tables"),
    )
    parser.add_argument(
        "--no-filter-canonical",
        action="store_true",
    )
    args = parser.parse_args()

    modes = parse_modes(args.modes)

    print(f"[INFO] metric_root={args.metric_root}")
    print(f"[INFO] modes={modes}")

    loaded = {}
    for mode in modes:
        df = load_mode(args.metric_root, mode)
        if df is None:
            continue

        if not args.no_filter_canonical:
            df = canonical_filter(df)

        if df.empty:
            print(f"[WARN] mode {mode} has 0 rows after filtering")
            continue

        loaded[mode] = df
        print(f"[INFO] {mode}: rows={len(df):,}")

    if REFERENCE_MODE not in loaded:
        raise RuntimeError("full_vr is required as the reference mode.")

    ref = loaded[REFERENCE_MODE]

    rows = []
    for mode, mode_df in loaded.items():
        if mode == REFERENCE_MODE:
            mean_time = float(pd.to_numeric(mode_df["ph_time_sec"], errors="coerce").mean())
            rows.append(
                {
                    "ph_mode": mode,
                    "label": MODE_LABELS.get(mode, mode),
                    "matched_rows": int(mode_df.shape[0]),
                    "metrics_compared": ",".join([m for m in PH_METRICS if m in mode_df.columns]),
                    "mean_full_vr_time_sec": mean_time,
                    "mean_mode_time_sec": mean_time,
                    "speedup": 1.0,
                    "rank_fidelity": 1.0,
                    "trend_fidelity": 1.0,
                    "normalized_mae": 0.0,
                    "transition_error_epoch": 0.0,
                    "recompute_rate": float(pd.to_numeric(mode_df.get("ph_recomputed", np.nan), errors="coerce").mean()),
                }
            )
            continue

        try:
            rows.append(
                summarize_mode(
                    ref,
                    mode_df,
                    mode=mode,
                    threshold=args.threshold,
                    window=args.window,
                )
            )
        except Exception as exc:
            print(f"[WARN] failed to summarize {mode}: {exc}")

    summary = pd.DataFrame(rows)

    order = {mode: i for i, mode in enumerate(PH_MODES_DEFAULT)}
    summary["_order"] = summary["ph_mode"].map(order).fillna(999)
    summary = summary.sort_values("_order").drop(columns=["_order"])

    print("[INFO] summary:")
    print(
        summary[
            [
                "ph_mode",
                "matched_rows",
                "speedup",
                "rank_fidelity",
                "trend_fidelity",
                "normalized_mae",
                "transition_error_epoch",
            ]
        ].to_string(index=False)
    )

    make_table(summary, args.tables_dir)
    make_figure5(summary, args.figures_dir)

    print("[DONE]")


if __name__ == "__main__":
    main()
