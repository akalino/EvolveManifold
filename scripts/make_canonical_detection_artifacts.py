#!/usr/bin/env python3
"""
Generate canonical detection artifacts for EvolveManifold.

Outputs:
  figures/fig03_detection_time_dotrange.pdf
  figures/fig03_detection_time_dotrange.png
  figures/fig03_detection_time_boxplots.pdf      # compatibility placeholder
  figures/fig03_detection_time_boxplots.png
  figures/fig04_mechanism_detection_heatmap.pdf
  figures/fig04_mechanism_detection_heatmap.png
  tables/table03_earliest_stable_metric_families.md
  tables/table03_earliest_stable_metric_families.tex
  tables/canonical_detection_times_metric_level.csv
  tables/canonical_detection_times_family_level.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PH_MODE_DEFAULT = "online_landmark_dynamic_support"

REGIME_LABELS = {
    "linear_to_kplane": "Projection",
    "cluster_tightening": "Cluster tightening",
    "cluster_merging": "Cluster merging",
    "radial_collapse": "Radial",
    "hole_fill": "Topology-first",
}

MECHANISM_ORDER = [
    "linear_to_kplane",
    "cluster_tightening",
    "cluster_merging",
    "radial_collapse",
    "hole_fill",
]

REGIME_ORDER = [REGIME_LABELS[m] for m in MECHANISM_ORDER]

FAMILY_ORDER = ["Spectral", "Geometric", "Topological"]

FAMILY_COLORS = {
    "Spectral": "#E69F00",      # orange
    "Geometric": "#CC79A7",     # reddish purple
    "Topological": "#000000",   # black
}

PNG_DPI = 600

METRIC_SPECS = {
    # spectral
    "effective_rank": ("Spectral", "decrease"),
    "top_k_variance_fraction": ("Spectral", "increase"),

    # geometric
    "mean_pairwise_distance": ("Geometric", "decrease"),
    "median_pairwise_distance": ("Geometric", "decrease"),
    "std_pairwise_distance": ("Geometric", "decrease"),
    "q10_pairwise_distance": ("Geometric", "decrease"),
    "q50_pairwise_distance": ("Geometric", "decrease"),
    "q90_pairwise_distance": ("Geometric", "decrease"),

    # topological
    "total_persistence_h1": ("Topological", "decrease"),
    "max_persistence_h1": ("Topological", "decrease"),
    "top5_persistence_h1": ("Topological", "decrease"),
    "betti_curve_area_h1": ("Topological", "decrease"),
    "betti_curve_peak_h1": ("Topological", "decrease"),
    "betti_curve_change_h1": ("Topological", "increase"),
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


def load_metrics(metric_root: Path, ph_mode: str) -> pd.DataFrame:
    files = find_metric_files(metric_root, ph_mode)

    if not files:
        raise FileNotFoundError(f"No metrics.parquet files found under {metric_root / ph_mode}")

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"[WARN] skipping unreadable parquet: {path} ({exc})")

    if not frames:
        raise RuntimeError("Found metric files, but none could be read.")

    df = pd.concat(frames, ignore_index=True)

    for col in ["n", "d", "seed", "epoch"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["mover_frac", "noise"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_canonical(df: pd.DataFrame, seeds: List[int] | None = None) -> pd.DataFrame:
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
        raise ValueError(f"Missing required metadata columns: {missing}")

    canonical_pairs = (
        ((df["geometry"] == "spiked_gaussian") & (df["mechanism"] == "linear_to_kplane"))
        | ((df["geometry"] == "isotropic") & (df["mechanism"] == "radial_collapse"))
        | ((df["geometry"] == "clustered_gaussian") & (df["mechanism"] == "cluster_tightening"))
        | ((df["geometry"] == "clustered_gaussian") & (df["mechanism"] == "cluster_merging"))
        | ((df["geometry"] == "torus") & (df["mechanism"] == "hole_fill"))
    )

    mask = (
        canonical_pairs
        & (df["n"] == 1000)
        & (df["d"] == 50)
        & (df["schedule"] == "linear")
        & (df["severity"] == "moderate")
        & np.isclose(df["mover_frac"], 1.0)
        & np.isclose(df["noise"], 0.0)
    )

    if seeds:
        mask = mask & df["seed"].isin(seeds)

    out = df[mask].copy()

    if out.empty:
        raise RuntimeError("Canonical filter selected 0 rows.")

    return out


def run_key_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "run_id",
        "run_dir",
        "experiment",
        "geometry",
        "mechanism",
        "model",
        "n",
        "d",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
    ]
    return [c for c in preferred if c in df.columns]


def oriented_score(values: pd.Series, direction: str, eps: float = 1e-12) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    baseline = values.iloc[0]
    denom = abs(float(baseline)) + eps

    if direction == "decrease":
        return (baseline - values) / denom
    if direction == "increase":
        return (values - baseline) / denom

    raise ValueError(f"unknown direction: {direction}")


def first_sustained_detection(
    trace: pd.DataFrame,
    metric: str,
    direction: str,
    threshold: float,
    window: int,
) -> Tuple[float | None, bool]:
    trace = trace.sort_values("epoch").copy()
    trace = trace[["epoch", metric]].dropna()

    if trace.empty or trace["epoch"].nunique() < 2:
        return None, False

    scores = oriented_score(trace[metric], direction)
    epochs = trace["epoch"].to_numpy()
    passed = scores.to_numpy() >= threshold

    for i in range(len(passed)):
        j = i + window
        if j <= len(passed) and bool(np.all(passed[i:j])):
            return float(epochs[i]), True

    return None, False


def compute_metric_detection_times(
    df: pd.DataFrame,
    threshold: float,
    window: int,
) -> pd.DataFrame:
    key_cols = run_key_columns(df)
    rows = []

    available_metrics = {
        metric: spec
        for metric, spec in METRIC_SPECS.items()
        if metric in df.columns
    }

    if not available_metrics:
        raise RuntimeError("None of the configured metric columns were found.")

    print("[INFO] available metric columns:")
    for metric, (family, direction) in available_metrics.items():
        print(f"  {metric}: {family}, {direction}")

    for key, g in df.groupby(key_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        meta = dict(zip(key_cols, key))
        g = g.sort_values("epoch")
        max_epoch = float(pd.to_numeric(g["epoch"], errors="coerce").max())

        for metric, (family, direction) in available_metrics.items():
            det_epoch, detected = first_sustained_detection(
                g,
                metric,
                direction,
                threshold,
                window,
            )

            normalized = (
                det_epoch / max_epoch
                if detected and max_epoch > 0 and det_epoch is not None
                else np.nan
            )

            rows.append(
                {
                    **meta,
                    "regime": REGIME_LABELS.get(meta.get("mechanism"), meta.get("mechanism")),
                    "metric_family": family,
                    "metric": metric,
                    "direction": direction,
                    "threshold": threshold,
                    "window": window,
                    "max_epoch": max_epoch,
                    "detection_epoch": det_epoch,
                    "normalized_detection_time": normalized,
                    "detected": detected,
                }
            )

    out = pd.DataFrame(rows)

    out["mechanism"] = pd.Categorical(
        out["mechanism"],
        categories=MECHANISM_ORDER,
        ordered=True,
    )
    out["regime"] = pd.Categorical(
        out["regime"],
        categories=REGIME_ORDER,
        ordered=True,
    )
    out["metric_family"] = pd.Categorical(
        out["metric_family"],
        categories=FAMILY_ORDER,
        ordered=True,
    )

    return out.sort_values(["mechanism", "seed", "metric_family", "metric"])


def compute_family_detection_times(metric_dt: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        c for c in [
            "run_id",
            "run_dir",
            "experiment",
            "geometry",
            "mechanism",
            "model",
            "n",
            "d",
            "schedule",
            "severity",
            "mover_frac",
            "noise",
            "seed",
            "regime",
            "metric_family",
        ]
        if c in metric_dt.columns
    ]

    rows = []

    for key, g in metric_dt.groupby(key_cols, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)

        meta = dict(zip(key_cols, key))

        detected = g[g["detected"].astype(bool)].copy()

        if detected.empty:
            best_metric = None
            det_epoch = np.nan
            normalized = np.nan
            is_detected = False
            max_epoch = float(pd.to_numeric(g["max_epoch"], errors="coerce").max())
        else:
            best = detected.sort_values(["detection_epoch", "metric"]).iloc[0]
            best_metric = best["metric"]
            det_epoch = float(best["detection_epoch"])
            normalized = float(best["normalized_detection_time"])
            is_detected = True
            max_epoch = float(best["max_epoch"])

        rows.append(
            {
                **meta,
                "best_metric": best_metric,
                "detection_epoch": det_epoch,
                "normalized_detection_time": normalized,
                "detected": is_detected,
                "max_epoch": max_epoch,
            }
        )

    out = pd.DataFrame(rows)

    out["mechanism"] = pd.Categorical(
        out["mechanism"],
        categories=MECHANISM_ORDER,
        ordered=True,
    )
    out["regime"] = pd.Categorical(
        out["regime"],
        categories=REGIME_ORDER,
        ordered=True,
    )
    out["metric_family"] = pd.Categorical(
        out["metric_family"],
        categories=FAMILY_ORDER,
        ordered=True,
    )

    return out.sort_values(["mechanism", "seed", "metric_family"])


def summarize_family_stats(family_dt: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (regime, family), g in family_dt.groupby(
        ["regime", "metric_family"],
        observed=True,
    ):
        detected = g[g["detected"].astype(bool)].copy()
        total_n = int(g.shape[0])
        detected_n = int(detected.shape[0])
        detection_rate = detected_n / total_n if total_n else 0.0

        if detected_n:
            x = pd.to_numeric(detected["detection_epoch"], errors="coerce")
            median_epoch = float(np.nanmedian(x))
            mean_epoch = float(np.nanmean(x))
            min_epoch = float(np.nanmin(x))
            max_epoch = float(np.nanmax(x))
            iqr_epoch = float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))
            std_epoch = float(np.nanstd(x))
        else:
            median_epoch = np.nan
            min_epoch = np.nan
            max_epoch = np.nan
            iqr_epoch = np.nan
            std_epoch = np.nan

        rows.append(
            {
                "regime": regime,
                "metric_family": family,
                "median_epoch": median_epoch,
                "mean_epoch": mean_epoch,
                "min_epoch": min_epoch,
                "max_epoch": max_epoch,
                "iqr_epoch": iqr_epoch,
                "std_epoch": std_epoch,
                "detected_n": detected_n,
                "total_n": total_n,
                "detection_rate": detection_rate,
            }
        )

    out = pd.DataFrame(rows)

    out["regime"] = pd.Categorical(out["regime"], categories=REGIME_ORDER, ordered=True)
    out["metric_family"] = pd.Categorical(
        out["metric_family"],
        categories=FAMILY_ORDER,
        ordered=True,
    )

    return out.sort_values(["regime", "metric_family"])


def summarize_table_iii(family_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for regime, g in family_stats.groupby("regime", observed=True):
        detected_g = g[g["detected_n"] > 0].copy()

        if detected_g.empty:
            earliest_family = "None detected"
            stable_family = "None detected"
            earliest_median = np.nan
            stable_iqr = np.nan
            earliest_rate = 0.0
        else:
            earliest = detected_g.sort_values(
                ["median_epoch", "detection_rate"],
                ascending=[True, False],
            ).iloc[0]

            stable_pool = detected_g[detected_g["detection_rate"] >= 0.60].copy()
            if stable_pool.empty:
                stable_pool = detected_g

            stable = stable_pool.sort_values(
                ["iqr_epoch", "std_epoch", "median_epoch"],
                ascending=[True, True, True],
            ).iloc[0]

            earliest_family = str(earliest["metric_family"])
            stable_family = str(stable["metric_family"])
            earliest_median = float(earliest["median_epoch"])
            stable_iqr = float(stable["iqr_epoch"])
            earliest_rate = float(earliest["detection_rate"])

        rows.append(
            {
                "Regime": str(regime),
                "Earliest": earliest_family,
                "Stable": stable_family,
                "Earliest median epoch": (
                    round(earliest_median, 1) if np.isfinite(earliest_median) else "NA"
                ),
                "Stable IQR epoch": (
                    round(stable_iqr, 1) if np.isfinite(stable_iqr) else "NA"
                ),
                "Detection rate": round(earliest_rate, 2),
            }
        )

    out = pd.DataFrame(rows)
    out["Regime"] = pd.Categorical(out["Regime"], categories=REGIME_ORDER, ordered=True)
    return out.sort_values("Regime")


def save_table_iii(table: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    md_path = tables_dir / "table03_earliest_stable_metric_families.md"
    tex_path = tables_dir / "table03_earliest_stable_metric_families.tex"

    table.to_markdown(md_path, index=False)

    latex = table.to_latex(
        index=False,
        escape=True,
        caption="Summary of earliest and most stable metric families by collapse regime.",
        label="tab:earliest-stable-metric-families",
    )
    tex_path.write_text(latex)

    print(f"[WROTE] {md_path}")
    print(f"[WROTE] {tex_path}")


def make_figure3(family_stats: pd.DataFrame, figures_dir: Path) -> None:
    """
    Fig. 3: mean detection epoch by metric family and collapse regime.

    Clean version:
      - mean points only
      - one marker shape per metric family
      - overlaid x indicates at least one no-detection after final checkpoint
      - horizontal dotted lines separate collapse regimes
      - vertical final-checkpoint label on dotted line
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_df = family_stats.copy()

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    y_base = {regime: i for i, regime in enumerate(REGIME_ORDER)}
    offsets = {
        "Spectral": 0.23,
        "Geometric": 0.00,
        "Topological": -0.23,
    }
    markers = {
        "Spectral": "o",
        "Geometric": "s",
        "Topological": "^",
    }

    for family in FAMILY_ORDER:
        sub = plot_df[plot_df["metric_family"].astype(str) == family]

        for regime in REGIME_ORDER:
            row = sub[sub["regime"].astype(str) == regime]
            if row.empty:
                continue

            row = row.iloc[0]
            y = y_base[regime] + offsets[family]
            detected_n = int(row["detected_n"])
            total_n = int(row["total_n"])
            failed_n = total_n - detected_n

            label = family if regime == REGIME_ORDER[0] else None

            if detected_n > 0 and np.isfinite(row["mean_epoch"]):
                x = float(row["mean_epoch"])

                ax.plot(
                    x,
                    y,
                    marker=markers[family],
                    linestyle="none",
                    markersize=7,
                    color=FAMILY_COLORS[family],
                    markerfacecolor=FAMILY_COLORS[family],
                    markeredgecolor=FAMILY_COLORS[family],
                    label=label
                )

                if failed_n > 0:
                    ax.plot(
                        x,
                        y,
                        marker="x",
                        linestyle="none",
                        markersize=9,
                        markeredgewidth=1.6,
                        color="gray"
                    )
            else:
                # No detections at all: put the family marker just past the final checkpoint
                # and overlay x to indicate failure.
                x = 52.0
                ax.plot(
                    x,
                    y,
                    marker=markers[family],
                    linestyle="none",
                    markersize=7,
                    color=FAMILY_COLORS[family],
                    markerfacecolor=FAMILY_COLORS[family],
                    markeredgecolor=FAMILY_COLORS[family],
                    label=label
                )
                ax.plot(
                    x,
                    y,
                    marker="x",
                    linestyle="none",
                    markersize=9,
                    markeredgewidth=1.5,
                    color="gray"
                )

    # Horizontal dotted separators between regimes.
    for boundary in np.arange(0.5, len(REGIME_ORDER) - 0.5, 1.0):
        ax.axhline(boundary, linestyle=":", linewidth=0.8, alpha=0.55)

    ax.set_yticks([y_base[r] for r in REGIME_ORDER])
    ax.set_yticklabels(REGIME_ORDER)

    ax.set_xlabel("Mean detection epoch")
    ax.set_ylabel("Collapse regime")
    ax.set_xlim(-1, 54)
    ax.set_xticks([0, 10, 20, 30, 40, 50])

    ax.axvline(50, linestyle="--", linewidth=1)

    ax.text(
        48.8,
        0.5,
        "final checkpoint",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="center",
        va="center",
        fontsize=8,
    )

    ax.set_title("Mean detection epoch by metric family and collapse regime")
    ax.grid(axis="x", alpha=0.25)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    out_pdf = figures_dir / "fig03_detection_time_dotrange.pdf"
    out_png = figures_dir / "fig03_detection_time_dotrange.png"

    # Paper placeholder compatibility.
    out_pdf_paper = figures_dir / "fig03_detection_time_boxplots.pdf"
    out_png_paper = figures_dir / "fig03_detection_time_boxplots.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(out_pdf_paper, bbox_inches="tight")
    fig.savefig(out_png_paper, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[WROTE] {out_pdf}")
    print(f"[WROTE] {out_png}")
    print(f"[WROTE] {out_pdf_paper}")
    print(f"[WROTE] {out_png_paper}")


def make_figure4(family_stats: pd.DataFrame, figures_dir: Path) -> None:
    """
    Fig. 4: heatmap of median detected epoch.

    No-detection is not imputed.
    Cell labels show median epoch and detected/total.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_df = family_stats.copy()

    heat = (
        plot_df
        .pivot(index="regime", columns="metric_family", values="mean_epoch")
        .reindex(index=REGIME_ORDER, columns=FAMILY_ORDER)
    )

    detected = (
        plot_df
        .pivot(index="regime", columns="metric_family", values="detected_n")
        .reindex(index=REGIME_ORDER, columns=FAMILY_ORDER)
    )

    total = (
        plot_df
        .pivot(index="regime", columns="metric_family", values="total_n")
        .reindex(index=REGIME_ORDER, columns=FAMILY_ORDER)
    )

    values = heat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.9, 4.2))

    masked = np.ma.masked_invalid(values)
    finite_values = values[np.isfinite(values)]

    if finite_values.size:
        vmin = max(0.0, float(np.nanmin(finite_values)))
        vmax = float(np.nanpercentile(finite_values, 90))
        if vmax <= vmin:
            vmax = float(np.nanmax(finite_values))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 50.0

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f2f2f2")

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(FAMILY_ORDER)))
    ax.set_xticklabels(FAMILY_ORDER)
    ax.set_yticks(np.arange(len(REGIME_ORDER)))
    ax.set_yticklabels(REGIME_ORDER)

    ax.set_xlabel("Metric family")
    ax.set_ylabel("Collapse regime")
    ax.set_title("Mean detection epoch by regime")

    for i, regime in enumerate(REGIME_ORDER):
        for j, family in enumerate(FAMILY_ORDER):
            val = values[i, j]
            dn = detected.loc[regime, family]
            tn = total.loc[regime, family]

            dn = 0 if pd.isna(dn) else int(dn)
            tn = 0 if pd.isna(tn) else int(tn)

            if np.isfinite(val):
                label = f"{val:.0f}\n{dn}/{tn}"
                norm_val = (val - vmin) / max(vmax - vmin, 1e-12)
                text_color = "white" if norm_val < 0.45 else "black"
            else:
                label = f"NA\n{dn}/{tn}"
                text_color = "black"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean detection epoch")

    fig.tight_layout()

    out_pdf = figures_dir / "fig04_mechanism_detection_heatmap.pdf"
    out_png = figures_dir / "fig04_mechanism_detection_heatmap.png"

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[WROTE] {out_pdf}")
    print(f"[WROTE] {out_png}")


def parse_seed_list(seed_text: str | None) -> List[int] | None:
    if not seed_text:
        return None
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metric-root",
        type=Path,
        default=default_evolve_root() / "metric_outputs",
    )
    parser.add_argument(
        "--ph-mode",
        default=PH_MODE_DEFAULT,
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
        default=Path("../figures"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("../tables"),
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list. Default uses all available canonical seeds.",
    )
    parser.add_argument(
        "--no-filter-canonical",
        action="store_true",
    )

    args = parser.parse_args()

    seeds = parse_seed_list(args.seeds)

    print(f"[INFO] metric_root={args.metric_root}")
    print(f"[INFO] ph_mode={args.ph_mode}")
    print(f"[INFO] threshold={args.threshold}")
    print(f"[INFO] window={args.window}")
    print(f"[INFO] seeds={seeds if seeds else 'all available'}")

    df = load_metrics(args.metric_root, args.ph_mode)
    print(f"[INFO] loaded rows={len(df):,}")

    if not args.no_filter_canonical:
        df = filter_canonical(df, seeds=seeds)
        print(f"[INFO] canonical rows={len(df):,}")

    print("[INFO] rows by mechanism/geometry/seed:")
    print(
        df.groupby(["mechanism", "geometry", "seed"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .to_string(index=False)
    )

    metric_dt = compute_metric_detection_times(
        df,
        threshold=args.threshold,
        window=args.window,
    )

    family_dt = compute_family_detection_times(metric_dt)
    family_stats = summarize_family_stats(family_dt)
    table_iii = summarize_table_iii(family_stats)

    args.tables_dir.mkdir(parents=True, exist_ok=True)

    metric_csv = args.tables_dir / "canonical_detection_times_metric_level.csv"
    family_csv = args.tables_dir / "canonical_detection_times_family_level.csv"
    family_stats_csv = args.tables_dir / "canonical_detection_family_summary.csv"

    metric_dt.to_csv(metric_csv, index=False)
    family_dt.to_csv(family_csv, index=False)
    family_stats.to_csv(family_stats_csv, index=False)

    print(f"[WROTE] {metric_csv}")
    print(f"[WROTE] {family_csv}")
    print(f"[WROTE] {family_stats_csv}")

    save_table_iii(table_iii, args.tables_dir)
    make_figure3(family_stats, args.figures_dir)
    make_figure4(family_stats, args.figures_dir)

    print("[DONE]")


if __name__ == "__main__":
    main()
