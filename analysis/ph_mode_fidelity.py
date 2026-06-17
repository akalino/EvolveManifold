#!/usr/bin/env python3
"""
Compare PH shortcut modes against full_vr on a small fidelity benchmark.

Inputs expected from measure_checkpoints_parallel_parquet.py:

  metric_outputs/<ph_mode>/_combined/all_metrics.parquet
  metric_outputs/<ph_mode>/_combined/all_metrics.csv
  metric_outputs/<ph_mode>/_status/measurement_status.csv

Outputs:

  metric_outputs/_comparison/runtime_by_mode.csv
  metric_outputs/_comparison/fidelity_by_mode_metric.csv
  metric_outputs/_comparison/fidelity_summary_by_mode.csv
  metric_outputs/_comparison/metric_relationships_by_mode.csv
  metric_outputs/_comparison/scaling_recommendations.csv
  metric_outputs/_comparison/comparison_report.md

The goal is to decide whether predictive/relative signal is preserved when
replacing full_vr with cheaper PH modes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_MODES = [
    "full_vr",
    "landmark_vr",
    "fixed_support_vr",
    "fixed_knn_vr",
    "skip_vr",
    "event_driven",
    "online_landmark_event",
    "online_landmark_dynamic_support",
]

PH_METRICS = [
    "total_persistence_h1",
    "max_persistence_h1",
    "top5_persistence_h1",
    "betti_curve_area_h1",
    "betti_curve_peak_h1",
    "betti_curve_change_h1",
]

NON_PH_METRICS = [
    "effective_rank",
    "top_k_variance_fraction",
    "mean_pairwise_distance",
    "median_pairwise_distance",
    "std_pairwise_distance",
    "q10_pairwise_distance",
    "q50_pairwise_distance",
    "q90_pairwise_distance",
    "projection_residual",
]

DEFAULT_JOIN_KEYS = [
    "experiment",
    "geometry",
    "mechanism",
    "model",
    "seed",
    "epoch",
]

MODE_TIERS = {
    "full_vr": 0,
    "landmark_vr": 1,
    "fixed_support_vr": 2,
    "fixed_knn_vr": 3,
    "skip_vr": 4,
    "event_driven": 5,
    "online_landmark_event": 6,
    "online_landmark_dynamic_support": 7,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_metrics_for_mode(metric_root: Path, mode: str) -> Optional[pd.DataFrame]:
    combined = metric_root / mode / "_combined"
    parquet_path = combined / "all_metrics.parquet"
    csv_path = combined / "all_metrics.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return None

    if "ph_mode" not in df.columns:
        df["ph_mode"] = mode
    else:
        df["ph_mode"] = df["ph_mode"].fillna(mode)

    return df


def load_all_modes(metric_root: Path, modes: Sequence[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for mode in modes:
        df = read_metrics_for_mode(metric_root, mode)
        if df is None:
            print(f"[WARN] missing metrics for mode={mode}")
            continue
        if len(df) == 0:
            print(f"[WARN] empty metrics for mode={mode}")
            continue
        out[mode] = df
        print(f"[INFO] loaded mode={mode}: rows={len(df)}, cols={len(df.columns)}")
    return out


def available_join_keys(df_a: pd.DataFrame, df_b: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    return [k for k in requested if k in df_a.columns and k in df_b.columns]


def finite_pair(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    return x[mask], y[mask]


def corr_or_nan(a: pd.Series, b: pd.Series, method: str) -> float:
    x, y = finite_pair(a, b)
    if len(x) < 3:
        return float("nan")
    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return float("nan")
    return float(x.corr(y, method=method))


def mae_or_nan(a: pd.Series, b: pd.Series) -> float:
    x, y = finite_pair(a, b)
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.abs(x.to_numpy() - y.to_numpy())))


def rmse_or_nan(a: pd.Series, b: pd.Series) -> float:
    x, y = finite_pair(a, b)
    if len(x) == 0:
        return float("nan")
    diff = x.to_numpy() - y.to_numpy()
    return float(np.sqrt(np.mean(diff * diff)))


def normalized_mae_or_nan(ref: pd.Series, pred: pd.Series) -> float:
    x, y = finite_pair(ref, pred)
    if len(x) == 0:
        return float("nan")
    scale = float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(x.to_numpy() - y.to_numpy())) / scale)


def sign_agreement(ref: pd.Series, pred: pd.Series) -> float:
    x, y = finite_pair(ref, pred)
    if len(x) == 0:
        return float("nan")
    sx = np.sign(x.to_numpy())
    sy = np.sign(y.to_numpy())
    valid = (sx != 0) | (sy != 0)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(sx[valid] == sy[valid]))


def curve_delta_frame(df: pd.DataFrame, group_cols: Sequence[str], metric: str) -> pd.DataFrame:
    cols = [c for c in group_cols if c in df.columns]
    if not cols or "epoch" not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    use = df[cols + ["epoch", metric]].dropna(subset=[metric]).copy()
    use = use.sort_values(cols + ["epoch"])
    use[f"delta_{metric}"] = use.groupby(cols, dropna=False)[metric].diff()
    return use[cols + ["epoch", f"delta_{metric}"]]


def transition_epoch(df: pd.DataFrame, group_cols: Sequence[str], metric: str) -> pd.DataFrame:
    """Return epoch of largest absolute step change per run/model for one metric."""
    d = curve_delta_frame(df, group_cols, metric)
    delta_col = f"delta_{metric}"
    if d.empty or delta_col not in d.columns:
        return pd.DataFrame()
    d = d.dropna(subset=[delta_col]).copy()
    if d.empty:
        return pd.DataFrame()
    d["abs_delta"] = d[delta_col].abs()
    idx = d.groupby(list(group_cols), dropna=False)["abs_delta"].idxmax()
    out = d.loc[idx, list(group_cols) + ["epoch", "abs_delta"]].copy()
    out = out.rename(columns={"epoch": f"transition_epoch_{metric}", "abs_delta": f"transition_abs_delta_{metric}"})
    return out


def compute_runtime_summary(all_modes: Dict[str, pd.DataFrame], metric_root: Path) -> pd.DataFrame:
    rows = []
    full_time = None

    for mode, df in all_modes.items():
        total_ph_time = pd.to_numeric(df.get("ph_time_sec", pd.Series(dtype=float)), errors="coerce").sum()
        mean_ph_time = pd.to_numeric(df.get("ph_time_sec", pd.Series(dtype=float)), errors="coerce").mean()
        recomputed_rate = pd.to_numeric(df.get("ph_recomputed", pd.Series(dtype=float)), errors="coerce").mean()
        rows.append({
            "ph_mode": mode,
            "mode_tier": MODE_TIERS.get(mode, 999),
            "rows": int(len(df)),
            "num_runs": int(df["model"].nunique()) if "model" in df.columns else np.nan,
            "total_ph_time_sec": float(total_ph_time),
            "mean_ph_time_sec": float(mean_ph_time) if np.isfinite(mean_ph_time) else np.nan,
            "ph_recomputed_rate": float(recomputed_rate) if np.isfinite(recomputed_rate) else np.nan,
        })

    rt = pd.DataFrame(rows).sort_values(["mode_tier", "ph_mode"])
    if "full_vr" in rt["ph_mode"].values:
        full_time = float(rt.loc[rt["ph_mode"] == "full_vr", "total_ph_time_sec"].iloc[0])
    if full_time and full_time > 0:
        rt["speedup_vs_full_vr"] = full_time / rt["total_ph_time_sec"].replace({0.0: np.nan})
    else:
        rt["speedup_vs_full_vr"] = np.nan

    return rt


def compute_fidelity(
    all_modes: Dict[str, pd.DataFrame],
    full_mode: str,
    join_keys: Sequence[str],
    ph_metrics: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if full_mode not in all_modes:
        raise ValueError(f"reference mode missing: {full_mode}")

    full = all_modes[full_mode]
    rows = []
    transition_rows = []

    group_cols = [c for c in ["experiment", "geometry", "mechanism", "model", "seed"] if c in full.columns]

    for mode, df in all_modes.items():
        if mode == full_mode:
            continue

        keys = available_join_keys(full, df, join_keys)
        if not keys:
            print(f"[WARN] no join keys for mode={mode}; skipping")
            continue

        common_metrics = [m for m in ph_metrics if m in full.columns and m in df.columns]
        if not common_metrics:
            print(f"[WARN] no common PH metrics for mode={mode}; skipping")
            continue

        full_cols = keys + common_metrics
        mode_cols = keys + common_metrics
        joined = full[full_cols].merge(
            df[mode_cols],
            on=keys,
            suffixes=("__full_vr", f"__{mode}"),
            how="inner",
        )

        for metric in common_metrics:
            ref_col = f"{metric}__full_vr"
            pred_col = f"{metric}__{mode}"
            rows.append({
                "ph_mode": mode,
                "metric": metric,
                "n_joined": int(len(joined)),
                "pearson_vs_full_vr": corr_or_nan(joined[ref_col], joined[pred_col], "pearson"),
                "spearman_vs_full_vr": corr_or_nan(joined[ref_col], joined[pred_col], "spearman"),
                "mae_vs_full_vr": mae_or_nan(joined[ref_col], joined[pred_col]),
                "rmse_vs_full_vr": rmse_or_nan(joined[ref_col], joined[pred_col]),
                "normalized_mae_vs_full_vr": normalized_mae_or_nan(joined[ref_col], joined[pred_col]),
            })

        # Compare epoch-to-epoch change direction and transition epoch.
        for metric in common_metrics:
            full_delta = curve_delta_frame(full, group_cols, metric)
            mode_delta = curve_delta_frame(df, group_cols, metric)
            delta_col = f"delta_{metric}"
            if not full_delta.empty and not mode_delta.empty:
                dk = available_join_keys(full_delta, mode_delta, list(group_cols) + ["epoch"])
                djoined = full_delta.merge(
                    mode_delta,
                    on=dk,
                    suffixes=("__full_vr", f"__{mode}"),
                    how="inner",
                )
                rows.append({
                    "ph_mode": mode,
                    "metric": f"delta_{metric}",
                    "n_joined": int(len(djoined)),
                    "pearson_vs_full_vr": corr_or_nan(djoined[f"{delta_col}__full_vr"],
                                                      djoined[f"{delta_col}__{mode}"], "pearson"),
                    "spearman_vs_full_vr": corr_or_nan(djoined[f"{delta_col}__full_vr"],
                                                       djoined[f"{delta_col}__{mode}"], "spearman"),
                    "mae_vs_full_vr": mae_or_nan(djoined[f"{delta_col}__full_vr"],
                                                 djoined[f"{delta_col}__{mode}"]),
                    "rmse_vs_full_vr": rmse_or_nan(djoined[f"{delta_col}__full_vr"],
                                                   djoined[f"{delta_col}__{mode}"]),
                    "normalized_mae_vs_full_vr": normalized_mae_or_nan(djoined[f"{delta_col}__full_vr"],
                                                                       djoined[f"{delta_col}__{mode}"]),
                    "sign_agreement_vs_full_vr": sign_agreement(djoined[f"{delta_col}__full_vr"],
                                                                djoined[f"{delta_col}__{mode}"]),
                })

            ft = transition_epoch(full, group_cols, metric)
            mt = transition_epoch(df, group_cols, metric)
            if not ft.empty and not mt.empty:
                tk = available_join_keys(ft, mt, group_cols)
                tj = ft.merge(mt, on=tk, suffixes=("__full_vr", f"__{mode}"), how="inner")
                fcol = f"transition_epoch_{metric}__full_vr"
                mcol = f"transition_epoch_{metric}__{mode}"
                if fcol in tj.columns and mcol in tj.columns and len(tj) > 0:
                    err = (pd.to_numeric(tj[mcol], errors="coerce") - pd.to_numeric(tj[fcol], errors="coerce")).abs()
                    transition_rows.append({
                        "ph_mode": mode,
                        "metric": metric,
                        "n_joined": int(len(tj)),
                        "mean_transition_epoch_abs_error": float(err.mean()),
                        "median_transition_epoch_abs_error": float(err.median()),
                        "max_transition_epoch_abs_error": float(err.max()),
                    })

    fidelity = pd.DataFrame(rows)
    transitions = pd.DataFrame(transition_rows)
    return fidelity, transitions


def summarize_fidelity(fidelity: pd.DataFrame, runtime: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    if fidelity.empty:
        return pd.DataFrame()

    # Main metrics only. Delta rows are useful diagnostically but noisier for the top-level score.
    main = fidelity[~fidelity["metric"].astype(str).str.startswith("delta_")].copy()
    if main.empty:
        main = fidelity.copy()

    rows = []
    for mode, g in main.groupby("ph_mode"):
        spearman = pd.to_numeric(g["spearman_vs_full_vr"], errors="coerce")
        pearson = pd.to_numeric(g["pearson_vs_full_vr"], errors="coerce")
        nmae = pd.to_numeric(g["normalized_mae_vs_full_vr"], errors="coerce")

        delta_g = fidelity[(fidelity["ph_mode"] == mode) & fidelity["metric"].astype(str).str.startswith("delta_")]
        delta_spearman = pd.to_numeric(delta_g.get("spearman_vs_full_vr", pd.Series(dtype=float)), errors="coerce")
        sign_agree = pd.to_numeric(delta_g.get("sign_agreement_vs_full_vr", pd.Series(dtype=float)), errors="coerce")

        trans_g = transitions[transitions["ph_mode"] == mode] if not transitions.empty else pd.DataFrame()
        median_transition_error = pd.to_numeric(
            trans_g.get("median_transition_epoch_abs_error", pd.Series(dtype=float)),
            errors="coerce",
        ).median()

        rt_row = runtime[runtime["ph_mode"] == mode]
        speedup = float(rt_row["speedup_vs_full_vr"].iloc[0]) if len(rt_row) else np.nan
        total_time = float(rt_row["total_ph_time_sec"].iloc[0]) if len(rt_row) else np.nan
        recompute_rate = float(rt_row["ph_recomputed_rate"].iloc[0]) if len(rt_row) else np.nan

        median_spearman = float(spearman.median()) if spearman.notna().any() else np.nan
        median_pearson = float(pearson.median()) if pearson.notna().any() else np.nan
        median_nmae = float(nmae.median()) if nmae.notna().any() else np.nan
        median_delta_spearman = float(delta_spearman.median()) if delta_spearman.notna().any() else np.nan
        median_sign_agreement = float(sign_agree.median()) if sign_agree.notna().any() else np.nan

        # A pragmatic score. Rank fidelity/trend preservation first, then speed.
        fidelity_component = np.nanmean([
            median_spearman,
            median_pearson,
            median_delta_spearman,
            median_sign_agreement,
        ])
        error_penalty = 0.0 if not np.isfinite(median_nmae) else min(max(median_nmae, 0.0), 1.0)
        speed_component = 0.0 if (not np.isfinite(speedup) or
                                  speedup <= 0) else min(math.log1p(speedup) / math.log1p(20.0), 1.0)
        score = 0.75 * fidelity_component + 0.25 * speed_component - 0.25 * error_penalty

        rows.append({
            "ph_mode": mode,
            "mode_tier": MODE_TIERS.get(mode, 999),
            "median_spearman_vs_full_vr": median_spearman,
            "median_pearson_vs_full_vr": median_pearson,
            "median_normalized_mae_vs_full_vr": median_nmae,
            "median_delta_spearman_vs_full_vr": median_delta_spearman,
            "median_delta_sign_agreement_vs_full_vr": median_sign_agreement,
            "median_transition_epoch_abs_error": float(median_transition_error) if
            np.isfinite(median_transition_error) else np.nan,
            "speedup_vs_full_vr": speedup,
            "total_ph_time_sec": total_time,
            "ph_recomputed_rate": recompute_rate,
            "scaling_score": float(score) if np.isfinite(score) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["scaling_score", "mode_tier"], ascending=[False, True])


def compute_metric_relationships(all_modes: Dict[str, pd.DataFrame], ph_metrics: Sequence[str],
                                 non_ph_metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for mode, df in all_modes.items():
        pms = [m for m in ph_metrics if m in df.columns]
        nms = [m for m in non_ph_metrics if m in df.columns]
        for pm in pms:
            for nm in nms:
                rows.append({
                    "ph_mode": mode,
                    "ph_metric": pm,
                    "other_metric": nm,
                    "n": int(finite_pair(df[pm], df[nm])[0].shape[0]),
                    "pearson": corr_or_nan(df[pm], df[nm], "pearson"),
                    "spearman": corr_or_nan(df[pm], df[nm], "spearman"),
                })
    return pd.DataFrame(rows)


def make_recommendations(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    rows = []
    for _, r in summary.iterrows():
        mode = r["ph_mode"]
        sp = r.get("median_spearman_vs_full_vr", np.nan)
        dsp = r.get("median_delta_spearman_vs_full_vr", np.nan)
        nmae = r.get("median_normalized_mae_vs_full_vr", np.nan)
        speed = r.get("speedup_vs_full_vr", np.nan)
        trans = r.get("median_transition_epoch_abs_error", np.nan)

        flags = []
        if np.isfinite(sp) and sp >= 0.9:
            flags.append("excellent rank fidelity")
        elif np.isfinite(sp) and sp >= 0.75:
            flags.append("usable rank fidelity")
        else:
            flags.append("weak rank fidelity")

        if np.isfinite(dsp) and dsp >= 0.7:
            flags.append("trend changes preserved")
        elif np.isfinite(dsp):
            flags.append("trend changes questionable")

        if np.isfinite(nmae) and nmae <= 0.15:
            flags.append("low normalized error")
        elif np.isfinite(nmae) and nmae <= 0.35:
            flags.append("moderate normalized error")
        elif np.isfinite(nmae):
            flags.append("high normalized error")

        if np.isfinite(speed) and speed >= 3:
            flags.append("material speedup")
        elif np.isfinite(speed) and speed > 1:
            flags.append("small speedup")
        elif np.isfinite(speed):
            flags.append("no speedup")

        if np.isfinite(trans) and trans <= 2:
            flags.append("transition timing preserved")
        elif np.isfinite(trans):
            flags.append("transition timing drift")

        if (np.isfinite(sp) and sp >= 0.85 and
            (not np.isfinite(nmae) or nmae <= 0.25) and
            (not np.isfinite(speed) or speed >= 1.5)):
            decision = "scale_candidate"
        elif np.isfinite(sp) and sp >= 0.7:
            decision = "diagnostic_only_or_tune"
        else:
            decision = "do_not_scale_without_changes"

        rows.append({
            "ph_mode": mode,
            "decision": decision,
            "reason": "; ".join(flags),
            "scaling_score": r.get("scaling_score", np.nan),
            "speedup_vs_full_vr": speed,
            "median_spearman_vs_full_vr": sp,
            "median_normalized_mae_vs_full_vr": nmae,
        })

    return pd.DataFrame(rows).sort_values(["decision", "scaling_score"], ascending=[True, False])


def write_report(
    out_dir: Path,
    runtime: pd.DataFrame,
    fidelity_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    modes_loaded: Sequence[str],
) -> None:
    lines = []
    lines.append("# PH mode runtime/fidelity comparison")
    lines.append("")
    lines.append("## Modes loaded")
    lines.append("")
    for m in modes_loaded:
        lines.append(f"- `{m}`")
    lines.append("")

    lines.append("## Runtime summary")
    lines.append("")
    if runtime.empty:
        lines.append("No runtime data available.")
    else:
        cols = [c for c in ["ph_mode",
                            "rows",
                            "num_runs",
                            "total_ph_time_sec",
                            "mean_ph_time_sec",
                            "speedup_vs_full_vr",
                            "ph_recomputed_rate"] if c in runtime.columns]
        lines.append(runtime[cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Fidelity summary")
    lines.append("")
    if fidelity_summary.empty:
        lines.append("No fidelity summary available. Ensure `full_vr` and at least one shortcut mode completed.")
    else:
        cols = [c for c in [
            "ph_mode",
            "median_spearman_vs_full_vr",
            "median_delta_spearman_vs_full_vr",
            "median_normalized_mae_vs_full_vr",
            "median_transition_epoch_abs_error",
            "speedup_vs_full_vr",
            "scaling_score",
        ] if c in fidelity_summary.columns]
        lines.append(fidelity_summary[cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Scaling recommendations")
    lines.append("")
    if recommendations.empty:
        lines.append("No recommendations available.")
    else:
        cols = [c for c in ["ph_mode", "decision", "reason", "scaling_score"] if c in recommendations.columns]
        lines.append(recommendations[cols].to_markdown(index=False))
    lines.append("")

    lines.append("## How to read this")
    lines.append("")
    lines.append("Use `full_vr` as the reference, but do not demand exact equality. "
                 "For scaling, prefer modes that preserve rank/order and epoch-wise trends while "
                 "producing material speedup. A shortcut with biased absolute persistence values "
                 "can still be acceptable if its Spearman correlation, trend sign agreement, "
                 "mechanism ordering, and downstream relationships are stable.")
    lines.append("")

    (out_dir / "comparison_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PH shortcut modes against full_vr outputs.")
    parser.add_argument(
        "--metric-root",
        default="/mnt/wd_black/research/evolve_collapse/metric_outputs",
        help="Root directory containing per-mode metric outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for comparison tables. Defaults to <metric-root>/_comparison.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        help="PH modes to compare.",
    )
    parser.add_argument(
        "--full-mode",
        default="full_vr",
        help="Reference mode.",
    )
    parser.add_argument(
        "--join-keys",
        nargs="+",
        default=DEFAULT_JOIN_KEYS,
        help="Columns used to align checkpoints across modes.",
    )
    args = parser.parse_args()

    metric_root = Path(args.metric_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else metric_root / "_comparison"
    ensure_dir(out_dir)

    all_modes = load_all_modes(metric_root, args.modes)
    if args.full_mode not in all_modes:
        raise SystemExit(f"[ERROR] missing reference mode {args.full_mode!r} under {metric_root}")

    runtime = compute_runtime_summary(all_modes, metric_root)
    fidelity, transitions = compute_fidelity(all_modes, args.full_mode, args.join_keys, PH_METRICS)
    fidelity_summary = summarize_fidelity(fidelity, runtime, transitions)
    relationships = compute_metric_relationships(all_modes, PH_METRICS, NON_PH_METRICS)
    recommendations = make_recommendations(fidelity_summary)

    runtime.to_csv(out_dir / "runtime_by_mode.csv", index=False)
    fidelity.to_csv(out_dir / "fidelity_by_mode_metric.csv", index=False)
    transitions.to_csv(out_dir / "transition_epoch_errors.csv", index=False)
    fidelity_summary.to_csv(out_dir / "fidelity_summary_by_mode.csv", index=False)
    relationships.to_csv(out_dir / "metric_relationships_by_mode.csv", index=False)
    recommendations.to_csv(out_dir / "scaling_recommendations.csv", index=False)

    # Parquet mirrors when possible.
    for name, df in [
        ("runtime_by_mode", runtime),
        ("fidelity_by_mode_metric", fidelity),
        ("transition_epoch_errors", transitions),
        ("fidelity_summary_by_mode", fidelity_summary),
        ("metric_relationships_by_mode", relationships),
        ("scaling_recommendations", recommendations),
    ]:
        try:
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
        except Exception as exc:
            print(f"[WARN] could not write parquet for {name}: {exc}")

    write_report(out_dir, runtime, fidelity_summary, recommendations, sorted(all_modes))

    print(f"[DONE] wrote comparison outputs to {out_dir}")
    if not recommendations.empty:
        print("[SUMMARY] recommendations:")
        print(recommendations[["ph_mode", "decision", "scaling_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
