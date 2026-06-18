import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXTERNAL_ROOT = "/media/alkal/WD_BLACK/evolve_collapse"
METRIC_ROOT = os.path.join(EXTERNAL_ROOT, "../old/metric_outputs")
SUMMARY_ROOT = os.path.join(EXTERNAL_ROOT, "../old/metric_summaries")
ASSET_ROOT = os.path.join(EXTERNAL_ROOT, "summary_assets")

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.20]
DEFAULT_WINDOWS = [1, 3]

GROUP_COLS = [
    "ph_mode",
    "experiment",
    "geometry",
    "mechanism",
    "model",
    "schedule",
    "severity",
    "n",
    "d",
    "k",
    "seed",
    "mover_frac",
    "noise",
]

METRIC_SPECS = {
    # Spectral: decreasing values indicate collapse.
    "effective_rank": {
        "family": "spectral",
        "direction": "decrease",
    },
    # Spectral: increasing values indicate concentration/collapse.
    "top_k_variance_fraction": {
        "family": "spectral",
        "direction": "increase",
    },

    # Geometric: most contraction metrics decrease under collapse.
    "mean_pairwise_distance": {
        "family": "geometric",
        "direction": "decrease",
    },
    "median_pairwise_distance": {
        "family": "geometric",
        "direction": "decrease",
    },
    "std_pairwise_distance": {
        "family": "geometric",
        "direction": "decrease",
    },
    "q10_pairwise_distance": {
        "family": "geometric",
        "direction": "decrease",
    },
    "q50_pairwise_distance": {
        "family": "geometric",
        "direction": "decrease",
    },
    "q90_pairwise_distance": {
        "family": "geometric",
        "direction": "decrease",
    },
    "projection_residual": {
        "family": "geometric",
        "direction": "decrease",
    },

    # Topological: collapse usually reduces persistence / Betti mass.
    "total_persistence_h1": {
        "family": "topological",
        "direction": "decrease",
    },
    "max_persistence_h1": {
        "family": "topological",
        "direction": "decrease",
    },
    "top5_persistence_h1": {
        "family": "topological",
        "direction": "decrease",
    },
    "betti_curve_area_h1": {
        "family": "topological",
        "direction": "decrease",
    },
    "betti_curve_peak_h1": {
        "family": "topological",
        "direction": "decrease",
    },
    # Change from baseline is already a deviation score, so increase means collapse/change.
    "betti_curve_change_h1": {
        "family": "topological",
        "direction": "increase",
    },
}


def read_metric_inputs(metric_dir: str, input_file: str) -> pd.DataFrame:
    if input_file is not None:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return pd.read_csv(path)

    metric_dir_path = Path(metric_dir)
    paths = sorted(metric_dir_path.glob("*_all_metrics.csv"))

    if not paths:
        paths = sorted(metric_dir_path.glob("*.csv"))

    if not paths:
        raise FileNotFoundError(f"No metric CSV files found in {metric_dir_path}")

    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                frames.append(df)
        except Exception as exc:
            print(f"[WARN] failed to read {path}: {exc}")

    if not frames:
        raise RuntimeError(f"No readable metric CSV files found in {metric_dir_path}")

    return pd.concat(frames, ignore_index=True)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in GROUP_COLS:
        if col not in df.columns:
            df[col] = None

    if "epoch" not in df.columns:
        raise ValueError("Input metrics must contain an 'epoch' column.")

    if "ph_mode" not in df.columns:
        df["ph_mode"] = "unknown"

    return df


def available_metric_specs(df: pd.DataFrame) -> dict:
    return {
        metric: spec
        for metric, spec in METRIC_SPECS.items()
        if metric in df.columns
    }


def collapse_score(values: pd.Series, direction: str, eps: float) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    baseline = values.iloc[0]

    if pd.isna(baseline):
        return pd.Series(np.nan, index=values.index)

    denom = abs(float(baseline)) + eps

    if direction == "decrease":
        return (baseline - values) / denom

    if direction == "increase":
        return (values - baseline) / denom

    raise ValueError(f"Unknown direction: {direction}")


def first_sustained_crossing(scores: np.ndarray, threshold: float, window: int):
    finite = np.isfinite(scores)
    n = len(scores)

    if n == 0:
        return False, None

    for i in range(n):
        j = i + window
        if j > n:
            break

        segment = scores[i:j]
        segment_finite = finite[i:j]

        if segment_finite.all() and (segment >= threshold).all():
            return True, i

    return False, None


def summarize_one_metric(
    group_df,
    group_key,
    group_cols,
    metric,
    spec,
    thresholds,
    windows,
    eps
):
    ordered = group_df.sort_values("epoch").copy()
    epochs = ordered["epoch"].to_numpy()

    values = pd.to_numeric(ordered[metric], errors="coerce")
    valid_count = int(values.notna().sum())

    if valid_count < 2:
        return []

    scores = collapse_score(values, spec["direction"], eps)
    score_values = scores.to_numpy(dtype=float)

    max_score = float(np.nanmax(score_values)) if np.isfinite(score_values).any() else np.nan
    final_score = float(score_values[-1]) if len(score_values) else np.nan

    rows = []

    if not isinstance(group_key, tuple):
        group_key = (group_key,)

    base = dict(zip(group_cols, group_key))

    t_min = float(np.nanmin(epochs)) if len(epochs) else np.nan
    t_max = float(np.nanmax(epochs)) if len(epochs) else np.nan
    t_span = t_max - t_min if np.isfinite(t_min) and np.isfinite(t_max) else np.nan

    for threshold in thresholds:
        for window in windows:
            detected, pos = first_sustained_crossing(
                score_values,
                threshold=threshold,
                window=window,
            )

            if detected:
                t_detect = float(epochs[pos])
                normalized_t_detect = (
                    (t_detect - t_min) / t_span
                    if np.isfinite(t_span) and t_span > 0
                    else 0.0
                )
            else:
                t_detect = float(t_max + 1) if np.isfinite(t_max) else np.nan
                normalized_t_detect = 1.0

            row = {
                **base,
                "metric": metric,
                "metric_family": spec["family"],
                "metric_direction": spec["direction"],
                "threshold": threshold,
                "window": window,
                "detected": bool(detected),
                "t_detect": t_detect,
                "normalized_t_detect": normalized_t_detect,
                "final_collapse_score": final_score,
                "max_collapse_score": max_score,
                "num_checkpoints": int(len(ordered)),
                "num_valid_values": valid_count,
                "start_epoch": t_min,
                "end_epoch": t_max,
            }
            rows.append(row)

    return rows


def compute_detection_times(
    df,
    thresholds,
    windows,
    eps,
) -> pd.DataFrame:
    df = ensure_columns(df)
    specs = available_metric_specs(df)

    if not specs:
        raise ValueError(
            "No known metric columns found. "
            f"Expected one or more of: {list(METRIC_SPECS.keys())}"
        )

    group_cols = [c for c in GROUP_COLS if c in df.columns]

    rows = []
    grouped = df.groupby(group_cols, dropna=False)

    for group_key, group_df in grouped:
        for metric, spec in specs.items():
            rows.extend(
                summarize_one_metric(
                    group_df=group_df,
                    group_key=group_key,
                    group_cols=group_cols,
                    metric=metric,
                    spec=spec,
                    thresholds=thresholds,
                    windows=windows,
                    eps=eps,
                )
            )

    return pd.DataFrame(rows)


def aggregate_detection_times(det_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "threshold",
        "window",
        "metric_family",
        "metric",
        "ph_mode",
        "mechanism",
        "geometry",
        "severity",
        "mover_frac",
        "noise",
        "n",
        "d",
    ]
    group_cols = [c for c in group_cols if c in det_df.columns]

    agg = (
        det_df
        .groupby(group_cols, dropna=False)
        .agg(
            runs=("t_detect", "size"),
            detected_runs=("detected", "sum"),
            detection_rate=("detected", "mean"),
            median_t_detect=("t_detect", "median"),
            mean_t_detect=("t_detect", "mean"),
            std_t_detect=("t_detect", "std"),
            median_normalized_t_detect=("normalized_t_detect", "median"),
            mean_normalized_t_detect=("normalized_t_detect", "mean"),
            iqr_normalized_t_detect=(
                "normalized_t_detect",
                lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
            ),
            median_final_collapse_score=("final_collapse_score", "median"),
            median_max_collapse_score=("max_collapse_score", "median"),
        )
        .reset_index()
    )

    agg["no_detection_rate"] = 1.0 - agg["detection_rate"]

    return agg


def metric_family_win_rates(det_df: pd.DataFrame, threshold: float, window: int) -> pd.DataFrame:
    """
    Computes which metric family detects earliest within each configuration.

    This uses the best metric within each family for a run/configuration, then
    compares family-level best detection times.
    """
    df = det_df[
        (det_df["threshold"] == threshold)
        & (det_df["window"] == window)
    ].copy()

    config_cols = [
        "ph_mode",
        "experiment",
        "geometry",
        "mechanism",
        "model",
        "schedule",
        "severity",
        "n",
        "d",
        "k",
        "seed",
        "mover_frac",
        "noise",
    ]
    config_cols = [c for c in config_cols if c in df.columns]

    family_best = (
        df
        .groupby(config_cols + ["metric_family"], dropna=False)
        .agg(best_family_t_detect=("t_detect", "min"))
        .reset_index()
    )

    if family_best.empty:
        return family_best

    min_by_config = (
        family_best
        .groupby(config_cols, dropna=False)["best_family_t_detect"]
        .transform("min")
    )

    family_best["is_winner"] = family_best["best_family_t_detect"] == min_by_config

    summary_cols = [
        "mechanism",
        "geometry",
        "severity",
        "mover_frac",
        "noise",
        "n",
        "d",
        "ph_mode",
    ]
    summary_cols = [c for c in summary_cols if c in family_best.columns]

    win = (
        family_best
        .groupby(summary_cols + ["metric_family"], dropna=False)
        .agg(win_rate=("is_winner", "mean"), count=("is_winner", "size"))
        .reset_index()
    )

    wide = win.pivot_table(
        index=summary_cols,
        columns="metric_family",
        values="win_rate",
        fill_value=0.0,
    ).reset_index()

    wide.columns.name = None

    for family in ["spectral", "geometric", "topological"]:
        if family not in wide.columns:
            wide[family] = 0.0

    wide = wide.rename(
        columns={
            "spectral": "spectral_win_rate",
            "geometric": "geometric_win_rate",
            "topological": "topological_win_rate",
        }
    )

    return wide


def write_outputs(
    det_df: pd.DataFrame,
    out_dir: str,
    asset_dir: str,
    primary_threshold: float,
    primary_window: int,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = Path(asset_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    det_path = out / "detection_times.csv"
    agg_path = out / "detection_times_aggregated.csv"
    win_path = out / "detection_win_rates.csv"

    det_df.to_csv(det_path, index=False)

    agg = aggregate_detection_times(det_df)
    agg.to_csv(agg_path, index=False)

    wins = metric_family_win_rates(
        det_df,
        threshold=primary_threshold,
        window=primary_window,
    )
    wins.to_csv(win_path, index=False)

    # Paper-table friendly aliases.
    agg.to_csv(tables / "table02_detection_time_summary.csv", index=False)
    wins.to_csv(tables / "detection_win_rates.csv", index=False)

    print(f"[DONE] wrote {det_path}")
    print(f"[DONE] wrote {agg_path}")
    print(f"[DONE] wrote {win_path}")
    print(f"[DONE] wrote {tables / 'table02_detection_time_summary.csv'}")


def parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Compute first sustained detection times from metric traces."
    )
    parser.add_argument(
        "--metric-dir",
        default=METRIC_ROOT,
        help="Directory containing metric CSVs.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional single metric CSV to use instead of scanning metric-dir.",
    )
    parser.add_argument(
        "--out-dir",
        default=SUMMARY_ROOT,
        help="Directory where detection-time summaries will be written.",
    )
    parser.add_argument(
        "--asset-dir",
        default=ASSET_ROOT,
        help="Directory where paper-friendly tables will be written.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.20",
        help="Comma-separated collapse-score thresholds.",
    )
    parser.add_argument(
        "--windows",
        default="1,3",
        help="Comma-separated sustained-window lengths.",
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.10,
        help="Primary threshold for paper tables.",
    )
    parser.add_argument(
        "--primary-window",
        type=int,
        default=3,
        help="Primary sustained window for paper tables.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Small constant for normalized collapse scores.",
    )

    args = parser.parse_args()

    thresholds = parse_float_list(args.thresholds)
    windows = parse_int_list(args.windows)

    df = read_metric_inputs(
        metric_dir=args.metric_dir,
        input_file=args.input_file,
    )

    det_df = compute_detection_times(
        df,
        thresholds=thresholds,
        windows=windows,
        eps=args.eps,
    )

    write_outputs(
        det_df,
        out_dir=args.out_dir,
        asset_dir=args.asset_dir,
        primary_threshold=args.primary_threshold,
        primary_window=args.primary_window,
    )


if __name__ == "__main__":
    main()