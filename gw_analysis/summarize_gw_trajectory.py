"""
Trajectory derived summary metrics.

Should have:
  - gw_path_length = sum_t d_t
  - gw_mean_step = mean_t d_t
  - gw_max_step = max_t d_t
  - gw_min_step = min_t d_t
  - gw_auc = auc from start
  - gw_final_from_start = GW(X_0, X_T)
  - gw_peak_epoch = argmax_t d_t
  - gw_acceleration = mean |d_t = d_{t-1}|
  - gw_front_load = early_path_length / total_path_length
  - gw_late_instability = late_path_length / total_path_length

Args:
  --gw-root
  --output table_gw_trajectory_summary.parquet
  --epoch-max optional
  --early-frac 0.25
  --late-frac 0.25
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = [
    "run_id",
    "experiment",
    "mechanism",
    "model",
    "geometry",
    "schedule",
    "severity",
    "n",
    "d",
    "k",
    "seed",
    "mover_frac",
    "noise",
    "comparison_type",
    "n_landmarks",
    "landmark_seed",
    "normalization",
    "epsilon"
]


def read_gw_files(gw_root):
    """
    Read GW pair-output parquet files from a directory tree.

    Files that do not contain pairwise GW rows are skipped. This avoids
    accidentally reading tranche manifests, summary tables, MDS outputs, or
    other parquet files living under the same output root.

    :param gw_root: Root directory containing GW output parquet files.
    :return: Concatenated dataframe of GW pair rows.
    """
    gw_root = Path(gw_root)
    files = sorted(gw_root.rglob("*.parquet"))

    required = {
        "source_epoch",
        "target_epoch",
        "gw_distance",
    }

    frames = []
    skipped = []

    for path in files:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            skipped.append((str(path), f"read_failed: {exc}"))
            continue

        if not required.issubset(set(df.columns)):
            skipped.append((str(path), "missing_required_columns"))
            continue

        df = df.dropna(subset=["source_epoch", "target_epoch", "gw_distance"])

        if len(df) == 0:
            skipped.append((str(path), "no_valid_gw_rows"))
            continue

        df["gw_output_path"] = str(path)
        frames.append(df)

    if skipped:
        print("[INFO] skipped non-GW parquet files:")
        for path, reason in skipped[:20]:
            print(f"  {reason}: {path}")
        if len(skipped) > 20:
            print(f"  ... plus {len(skipped) - 20} more")

    if not frames:
        raise RuntimeError(f"No valid GW output parquet files found under {gw_root}")

    return pd.concat(frames, ignore_index=True)


def safe_auc(x, y):
    """
    Avoid division errors when computing AUC.
    :param x: numpy array.
    :param y: numpy array.
    :return: float, safe AUC.
    """
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    auc = float(np.trapz(y[order], x[order]))
    return auc


def summarize_one_group(df):
    """
    Summarizes a single experimental group.

    :param df: cutback df.
    :return: dictionary of summary metric values.
    """
    df = df.sort_values(["source_epoch", "target_epoch"]).copy()
    distances = df["gw_distance"].to_numpy(dtype=float)
    source_epochs = df["source_epoch"].to_numpy(dtype=float)
    target_epochs = df["target_epoch"].to_numpy(dtype=float)
    epoch_delta = np.maximum(target_epochs - source_epochs, 1.0)
    step_speed = distances / epoch_delta
    out = {}

    first = df.iloc[0]
    for col in ID_COLS:
        if col in df.columns:
            out[col] = first.get(col)

    out["num_pairs"] = int(len(df))
    out["source_epoch_min"] = int(df["source_epoch"].min())
    out["source_epoch_max"] = int(df["source_epoch"].max())

    out["gw_path_length_proxy"] = float(np.nansum(distances))
    out["gw_mean_distance"] = float(np.nanmean(distances))
    out["gw_median_distance"] = float(np.nanmedian(distances))
    out["gw_max_distance"] = float(np.nanmax(distances))
    out["gw_std_distance"] = float(np.nanstd(distances))

    out["gw_mean_speed"] = float(np.nanmean(step_speed))
    out["gw_max_speed"] = float(np.nanmax(step_speed))

    max_idx = int(np.nanargmax(distances))
    out["gw_peak_source_epoch"] = int(source_epochs[max_idx])
    out["gw_peak_target_epoch"] = int(target_epochs[max_idx])
    out["gw_peak_distance"] = float(distances[max_idx])

    if len(distances) >= 2:
        acceleration = np.abs(np.diff(distances))
        out["gw_mean_abs_acceleration"] = float(np.nanmean(acceleration))
        out["gw_max_abs_acceleration"] = float(np.nanmax(acceleration))
    else:
        out["gw_mean_abs_acceleration"] = float("nan")
        out["gw_max_abs_acceleration"] = float("nan")

    if first.get("comparison_type") == "from_start":
        out["gw_auc_from_start"] = safe_auc(target_epochs, distances)
        out["gw_final_from_start"] = float(
            df.sort_values("target_epoch")["gw_distance"].iloc[-1]
        )
    else:
        out["gw_auc_from_start"] = float("nan")
        out["gw_final_from_start"] = float("nan")

    total = float(np.nansum(distances))
    if total > 0 and len(distances) >= 4:
        n = len(distances)
        early_n = max(1, int(np.ceil(0.25 * n)))
        late_n = max(1, int(np.ceil(0.25 * n)))
        out["gw_front_loadedness"] = float(np.nansum(distances[:early_n]) / total)
        out["gw_late_instability"] = float(np.nansum(distances[-late_n:]) / total)
    else:
        out["gw_front_loadedness"] = float("nan")
        out["gw_late_instability"] = float("nan")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gw_root = Path(args.gw_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    df = read_gw_files(gw_root)

    group_cols = [c for c in ID_COLS if c in df.columns]
    rows = []

    for _, group in df.groupby(group_cols, dropna=False):
        rows.append(summarize_one_group(group))

    summary = pd.DataFrame(rows)

    if output.suffix == ".parquet":
        summary.to_parquet(output, index=False)
    elif output.suffix == ".csv":
        summary.to_csv(output, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")

    print(f"[DONE] wrote {output}")
    print(f"[DONE] rows={len(summary)}")


if __name__ == "__main__":
    main()
