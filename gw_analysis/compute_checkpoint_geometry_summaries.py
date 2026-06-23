"""
Compute checkpoint-level geometry summaries for parquet trajectories.

This script is intended to complement GW trajectory tracing by recording
basic scale, radius, and anisotropy diagnostics for each checkpoint.

Example
-------
python compute_checkpoint_geometry_summaries.py \\
    --run-dir "$EVOLVE_ROOT/evolve_checkpoints/collapse_ph/<run_id>" \\
    --output "$EVOLVE_ROOT/gw_outputs/<run_id>__checkpoint_geometry.parquet"
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


CKPT_RE = re.compile(r"ckpt_epoch_(\d+)\.parquet$")


def epoch_from_path(path):
    """
    Parse the checkpoint epoch from a parquet checkpoint path.

    :param path: Path to checkpoint file.
    :return: Integer epoch, or -1 if the filename does not match.
    """
    match = CKPT_RE.match(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def load_x(path):
    """
    Load a point cloud from a parquet checkpoint.

    Coordinate columns are expected to be named ``dim_0000``, ``dim_0001``,
    and so on. If those columns are unavailable, all numeric columns are used.

    :param path: Path to parquet checkpoint.
    :return: Point cloud as a numpy array.
    """
    df = pd.read_parquet(path)

    dim_cols = sorted([c for c in df.columns if str(c).startswith("dim_")])
    if dim_cols:
        return df[dim_cols].to_numpy(dtype=float, copy=True)

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric coordinate columns found in {path}")

    return numeric.to_numpy(dtype=float, copy=True)


def checkpoint_paths(run_dir):
    """
    Find checkpoint parquet files for one run directory.

    The manifest is preferred when available because it is the intended source
    of truth for the parquet checkpoint layout. If the manifest is missing or
    stale, this falls back to ``run_dir/checkpoints/ckpt_epoch_*.parquet``.

    :param run_dir: Run directory.
    :return: Sorted list of checkpoint paths.
    """
    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            paths = []

            for item in manifest.get("checkpoints", []) or []:
                raw = item.get("path")
                if not raw:
                    continue

                p = Path(raw)
                if not p.is_absolute():
                    p = run_dir / p

                if p.exists() and CKPT_RE.match(p.name):
                    paths.append(p)

            if paths:
                return sorted(paths, key=epoch_from_path)

        except Exception as exc:
            print(f"[WARN] failed to read manifest {manifest_path}: {exc}")

    ckpt_dir = run_dir / "checkpoints"
    paths = sorted(ckpt_dir.glob("ckpt_epoch_*.parquet"), key=epoch_from_path)

    if not paths:
        paths = sorted(run_dir.glob("ckpt_epoch_*.parquet"), key=epoch_from_path)

    if not paths:
        raise FileNotFoundError(f"No checkpoint parquet files found under {run_dir}")

    return paths


def participation_ratio(x):
    """
    Compute a simple participation ratio from centered singular values.

    This is a lightweight effective-dimension proxy. It is not intended to
    replace a full intrinsic-dimension estimator, but it is useful as a cheap
    checkpoint-level anisotropy diagnostic.

    :param x: Point cloud with shape ``(n_points, ambient_dim)``.
    :return: Participation ratio.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    svals = np.linalg.svd(xc, compute_uv=False)
    eig = svals * svals

    denom = np.sum(eig * eig)
    if denom <= 0:
        return 0.0

    return float((np.sum(eig) ** 2) / denom)


def summarize_checkpoint(x):
    """
    Compute geometry summaries for a single checkpoint.

    :param x: Point cloud.
    :return: Dictionary of checkpoint-level geometry summaries.
    """
    distances = pdist(x)
    positive = distances[distances > 0]

    if positive.size == 0:
        positive = np.array([0.0])

    centroid = x.mean(axis=0)
    radius = np.linalg.norm(x - centroid, axis=1)

    qd = np.quantile(positive, [0.05, 0.25, 0.5, 0.75, 0.95])
    qr = np.quantile(radius, [0.5, 0.9, 0.95, 0.99])

    return {
        "n_points": int(x.shape[0]),
        "ambient_dim": int(x.shape[1]),
        "pairwise_q05": float(qd[0]),
        "pairwise_q25": float(qd[1]),
        "pairwise_median": float(qd[2]),
        "pairwise_q75": float(qd[3]),
        "pairwise_q95": float(qd[4]),
        "pairwise_iqr": float(qd[3] - qd[1]),
        "pairwise_mean": float(np.mean(positive)),
        "pairwise_std": float(np.std(positive)),
        "radius_mean": float(np.mean(radius)),
        "radius_median": float(qr[0]),
        "radius_q90": float(qr[1]),
        "radius_q95": float(qr[2]),
        "radius_q99": float(qr[3]),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "participation_ratio": participation_ratio(x),
    }


def main():
    """
    Run the checkpoint geometry summary CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = checkpoint_paths(run_dir)

    rows = []

    for path in paths:
        print(f"[CHECKPOINT] {path.name}")
        x = load_x(path)

        row = summarize_checkpoint(x)
        row["run_dir"] = str(run_dir)
        row["checkpoint_path"] = str(path)
        row["epoch"] = epoch_from_path(path)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("epoch")

    first_median = float(df["pairwise_median"].iloc[0])
    if first_median > 0:
        df["pairwise_median_ratio_to_start"] = df["pairwise_median"] / first_median
    else:
        df["pairwise_median_ratio_to_start"] = np.nan

    first_radius = float(df["radius_mean"].iloc[0])
    if first_radius > 0:
        df["radius_mean_ratio_to_start"] = df["radius_mean"] / first_radius
    else:
        df["radius_mean_ratio_to_start"] = np.nan

    if output.suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif output.suffix == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError("Output must end in .parquet or .csv")

    print(f"[DONE] wrote {output}")


if __name__ == "__main__":
    main()
