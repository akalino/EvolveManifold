"""
Compute IsoScore and intrinsic-dimension metrics for one parquet trajectory.

Example:

    python iso_id_analysis/compute_iso_id_trajectory.py \
      --run-dir "$EVOLVE_ROOT/checkpoints/example_run" \
      --output "$EVOLVE_ROOT/iso_id_outputs/example_run_iso_id.parquet"
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from metrics_iso_id import compute_iso_id_metrics


def load_checkpoint(path):
    """
    Load a parquet checkpoint as a numeric point cloud.

    :param path: Path to checkpoint parquet file.
    :return: ``numpy.ndarray`` point cloud.
    """
    df = pd.read_parquet(path)

    drop_cols = {
        "point_id",
        "epoch",
        "step",
        "label",
        "cluster",
        "geometry",
        "mechanism",
        "schedule",
        "severity",
        "noise",
        "seed",
    }
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].to_numpy(dtype=float)


def load_manifest_metadata(run_dir):
    """
    Load manifest metadata if available.

    :param run_dir: Run directory.
    :return: Metadata dictionary.
    """
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mle-k", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if output.exists() and not args.overwrite:
        print(f"[SKIP] {output}")
        return

    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_epoch_*.parquet"))
    if not ckpts:
        ckpts = sorted(run_dir.glob("ckpt_epoch_*.parquet"))

    if not ckpts:
        raise RuntimeError(f"No parquet checkpoints found under {run_dir}")

    meta = load_manifest_metadata(run_dir)
    rows = []

    for ckpt in ckpts:
        x = load_checkpoint(ckpt)
        row = compute_iso_id_metrics(x, mle_k=args.mle_k)
        row.update({
            "run_dir": str(run_dir),
            "checkpoint_file": ckpt.name,
            "n_points": int(x.shape[0]),
            "ambient_dim": int(x.shape[1]),
        })

        for key in [
            "geometry",
            "mechanism",
            "schedule",
            "severity",
            "mover_frac",
            "noise",
            "seed",
            "n",
            "d",
        ]:
            if key in meta:
                row[key] = meta[key]

        rows.append(row)

    out = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix == ".csv":
        out.to_csv(output, index=False)
    else:
        out.to_parquet(output, index=False)

    print(f"[DONE] wrote {len(out)} rows -> {output}")


if __name__ == "__main__":
    main()
