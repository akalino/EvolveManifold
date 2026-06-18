#!/usr/bin/env python3
"""
Batch-compute adjacent GW distances for all available parquet checkpoint runs.

Assumes each run directory contains:

  manifest.json
  checkpoints/
    ckpt_epoch_*.parquet

Example:

  python compute_gw_all_parquet_runs.py \
    --checkpoint-root "$EVOLVE_ROOT/evolve_checkpoints/collapse_ph" \
    --output-root "$EVOLVE_ROOT/gw_outputs" \
    --n-landmarks 256
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def find_run_dirs(checkpoint_root: Path) -> list[Path]:
    """
    A run dir is any directory with a manifest.json and at least two parquet checkpoints.
    """
    run_dirs = []

    for manifest in checkpoint_root.rglob("manifest.json"):
        run_dir = manifest.parent
        ckpts = sorted((run_dir / "checkpoints").glob("ckpt_epoch_*.parquet"))

        if len(ckpts) >= 2:
            run_dirs.append(run_dir)

    return sorted(run_dirs)


def safe_output_name(run_dir: Path, checkpoint_root: Path) -> str:
    """
    Make a stable filename from the run path relative to checkpoint_root.
    """
    rel = run_dir.relative_to(checkpoint_root)
    return "__".join(rel.parts)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="Root containing parquet checkpoint run directories.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where GW parquet outputs should be written.",
    )
    parser.add_argument(
        "--worker-script",
        default="compute_subsample_gw_trajectory.py",
        help="Path to the single-run GW script.",
    )

    parser.add_argument("--n-landmarks", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=5e-1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-iter", type=int, default=1000)

    parser.add_argument(
        "--normalize",
        choices=["start_median", "global_median", "per_snapshot_median", "none"],
        default="start_median",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs that already exist.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    worker_script = Path(args.worker_script).expanduser()

    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(checkpoint_root)

    print(f"[INFO] checkpoint_root: {checkpoint_root}")
    print(f"[INFO] output_root:     {output_root}")
    print(f"[INFO] runs found:      {len(run_dirs)}")
    print(f"[INFO] n_landmarks:     {args.n_landmarks}")
    print(f"[INFO] mode:            adjacent")

    if not run_dirs:
        print("[WARN] No run directories found.")
        return

    completed = 0
    skipped = 0
    failed = 0

    for i, run_dir in enumerate(run_dirs, start=1):
        out_name = safe_output_name(run_dir, checkpoint_root)
        output_path = output_root / f"{out_name}__gw_adjacent_L{args.n_landmarks}.parquet"

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP {i}/{len(run_dirs)}] {output_path.name}")
            skipped += 1
            continue

        cmd = [
            "python",
            str(worker_script),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_path),
            "--mode",
            "adjacent",
            "--n-landmarks",
            str(args.n_landmarks),
            "--epsilon",
            str(args.epsilon),
            "--seed",
            str(args.seed),
            "--max-iter",
            str(args.max_iter),
            "--normalize",
            args.normalize,
        ]

        print(f"[RUN {i}/{len(run_dirs)}] {run_dir}")

        if args.dry_run:
            print(" ".join(cmd))
            continue

        result = subprocess.run(cmd)

        if result.returncode == 0:
            completed += 1
        else:
            failed += 1
            print(f"[FAIL] {run_dir}")

    print()
    print("[DONE]")
    print(f"  completed: {completed}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")


if __name__ == "__main__":
    main()
