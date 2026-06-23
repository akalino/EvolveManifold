"""
Batch-compute IsoScore and intrinsic-dimension metrics for parquet runs.

Example:

    python iso_id_analysis/compute_iso_id_all_parquet_runs.py \
      --checkpoint-root "$EVOLVE_ROOT/checkpoints" \
      --output-root "$EVOLVE_ROOT/iso_id_outputs" \
      --mle-k 20
"""

import argparse
import subprocess
from pathlib import Path


def find_run_dirs(checkpoint_root):
    """
    Find run directories containing manifest.json and parquet checkpoints.

    :param checkpoint_root: Root checkpoint directory.
    :return: Sorted list of run directories.
    """
    run_dirs = []

    for manifest in checkpoint_root.rglob("manifest.json"):
        run_dir = manifest.parent
        ckpts = sorted((run_dir / "checkpoints").glob("ckpt_epoch_*.parquet"))
        if len(ckpts) >= 1:
            run_dirs.append(run_dir)

    return sorted(run_dirs)


def safe_output_name(run_dir, checkpoint_root):
    """
    Make a stable filename from a run path.

    :param run_dir: Run directory.
    :param checkpoint_root: Checkpoint root.
    :return: Safe output stem.
    """
    rel = run_dir.relative_to(checkpoint_root)
    return "__".join(rel.parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--worker-script",
        default="compute_iso_id_trajectory.py",
    )
    parser.add_argument("--mle-k", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(checkpoint_root)

    print(f"[INFO] checkpoint_root: {checkpoint_root}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] runs found: {len(run_dirs)}")

    completed = 0
    skipped = 0
    failed = 0

    for i, run_dir in enumerate(run_dirs, start=1):
        out_name = safe_output_name(run_dir, checkpoint_root)
        output = output_root / f"{out_name}__iso_id.parquet"

        if output.exists() and not args.overwrite:
            print(f"[SKIP {i}/{len(run_dirs)}] {output.name}")
            skipped += 1
            continue

        cmd = [
            "python",
            args.worker_script,
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--mle-k",
            str(args.mle_k),
        ]

        if args.overwrite:
            cmd.append("--overwrite")

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

    print("[DONE]")
    print(f" completed: {completed}")
    print(f" skipped: {skipped}")
    print(f" failed: {failed}")


if __name__ == "__main__":
    main()
