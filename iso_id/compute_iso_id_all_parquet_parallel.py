#!/usr/bin/env python3

"""
Parallel launcher for IsoScore / intrinsic-dimension trajectory metrics.

This script discovers parquet checkpoint trajectory runs and launches
``compute_iso_id_trajectory.py`` once per run. It is intentionally modeled
after the GW parallel launcher pattern: subprocess workers, per-run logs,
completed-output skipping, and a parquet/csv manifest.

Example:

    python iso_id_analysis/compute_iso_id_all_parquet_runs_parallel.py \\
      --checkpoint-root "$EVOLVE_ROOT/evolve_checkpoints/collapse_ph" \\
      --output-root "$EVOLVE_ROOT/iso_id_outputs/all_cloud" \\
      --workers 8 \\
      --mle-k 20 \\
      --blas-threads 1
"""

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def find_run_dirs(checkpoint_root):
    """
    Find run directories containing parquet checkpoints.

    :param checkpoint_root: Root directory containing generated trajectories.
    :return: Sorted list of run directories.
    """
    checkpoint_root = Path(checkpoint_root)

    run_dirs = set()

    for manifest in checkpoint_root.rglob("manifest.json"):
        run_dir = manifest.parent
        if has_parquet_checkpoints(run_dir):
            run_dirs.add(run_dir)

    for ckpt in checkpoint_root.rglob("ckpt_epoch_*.parquet"):
        if ckpt.parent.name == "checkpoints":
            run_dirs.add(ckpt.parent.parent)
        else:
            run_dirs.add(ckpt.parent)

    return sorted(run_dirs)


def has_parquet_checkpoints(run_dir):
    """
    Check whether a run directory contains parquet checkpoints.

    :param run_dir: Candidate run directory.
    :return: ``True`` if checkpoint parquet files are found.
    """
    run_dir = Path(run_dir)

    if list((run_dir / "checkpoints").glob("ckpt_epoch_*.parquet")):
        return True

    if list(run_dir.glob("ckpt_epoch_*.parquet")):
        return True

    return False


def safe_output_stem(run_dir, checkpoint_root):
    """
    Convert a run directory path into a stable output filename stem.

    :param run_dir: Run directory.
    :param checkpoint_root: Root checkpoint directory.
    :return: Safe filename stem.
    """
    run_dir = Path(run_dir)
    checkpoint_root = Path(checkpoint_root)

    try:
        rel = run_dir.relative_to(checkpoint_root)
        parts = rel.parts
    except ValueError:
        parts = run_dir.parts[-4:]

    stem = "__".join(parts)
    stem = stem.replace(" ", "_")
    stem = stem.replace("/", "__")
    return stem


def build_jobs(args):
    """
    Build pending worker jobs.

    :param args: Parsed CLI arguments.
    :return: Tuple ``(jobs, skipped_rows)``.
    """
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    run_dirs = find_run_dirs(checkpoint_root)
    jobs = []
    skipped = []

    for run_dir in run_dirs:
        stem = safe_output_stem(run_dir, checkpoint_root)
        output = output_root / f"{stem}__iso_id.parquet"

        row = {
            "run_dir": str(run_dir),
            "output": str(output),
            "status": "pending",
        }

        if output.exists() and not args.overwrite:
            row["status"] = "skipped_exists"
            skipped.append(row)
            continue

        jobs.append({
            "run_dir": str(run_dir),
            "output": str(output),
            "stem": stem,
        })

    return jobs, skipped


def run_one_job(job, args):
    """
    Run one trajectory worker as a subprocess.

    :param job: Job dictionary.
    :param args: Parsed CLI arguments.
    :return: Manifest row dictionary.
    """
    output_root = Path(args.output_root).expanduser().resolve()
    log_dir = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / f"{job['stem']}.stdout.txt"
    stderr_path = log_dir / f"{job['stem']}.stderr.txt"

    cmd = [
        "python",
        args.worker_script,
        "--run-dir",
        job["run_dir"],
        "--output",
        job["output"],
        "--mle-k",
        str(args.mle_k),
    ]

    if args.overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()

    if args.blas_threads is not None:
        blas_threads = str(args.blas_threads)
        env["OMP_NUM_THREADS"] = blas_threads
        env["OPENBLAS_NUM_THREADS"] = blas_threads
        env["MKL_NUM_THREADS"] = blas_threads
        env["VECLIB_MAXIMUM_THREADS"] = blas_threads
        env["NUMEXPR_NUM_THREADS"] = blas_threads

    start = time.time()

    with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_f:
        proc = subprocess.run(
            cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
        )

    elapsed = time.time() - start

    status = "completed" if proc.returncode == 0 else "failed"

    return {
        "run_dir": job["run_dir"],
        "output": job["output"],
        "status": status,
        "returncode": int(proc.returncode),
        "elapsed_sec": float(elapsed),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "worker_script": args.worker_script,
        "mle_k": int(args.mle_k),
    }


def write_manifest(rows, manifest_path):
    """
    Write status manifest as parquet or csv.

    :param rows: Manifest rows.
    :param manifest_path: Output manifest path.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    if manifest_path.suffix == ".csv":
        df.to_csv(manifest_path, index=False)
    else:
        df.to_parquet(manifest_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--worker-script",
        default="compute_iso_id_trajectory.py",
    )
    parser.add_argument("--mle-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest-name",
        default="iso_id_job_manifest.parquet",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / args.manifest_name

    jobs, skipped = build_jobs(args)

    print(f"[INFO] checkpoint_root: {Path(args.checkpoint_root).expanduser().resolve()}")
    print(f"[INFO] output_root: {output_root}")
    print(f"[INFO] worker_script: {args.worker_script}")
    print(f"[INFO] pending jobs: {len(jobs)}")
    print(f"[INFO] skipped existing: {len(skipped)}")
    print(f"[INFO] workers: {args.workers}")
    print(f"[INFO] blas_threads: {args.blas_threads}")

    if args.dry_run:
        rows = []

        for job in jobs:
            cmd = [
                "python",
                args.worker_script,
                "--run-dir",
                job["run_dir"],
                "--output",
                job["output"],
                "--mle-k",
                str(args.mle_k),
            ]
            if args.overwrite:
                cmd.append("--overwrite")

            print(" ".join(cmd))

            rows.append({
                "run_dir": job["run_dir"],
                "output": job["output"],
                "status": "dry_run",
                "returncode": None,
                "elapsed_sec": 0.0,
                "stdout_log": None,
                "stderr_log": None,
                "worker_script": args.worker_script,
                "mle_k": int(args.mle_k),
            })

        rows.extend(skipped)
        write_manifest(rows, manifest_path)

        print(f"[DONE] wrote dry-run manifest -> {manifest_path}")
        return

    results = []

    if skipped:
        results.extend(skipped)

    if not jobs:
        write_manifest(results, manifest_path)
        print(f"[DONE] no pending jobs; wrote manifest -> {manifest_path}")
        return

    workers = max(1, int(args.workers))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(run_one_job, job, args): job
            for job in jobs
        }

        for i, future in enumerate(as_completed(future_map), start=1):
            job = future_map[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "run_dir": job["run_dir"],
                    "output": job["output"],
                    "status": "launcher_failed",
                    "returncode": None,
                    "elapsed_sec": 0.0,
                    "stdout_log": None,
                    "stderr_log": None,
                    "worker_script": args.worker_script,
                    "mle_k": int(args.mle_k),
                    "error": repr(exc),
                }

            results.append(result)

            print(
                f"[{i}/{len(jobs)}] {result['status']} "
                f"{Path(result['run_dir']).name}"
            )

            # Keep the manifest useful even if the machine dies mid-run.
            write_manifest(results, manifest_path)

    write_manifest(results, manifest_path)

    df = pd.DataFrame(results)

    print(f"[DONE] wrote manifest -> {manifest_path}")

    if len(df) > 0 and "status" in df.columns:
        print(df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
