"""
Parallel batch-compute GW distances for all parquet checkpoint runs.

This is a parallel version of the single-process batch launcher. Each run is
delegated to the single-run worker script in a subprocess. A job manifest is
written at the end with status, runtime, return code, and log paths.

Example
-------
python compute_gw_all_parquet_runs_parallel.py \\
    --checkpoint-root "$EVOLVE_ROOT/evolve_checkpoints/collapse_ph" \\
    --output-root "$EVOLVE_ROOT/gw_outputs" \\
    --workers 4 \\
    --n-landmarks 256
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
    Find all run directories below a checkpoint root.

    A run directory is defined as a directory containing ``manifest.json`` and
    at least two parquet checkpoint files under ``checkpoints/``.

    :param checkpoint_root: Root directory containing runs.
    :return: Sorted list of run directories.
    """
    run_dirs = []

    for manifest in checkpoint_root.rglob("manifest.json"):
        run_dir = manifest.parent
        ckpts = sorted((run_dir / "checkpoints").glob("ckpt_epoch_*.parquet"))

        if len(ckpts) >= 2:
            run_dirs.append(run_dir)

    return sorted(run_dirs)


def safe_output_name(run_dir, checkpoint_root):
    """
    Build a stable output filename stem from a run directory.

    :param run_dir: Run directory.
    :param checkpoint_root: Root used to compute the relative path.
    :return: Filename-safe output stem.
    """
    rel = run_dir.relative_to(checkpoint_root)
    return "__".join(rel.parts)


def build_command(args, run_dir, output_path):
    """
    Build the subprocess command for one GW worker job.

    :param args: Parsed CLI arguments.
    :param run_dir: Run directory.
    :param output_path: Output parquet path.
    :return: Command list for ``subprocess.run``.
    """
    cmd = [
        "python",
        str(Path(args.worker_script).expanduser()),
        "--run-dir",
        str(run_dir),
        "--output",
        str(output_path),
        "--mode",
        args.mode,
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

    if args.tol is not None:
        cmd.extend(["--tol", str(args.tol)])

    if args.ot_method is not None:
        cmd.extend(["--ot-method", args.ot_method])

    if args.landmark_method is not None:
        cmd.extend(["--landmark-method", args.landmark_method])

    return cmd


def build_jobs(args, checkpoint_root, output_root, logs_root):
    """
    Build all batch jobs.

    :param args: Parsed CLI arguments.
    :param checkpoint_root: Root containing checkpoint run directories.
    :param output_root: Root for GW outputs.
    :param logs_root: Root for stdout and stderr logs.
    :return: Pair ``(run_dirs, jobs)``.
    """
    run_dirs = find_run_dirs(checkpoint_root)
    jobs = []

    for run_dir in run_dirs:
        out_name = safe_output_name(run_dir, checkpoint_root)

        output_name = (
            f"{out_name}"
            f"__{args.ot_method}"
            f"__{args.mode}"
            f"__L{args.n_landmarks}.parquet"
        )

        output_path = output_root / output_name

        if output_path.exists() and not args.overwrite:
            continue

        cmd = build_command(args, run_dir, output_path)

        jobs.append(
            {
                "run_dir": str(run_dir),
                "output_path": str(output_path),
                "cmd": cmd,
                "stdout_path": str(logs_root / f"{out_name}.stdout.txt"),
                "stderr_path": str(logs_root / f"{out_name}.stderr.txt"),
                "blas_threads": int(args.blas_threads),
            }
        )

    return run_dirs, jobs


def run_one_job(job):
    """
    Run one subprocess job.

    BLAS thread counts are constrained per worker to avoid oversubscribing CPU
    resources when many jobs are launched in parallel.

    :param job: Job dictionary.
    :return: Job result dictionary.
    """
    start = time.perf_counter()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(job["blas_threads"])
    env["OPENBLAS_NUM_THREADS"] = str(job["blas_threads"])
    env["MKL_NUM_THREADS"] = str(job["blas_threads"])
    env["NUMEXPR_NUM_THREADS"] = str(job["blas_threads"])

    stdout_path = Path(job["stdout_path"])
    stderr_path = Path(job["stderr_path"])

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        result = subprocess.run(
            job["cmd"],
            stdout=out,
            stderr=err,
            env=env,
            check=False,
        )

    elapsed = time.perf_counter() - start

    status = "completed"
    if result.returncode != 0:
        status = "failed"

    return {
        **job,
        "returncode": int(result.returncode),
        "runtime_sec": float(elapsed),
        "status": status,
    }


def write_manifest(rows, manifest_path):
    """
    Write a batch job manifest.

    :param rows: List of job result dictionaries.
    :param manifest_path: Output manifest path.
    """
    df = pd.DataFrame(rows)

    if manifest_path.suffix == ".parquet":
        df.to_parquet(manifest_path, index=False)
    elif manifest_path.suffix == ".csv":
        df.to_csv(manifest_path, index=False)
    else:
        raise ValueError("Manifest path must end in .parquet or .csv")


def main():
    """
    Run the parallel GW batch launcher.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--worker-script", default="compute_subsample_gw_trajectory.py")

    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--blas-threads", type=int, default=1)

    parser.add_argument("--mode", default="adjacent")
    parser.add_argument("--n-landmarks", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--normalize", default="start_median")

    parser.add_argument("--ot-method", default="entropic_gw")
    parser.add_argument("--landmark-method", default=None)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--manifest-name",
        default="gw_job_manifest.parquet",
        help="Filename for the job manifest.",
    )

    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    logs_root = output_root / "_logs"

    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    run_dirs, jobs = build_jobs(args, checkpoint_root, output_root, logs_root)

    manifest_path = output_root / args.manifest_name

    print(f"[INFO] runs found: {len(run_dirs)}")
    print(f"[INFO] jobs to run: {len(jobs)}")
    print(f"[INFO] workers: {args.workers}")
    print(f"[INFO] blas_threads: {args.blas_threads}")

    if args.dry_run:
        for job in jobs:
            print(" ".join(job["cmd"]))

        dry_rows = []
        for job in jobs:
            dry_rows.append(
                {
                    **job,
                    "returncode": None,
                    "runtime_sec": None,
                    "status": "dry_run",
                }
            )

        write_manifest(dry_rows, manifest_path)
        print(f"[DONE] wrote dry-run manifest {manifest_path}")
        return

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {}

        for job in jobs:
            future = pool.submit(run_one_job, job)
            future_map[future] = job

        for i, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            results.append(result)

            run_name = Path(result["run_dir"]).name
            print(f"[{i}/{len(jobs)}] {result['status']} {run_name}")

    write_manifest(results, manifest_path)

    result_df = pd.DataFrame(results)

    print(f"[DONE] wrote {manifest_path}")

    if "status" in result_df.columns and len(result_df) > 0:
        print(result_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
