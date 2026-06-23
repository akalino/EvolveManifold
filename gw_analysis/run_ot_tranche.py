#!/usr/bin/env python3
"""
Run an OT tracing tranche from a YAML config.

The config expands over selected OT parameters and dispatches one subprocess
per run/parameter combination. This is intended to replace long command-line
calls with reusable tranche files.

Example
-------
python run_ot_tranche.py --config configs/ot_tracing/ot_primary_adjacent.yaml
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml


CKPT_PARQUET_RE = re.compile(r"ckpt_epoch_(\d+)\.parquet$")


def expand_env(value):
    """
    Expand shell-style environment variables and user home markers.

    :param value: Raw config value.
    :return: Expanded value.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))

    return value


def as_list(value):
    """
    Convert a scalar or list-like config value into a list.

    :param value: Config value.
    :return: List of values.
    """
    if isinstance(value, list):
        return value
    return [value]


def read_json_if_exists(path):
    """
    Read a JSON file if it exists.

    :param path: JSON path.
    :return: Dictionary payload or empty dictionary.
    """
    path = Path(path)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def find_run_dirs(checkpoint_root):
    """
    Find run directories containing a manifest and at least two parquet checkpoints.

    :param checkpoint_root: Root containing checkpoint run directories.
    :return: Sorted list of run directories.
    """
    run_dirs = []

    for manifest in checkpoint_root.rglob("manifest.json"):
        run_dir = manifest.parent
        ckpt_dir = run_dir / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("ckpt_epoch_*.parquet"))

        if len(ckpts) >= 2:
            run_dirs.append(run_dir)

    return sorted(run_dirs)


def parse_run_metadata(run_dir):
    """
    Extract lightweight metadata for filtering.

    :param run_dir: Run directory.
    :return: Metadata dictionary.
    """
    manifest = read_json_if_exists(run_dir / "manifest.json")
    metadata = read_json_if_exists(run_dir / "metadata.json")

    run_name = run_dir.name
    model = manifest.get("model") or manifest.get("run_id") or run_name

    out = {
        "run_id": manifest.get("run_id") or model,
        "model": model,
        "mechanism": manifest.get("mechanism") or metadata.get("mechanism"),
        "geometry": metadata.get("base_geometry") or metadata.get("geometry"),
        "schedule": metadata.get("schedule"),
        "severity": metadata.get("severity"),
        "noise": metadata.get("noise"),
    }

    if out["mechanism"] is None:
        parent_name = run_dir.parent.name
        out["mechanism"] = parent_name

    if out["geometry"] is None:
        match = re.match(r"(.+)_n\d+_d\d+_k\d+__", model)
        if match:
            out["geometry"] = match.group(1)

    if out["schedule"] is None:
        for candidate in ["linear", "exponential", "sigmoid"]:
            if f"__{candidate}__" in model:
                out["schedule"] = candidate

    if out["severity"] is None:
        for candidate in ["weak", "moderate", "strong"]:
            if f"__{candidate}__" in model:
                out["severity"] = candidate

    if out["noise"] is None:
        match = re.search(r"noise([0-9.]+)", model)
        if match:
            try:
                out["noise"] = float(match.group(1))
            except Exception:
                out["noise"] = None

    return out


def normalize_filter_value(value):
    """
    Normalize values for loose config filtering.

    :param value: Raw value.
    :return: Normalized string.
    """
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:g}"

    return str(value)


def passes_filter(meta, filters):
    """
    Decide whether a run passes tranche filters.

    Empty filter lists mean no restriction.

    :param meta: Run metadata.
    :param filters: Filter dictionary.
    :return: True if the run should be included.
    """
    key_map = {
        "mechanisms": "mechanism",
        "geometries": "geometry",
        "schedules": "schedule",
        "severities": "severity",
        "noises": "noise",
    }

    for filter_key, meta_key in key_map.items():
        allowed = filters.get(filter_key, []) or []

        if not allowed:
            continue

        allowed_norm = {normalize_filter_value(x) for x in allowed}
        value_norm = normalize_filter_value(meta.get(meta_key))

        if value_norm not in allowed_norm:
            return False

    return True


def safe_output_stem(run_dir, checkpoint_root):
    """
    Build a stable output stem from a run path.

    :param run_dir: Run directory.
    :param checkpoint_root: Checkpoint root.
    :return: Safe output stem.
    """
    rel = run_dir.relative_to(checkpoint_root)
    return "__".join(rel.parts)


def param_grid(config):
    """
    Expand OT parameter values into a list of parameter dictionaries.

    :param config: Tranche config.
    :return: List of parameter dictionaries.
    """
    keys = [
        "mode",
        "ot_method",
        "landmark_method",
        "n_landmarks",
        "epsilon",
        "seed",
        "normalize",
    ]

    values = [as_list(config.get(key)) for key in keys]

    rows = []

    for combo in itertools.product(*values):
        row = {}
        for key, value in zip(keys, combo):
            row[key] = value

        row["max_iter"] = config.get("max_iter", 1000)
        row["tol"] = config.get("tol", 1.0e-6)

        rows.append(row)

    return rows


def format_float(value):
    """
    Format a float for filenames.

    :param value: Raw value.
    :return: Filename-safe string.
    """
    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    return text


def output_path_for_job(output_root, run_stem, params):
    """
    Build output path for one job.

    :param output_root: Output root.
    :param run_stem: Run filename stem.
    :param params: Parameter dictionary.
    :return: Output path.
    """
    eps_name = format_float(params["epsilon"])

    name = (
        f"{run_stem}"
        f"__{params['ot_method']}"
        f"__{params['mode']}"
        f"__{params['landmark_method']}"
        f"__L{params['n_landmarks']}"
        f"__eps{eps_name}"
        f"__seed{params['seed']}"
        f"__norm{params['normalize']}.parquet"
    )

    return output_root / name


def build_command(config, run_dir, output_path, params):
    """
    Build the worker subprocess command.

    :param config: Tranche config.
    :param run_dir: Run directory.
    :param output_path: Output path.
    :param params: Parameter dictionary.
    :return: Command list.
    """
    cmd = [
        "python",
        str(config["worker_script"]),
        "--run-dir",
        str(run_dir),
        "--output",
        str(output_path),
        "--mode",
        str(params["mode"]),
        "--n-landmarks",
        str(params["n_landmarks"]),
        "--landmark-method",
        str(params["landmark_method"]),
        "--ot-method",
        str(params["ot_method"]),
        "--epsilon",
        str(params["epsilon"]),
        "--seed",
        str(params["seed"]),
        "--normalize",
        str(params["normalize"]),
        "--max-iter",
        str(params["max_iter"]),
        "--tol",
        str(params["tol"]),
    ]

    return cmd


def build_jobs(config):
    """
    Build all tranche jobs.

    :param config: Tranche config.
    :return: Pair ``(run_dirs, jobs)``.
    """
    checkpoint_root = Path(expand_env(config["checkpoint_root"])).resolve()
    output_root = Path(expand_env(config["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logs_root = output_root / "_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    config["worker_script"] = expand_env(config["worker_script"])

    run_dirs = find_run_dirs(checkpoint_root)
    filters = config.get("filters", {}) or {}

    selected = []

    for run_dir in run_dirs:
        meta = parse_run_metadata(run_dir)
        if passes_filter(meta, filters):
            selected.append((run_dir, meta))

    limit_runs = config.get("limit_runs")
    if limit_runs is not None:
        selected = selected[:int(limit_runs)]

    params_rows = param_grid(config)

    jobs = []

    for run_dir, meta in selected:
        run_stem = safe_output_stem(run_dir, checkpoint_root)

        for params in params_rows:
            output_path = output_path_for_job(output_root, run_stem, params)

            if output_path.exists() and not bool(config.get("overwrite", False)):
                jobs.append(
                    {
                        "run_dir": str(run_dir),
                        "output_path": str(output_path),
                        "cmd": None,
                        "status": "skipped_exists",
                        "returncode": None,
                        "runtime_sec": 0.0,
                        "stdout_path": None,
                        "stderr_path": None,
                        **meta,
                        **params,
                    }
                )
                continue

            log_stem = output_path.stem

            job = {
                "run_dir": str(run_dir),
                "output_path": str(output_path),
                "cmd": build_command(config, run_dir, output_path, params),
                "status": "pending",
                "returncode": None,
                "runtime_sec": None,
                "stdout_path": str(logs_root / f"{log_stem}.stdout.txt"),
                "stderr_path": str(logs_root / f"{log_stem}.stderr.txt"),
                **meta,
                **params,
            }

            jobs.append(job)

    return run_dirs, selected, jobs


def run_one_job(job, blas_threads):
    """
    Run one worker job.

    :param job: Job dictionary.
    :param blas_threads: BLAS thread count for subprocess.
    :return: Result dictionary.
    """
    if job["cmd"] is None:
        return job

    start = time.perf_counter()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(blas_threads)
    env["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    env["MKL_NUM_THREADS"] = str(blas_threads)
    env["NUMEXPR_NUM_THREADS"] = str(blas_threads)

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
    Write a tranche manifest.

    :param rows: Manifest rows.
    :param manifest_path: Output path.
    """
    df = pd.DataFrame(rows)

    if "cmd" in df.columns:
        df["cmd_str"] = df["cmd"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else ""
        )

    if manifest_path.suffix == ".csv":
        df.to_csv(manifest_path, index=False)
    else:
        df.to_parquet(manifest_path, index=False)


def load_config(path):
    """
    Load a YAML tranche config.

    :param path: Config path.
    :return: Config dictionary.
    """
    path = Path(path).expanduser().resolve()

    with path.open("r") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        raise ValueError(f"Empty config: {path}")

    config["_config_path"] = str(path)
    return config


def main():
    """
    Run the OT tranche launcher.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    output_root = Path(expand_env(config["output_root"])).resolve()
    manifest_path = output_root / f"{config.get('name', 'ot_tranche')}__manifest.parquet"

    run_dirs, selected, jobs = build_jobs(config)

    pending = [job for job in jobs if job["status"] == "pending"]
    skipped = [job for job in jobs if job["status"] == "skipped_exists"]

    print(f"[TRANCHE] {config.get('name')}")
    print(f"[INFO] config:        {config['_config_path']}")
    print(f"[INFO] output_root:   {output_root}")
    print(f"[INFO] runs found:    {len(run_dirs)}")
    print(f"[INFO] runs selected: {len(selected)}")
    print(f"[INFO] jobs total:    {len(jobs)}")
    print(f"[INFO] jobs pending:  {len(pending)}")
    print(f"[INFO] jobs skipped:  {len(skipped)}")
    print(f"[INFO] workers:       {config.get('workers', 2)}")

    if args.dry_run:
        for job in pending[:20]:
            print(" ".join(job["cmd"]))

        if len(pending) > 20:
            print(f"[INFO] showing first 20 of {len(pending)} pending commands")

        write_manifest(jobs, manifest_path)
        print(f"[DONE] wrote dry-run manifest {manifest_path}")
        return

    results = list(skipped)

    workers = int(config.get("workers", 2))
    blas_threads = int(config.get("blas_threads", 1))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}

        for job in pending:
            future = pool.submit(run_one_job, job, blas_threads)
            future_map[future] = job

        for i, future in enumerate(as_completed(future_map), start=1):
            result = future.result()
            results.append(result)

            print(
                f"[{i}/{len(pending)}] {result['status']} "
                f"{Path(result['run_dir']).name}"
            )

    write_manifest(results, manifest_path)

    df = pd.DataFrame(results)

    print(f"[DONE] wrote {manifest_path}")

    if len(df) > 0:
        print(df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
