"""
Generate Mapper evolution animations for every run directory.

Expected layout:
    mapper_outputs/canonical_mapper_probe/
      runs/
        spiked_gaussian__linear_to_kplane__seed_5/
          mapper_summary.csv
          graphs/*.pkl
        torus__hole_fill__seed_37/
          mapper_summary.csv
          graphs/*.pkl

Usage:
    python evaluation/mapper_analysis/animate_all_mapper_runs.py \
      --config config/mapper_canonical.json
"""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_base_root(paths_cfg):
    env_name = paths_cfg.get("base_root_env")
    fallback_env_name = paths_cfg.get("base_root_fallback_env")
    default = paths_cfg.get("base_root_default", ".")

    if env_name and os.getenv(env_name):
        return Path(os.getenv(env_name)).expanduser().resolve()

    if fallback_env_name and os.getenv(fallback_env_name):
        return Path(os.getenv(fallback_env_name)).expanduser().resolve()

    return Path(default).expanduser().resolve()


def resolve_runs_root(cfg):
    paths_cfg = cfg.get("paths", {})
    vis_cfg = cfg.get("visualization", {})

    runs_root = vis_cfg.get("runs_root")

    if runs_root:
        runs_root = Path(runs_root).expanduser()
        if not runs_root.is_absolute():
            runs_root = resolve_base_root(paths_cfg) / runs_root
        return runs_root.resolve()

    base_root = resolve_base_root(paths_cfg)
    mapper_subdir = paths_cfg.get("mapper_output_subdir", "mapper_outputs")
    return (base_root / mapper_subdir / "runs").resolve()


def write_temp_run_config(cfg, run_dir, out_name):
    cfg2 = dict(cfg)
    cfg2["visualization"] = dict(cfg.get("visualization", {}))
    cfg2["visualization"]["mapper_dir"] = str(run_dir)
    cfg2["visualization"]["out"] = out_name

    temp_path = run_dir / "_viz_config.json"
    with open(temp_path, "w") as f:
        json.dump(cfg2, f, indent=2)

    return temp_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--animate-script",
        default="animate_mapper_graphs.py",
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    vis_cfg = cfg.get("visualization", {})

    runs_root = resolve_runs_root(cfg)
    out_name = vis_cfg.get("out_name", "mapper_evolution.gif")

    if not runs_root.exists():
        raise FileNotFoundError(f"Missing runs root: {runs_root}")

    run_dirs = sorted(
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "mapper_summary.csv").exists()
    )

    if not run_dirs:
        raise FileNotFoundError(f"No run dirs with mapper_summary.csv found in {runs_root}")

    print(f"Runs root: {runs_root}")
    print(f"Found {len(run_dirs)} runs")

    for i, run_dir in enumerate(run_dirs, start=1):
        print(f"\n[{i}/{len(run_dirs)}] Animating {run_dir.name}")

        temp_config = write_temp_run_config(cfg, run_dir, out_name)

        cmd = [
            sys.executable,
            args.animate_script,
            "--config",
            str(temp_config),
        ]

        subprocess.run(cmd, check=True)

        print(f"Wrote: {run_dir / out_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
