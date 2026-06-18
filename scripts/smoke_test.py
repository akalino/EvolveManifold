"""Runs a minimal end-to-end smoke test for EvolveManifold.

This script validates the workflow:

1. Generate parquet checkpoint trajectories from ``configs/tranches/smoke.json``.
2. Measure those checkpoints with a lightweight PH workflow.
3. Confirm that manifests, checkpoint parquet files, measurement status, and
   combined metric outputs were written.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tranches" / "smoke.json"


def run_command(cmd, env):
    """Run a subprocess command from the repository root.

    :param cmd: Command and arguments to execute.
    :param env: Environment variables for the child process.
    :raises SystemExit: If the command exits with a nonzero status.
    """
    print("\n[smoke] running:", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )

    if result.returncode != 0:
        raise SystemExit(
            "[smoke] command failed with exit code {}: {}".format(
                result.returncode,
                " ".join(cmd),
            )
        )


def assert_exists(path, label):
    """Require that a file or directory exists.

    :param path: Path to check.
    :param label: Human-readable description of the expected artifact.
    :raises AssertionError: If the path does not exist.
    """
    if not path.exists():
        raise AssertionError("[smoke] missing {}: {}".format(label, path))

    print("[smoke] found {}: {}".format(label, path))


def parse_args():
    """Parse command-line arguments.

    :returns: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run EvolveManifold smoke test."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to smoke tranche config.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Optional persistent work directory. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep temporary smoke-test outputs instead of deleting them.",
    )
    parser.add_argument(
        "--ph-mode",
        default="landmark_vr",
        choices=[
            "full_vr",
            "landmark_vr",
            "skip_vr",
            "fixed_support_vr",
            "fixed_knn_vr",
            "event_driven",
            "online_landmark_event",
            "online_landmark_dynamic_support",
        ],
        help="PH workflow mode to use for measurement.",
    )
    return parser.parse_args()


def main():
    """Run the smoke test.

    :raises SystemExit: If the config file does not exist or a subprocess
        command fails.
    :raises AssertionError: If expected smoke-test artifacts are missing or
        malformed.
    """
    args = parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit("[smoke] config does not exist: {}".format(config_path))

    created_temp = args.workdir is None
    if args.workdir:
        base_root = Path(args.workdir).resolve()
    else:
        base_root = Path(tempfile.mkdtemp(prefix="evolve_smoke_"))

    checkpoint_root = base_root / "evolve_checkpoints"
    metric_root = base_root / "metric_outputs"

    env = os.environ.copy()
    env["EVOLVE_COLLAPSE_ROOT"] = str(base_root)
    env["EVOLVE_ROOT"] = str(base_root)

    try:
        print("[smoke] repo root: {}".format(REPO_ROOT))
        print("[smoke] config: {}".format(config_path))
        print("[smoke] base root: {}".format(base_root))

        run_command(
            [
                sys.executable,
                "run_cloud_manifest.py",
                "--config",
                str(config_path),
                "--root-dir",
                str(checkpoint_root),
            ],
            env=env,
        )

        manifests = sorted(checkpoint_root.rglob("manifest.json"))
        if not manifests:
            raise AssertionError(
                "[smoke] no run manifests found under {}".format(checkpoint_root)
            )

        parquet_checkpoints = sorted(checkpoint_root.rglob("ckpt_epoch_*.parquet"))
        if not parquet_checkpoints:
            raise AssertionError(
                "[smoke] no parquet checkpoints found under {}".format(
                    checkpoint_root
                )
            )

        print("[smoke] manifests: {}".format(len(manifests)))
        print("[smoke] checkpoint parquet files: {}".format(len(parquet_checkpoints)))

        run_command(
            [
                sys.executable,
                "run_measurement_tranched.py",
                "--config",
                str(config_path),
                "--tranche",
                "smoke",
                "--root-dir",
                str(checkpoint_root),
                "--out-dir",
                str(metric_root),
                "--ph-mode",
                args.ph_mode,
                "--workers",
                "1",
                "--no-csv",
            ],
            env=env,
        )

        status_path = metric_root / args.ph_mode / "_status" / "measurement_status.csv"
        combined_path = metric_root / args.ph_mode / "_combined" / "all_metrics.parquet"

        assert_exists(status_path, "measurement status CSV")
        assert_exists(combined_path, "combined metrics parquet")

        df_status = pd.read_csv(status_path)
        if df_status.empty:
            raise AssertionError("[smoke] measurement status table is empty")

        if "status" not in df_status.columns:
            raise AssertionError(
                "[smoke] measurement status table has no 'status' column"
            )

        failed = df_status[df_status["status"] == "failed"]
        if len(failed):
            raise AssertionError(
                "[smoke] one or more measurements failed:\n{}".format(failed)
            )

        df_metrics = pd.read_parquet(combined_path)
        if df_metrics.empty:
            raise AssertionError("[smoke] combined metrics table is empty")

        required_columns = {
            "epoch",
            "geometry",
            "mechanism",
            "effective_rank",
            "ph_mode",
            "max_persistence_h1",
            "betti_curve_area_h1",
        }
        missing = sorted(required_columns - set(df_metrics.columns))
        if missing:
            raise AssertionError(
                "[smoke] combined metrics missing columns: {}".format(missing)
            )

        print("\n[smoke] PASS")
        print("[smoke] measured rows: {}".format(len(df_metrics)))
        print("[smoke] output root: {}".format(base_root))

    finally:
        if created_temp and not args.keep:
            shutil.rmtree(base_root, ignore_errors=True)
            print("[smoke] removed temporary output root: {}".format(base_root))


if __name__ == "__main__":
    main()
