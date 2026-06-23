"""
Generate GW calibration jobs for representative runs.

This script writes a shell script rather than running the jobs directly. The
resulting script can be inspected, edited, and submitted locally or on a
larger machine.

Example
-------
python calibrate_gw_settings.py \\
    --run-dirs run_a run_b run_c \\
    --output-root "$EVOLVE_ROOT/gw_calibration" \\
    --job-script "$EVOLVE_ROOT/gw_calibration/run_gw_calibration.sh"
"""

import argparse
from pathlib import Path


def format_float(value):
    """
    Format a float for use in filenames.

    :param value: Numeric value.
    :return: Filename-safe string.
    """
    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    return text


def build_output_path(output_root, run_dir, method, n_landmarks, epsilon, seed, norm):
    """
    Build the output path for one calibration job.

    :param output_root: Calibration output root.
    :param run_dir: Input run directory.
    :param method: OT method name.
    :param n_landmarks: Number of landmarks.
    :param epsilon: Entropic regularization.
    :param seed: Landmark seed.
    :param norm: Normalization mode.
    :return: Output parquet path.
    """
    run_name = run_dir.name
    eps_name = format_float(epsilon)

    name = (
        f"{run_name}__{method}"
        f"__L{n_landmarks}"
        f"__eps{eps_name}"
        f"__seed{seed}"
        f"__norm{norm}.parquet"
    )

    return output_root / name


def build_command(args, run_dir, output_path, method, n_landmarks, epsilon, seed, norm):
    """
    Build one worker command.

    :param args: Parsed CLI arguments.
    :param run_dir: Input run directory.
    :param output_path: Output parquet path.
    :param method: OT method name.
    :param n_landmarks: Number of landmarks.
    :param epsilon: Entropic regularization.
    :param seed: Landmark seed.
    :param norm: Normalization mode.
    :return: Shell command string.
    """
    cmd = [
        "python",
        args.worker_script,
        "--run-dir",
        str(run_dir),
        "--output",
        str(output_path),
        "--mode",
        args.mode,
        "--n-landmarks",
        str(n_landmarks),
        "--epsilon",
        str(epsilon),
        "--seed",
        str(seed),
        "--normalize",
        norm,
        "--ot-method",
        method,
    ]

    if args.max_iter is not None:
        cmd.extend(["--max-iter", str(args.max_iter)])

    if args.tol is not None:
        cmd.extend(["--tol", str(args.tol)])

    return " ".join(cmd)


def main():
    """
    Run the GW calibration job-script generator.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--worker-script", default="compute_subsample_gw_trajectory.py")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--job-script", required=True)

    parser.add_argument("--mode", default="adjacent")
    parser.add_argument("--landmarks", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 0.5, 0.1, 0.05])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument(
        "--normalizations",
        nargs="+",
        default=["start_median", "none", "per_snapshot_median"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["entropic_gw"],
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)

    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    job_script = Path(args.job_script).expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    job_script.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]

    job_count = 0

    for raw_run_dir in args.run_dirs:
        run_dir = Path(raw_run_dir).expanduser().resolve()

        for n_landmarks in args.landmarks:
            for epsilon in args.epsilons:
                for seed in args.seeds:
                    for norm in args.normalizations:
                        for method in args.methods:
                            output_path = build_output_path(
                                output_root,
                                run_dir,
                                method,
                                n_landmarks,
                                epsilon,
                                seed,
                                norm,
                            )

                            cmd = build_command(
                                args,
                                run_dir,
                                output_path,
                                method,
                                n_landmarks,
                                epsilon,
                                seed,
                                norm,
                            )

                            lines.append(cmd)
                            job_count += 1

    job_script.write_text("\n".join(lines) + "\n")
    job_script.chmod(0o755)

    print(f"[DONE] wrote {job_script}")
    print(f"[DONE] jobs={job_count}")


if __name__ == "__main__":
    main()
