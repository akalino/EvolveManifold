"""
Plot GW trajectory curves for one or more GW output files.

This script supports adjacent, from-start, and generic GW outputs. It is
intended as a quick plotting utility rather than a publication figure script.

Example
-------
python plot_gw_trajectory.py \\
    --gw-file "$EVOLVE_ROOT/gw_outputs/example__gw_adjacent_L256.parquet" \\
    --output-dir "$EVOLVE_ROOT/gw_outputs/plots"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def safe_name(value):
    """
    Convert a string into a filesystem-safe name.

    :param value: Raw string.
    :return: Safe string for filenames.
    """
    out = str(value)
    out = out.replace("/", "_")
    out = out.replace("\\", "_")
    out = out.replace(" ", "_")
    out = out.replace(":", "_")

    while "__" in out:
        out = out.replace("__", "_")

    return out


def plot_one(df, output_dir, title_prefix):
    """
    Plot one GW trajectory curve.

    :param df: GW output dataframe for one comparison type.
    :param output_dir: Directory where the plot should be written.
    :param title_prefix: Optional title prefix.
    """
    df = df.sort_values(["source_epoch", "target_epoch"])

    if "comparison_type" in df.columns:
        comparison_type = str(df["comparison_type"].iloc[0])
    else:
        comparison_type = "gw"

    if "run_id" in df.columns:
        run_id = str(df["run_id"].iloc[0])
    elif "model" in df.columns:
        run_id = str(df["model"].iloc[0])
    else:
        run_id = "run"

    if comparison_type == "adjacent":
        x = df["target_epoch"]
        y = df["gw_distance"]
        ylabel = "Adjacent GW distance"

    elif comparison_type == "from_start":
        x = df["target_epoch"]
        y = df["gw_distance"]
        ylabel = "GW distance from start"

    else:
        x = range(len(df))
        y = df["gw_distance"]
        ylabel = "GW distance"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}{comparison_type}: {run_id}")
    ax.grid(True, alpha=0.3)

    out = output_dir / f"{safe_name(run_id)}__{comparison_type}__gw_curve.png"

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"[PLOT] {out}")


def main():
    """
    Run the GW trajectory plotting CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title-prefix", default="")
    args = parser.parse_args()

    gw_file = Path(args.gw_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(gw_file)

    if "comparison_type" in df.columns:
        for _, group in df.groupby("comparison_type"):
            plot_one(group, output_dir, args.title_prefix)
    else:
        plot_one(df, output_dir, args.title_prefix)


if __name__ == "__main__":
    main()
