"""
Build checkpoint trajectory geometry from full pairwise GW distances.

This script expects a GW output produced with ``--mode full``. It reconstructs
the checkpoint-by-checkpoint GW distance matrix, computes a simple classical
MDS embedding, and writes trajectory-level summaries.

Example
-------
python trajectory_geometry.py \\
    --gw-full-file "$EVOLVE_ROOT/gw_outputs/example__gw_full_L256.parquet" \\
    --output-prefix "$EVOLVE_ROOT/gw_outputs/example__gw_full"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def classical_mds(distance_matrix, dim=2):
    """
    Compute a classical MDS embedding from a distance matrix.

    :param distance_matrix: Square distance matrix.
    :param dim: Embedding dimension.
    :return: Coordinate matrix with shape ``(n_points, dim)``.
    """
    n = distance_matrix.shape[0]
    d2 = distance_matrix ** 2

    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ d2 @ centering

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    keep = np.maximum(eigvals[:dim], 0.0)

    return eigvecs[:, :dim] * np.sqrt(keep)


def build_distance_matrix(df):
    """
    Build a symmetric checkpoint distance matrix from full-mode GW rows.

    :param df: Full-mode GW dataframe.
    :return: Pair ``(distance_matrix, epochs)``.
    """
    epochs = sorted(set(df["source_epoch"]).union(set(df["target_epoch"])))
    epoch_to_index = {epoch: i for i, epoch in enumerate(epochs)}

    dmat = np.zeros((len(epochs), len(epochs)), dtype=float)

    for _, row in df.iterrows():
        i = epoch_to_index[int(row["source_epoch"])]
        j = epoch_to_index[int(row["target_epoch"])]
        d = float(row["gw_distance"])

        dmat[i, j] = d
        dmat[j, i] = d

    return dmat, epochs


def trajectory_summaries(dmat, epochs):
    """
    Compute simple trajectory summaries from a checkpoint distance matrix.

    :param dmat: Symmetric checkpoint distance matrix.
    :param epochs: Ordered checkpoint epochs.
    :return: Dictionary of trajectory summaries.
    """
    adjacent = []

    for i in range(len(epochs) - 1):
        adjacent.append(float(dmat[i, i + 1]))

    adjacent = np.asarray(adjacent, dtype=float)

    if len(epochs) >= 2:
        chord = float(dmat[0, -1])
    else:
        chord = 0.0

    path_length = float(np.sum(adjacent))

    if chord > 0:
        tortuosity = path_length / chord
    else:
        tortuosity = np.nan

    return {
        "num_epochs": int(len(epochs)),
        "trajectory_chord": chord,
        "trajectory_path_length": path_length,
        "trajectory_tortuosity": float(tortuosity),
        "trajectory_max_adjacent_step": (
            float(np.max(adjacent)) if len(adjacent) else np.nan
        ),
        "trajectory_mean_adjacent_step": (
            float(np.mean(adjacent)) if len(adjacent) else np.nan
        ),
    }


def write_outputs(output_prefix, dmat, epochs, coords, summary):
    """
    Write distance matrix, MDS coordinates, and trajectory summaries.

    :param output_prefix: Output path prefix.
    :param dmat: Distance matrix.
    :param epochs: Ordered checkpoint epochs.
    :param coords: MDS coordinates.
    :param summary: Summary dictionary.
    """
    mds = pd.DataFrame(
        {
            "epoch": epochs,
            "mds1": coords[:, 0],
            "mds2": coords[:, 1],
        }
    )

    mds_path = str(output_prefix) + "__mds.parquet"
    summary_path = str(output_prefix) + "__trajectory_summary.parquet"
    matrix_path = str(output_prefix) + "__distance_matrix.npy"

    mds.to_parquet(mds_path, index=False)
    pd.DataFrame([summary]).to_parquet(summary_path, index=False)
    np.save(matrix_path, dmat)

    print(f"[DONE] wrote {mds_path}")
    print(f"[DONE] wrote {summary_path}")
    print(f"[DONE] wrote {matrix_path}")


def main():
    """
    Run the trajectory geometry CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw-full-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    gw_file = Path(args.gw_full_file).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(gw_file)

    dmat, epochs = build_distance_matrix(df)
    coords = classical_mds(dmat, dim=2)
    summary = trajectory_summaries(dmat, epochs)

    write_outputs(output_prefix, dmat, epochs, coords, summary)


if __name__ == "__main__":
    main()
