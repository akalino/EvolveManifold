"""
Summarize IsoScore and intrinsic-dimension metric traces.
"""

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "iso_score",
    "anisotropy_ratio",
    "id_two_nn",
    "id_mle",
]


def summarize_file(path):
    """
    Summarize one metric trace.

    :param path: Parquet or CSV metric trace.
    :return: Summary row.
    """
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    row = {
        "source_file": str(path),
        "n_checkpoints": int(len(df)),
    }

    for col in [
        "geometry",
        "mechanism",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
        "n",
        "d",
        "n_points",
        "ambient_dim",
    ]:
        if col in df.columns and len(df[col].dropna()) > 0:
            row[col] = df[col].dropna().iloc[0]

    for metric in METRICS:
        if metric not in df.columns:
            continue

        vals = df[metric]
        row[f"{metric}_start"] = vals.iloc[0]
        row[f"{metric}_final"] = vals.iloc[-1]
        row[f"{metric}_delta"] = vals.iloc[-1] - vals.iloc[0]
        row[f"{metric}_min"] = vals.min()
        row[f"{metric}_max"] = vals.max()
        row[f"{metric}_mean"] = vals.mean()

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    files = sorted(input_root.rglob("*__iso_id.parquet"))
    files += sorted(input_root.rglob("*__iso_id.csv"))

    if not files:
        raise RuntimeError(f"No iso_id outputs found under {input_root}")

    rows = [summarize_file(path) for path in files]
    out = pd.DataFrame(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv":
        out.to_csv(output, index=False)
    else:
        out.to_parquet(output, index=False)

    print(f"[DONE] wrote {len(out)} rows -> {output}")


if __name__ == "__main__":
    main()
