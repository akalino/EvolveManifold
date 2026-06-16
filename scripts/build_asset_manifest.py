"""
Convert checkpoint, metric, and output files to benchmark assets.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import yaml

import pandas as pd

from dataclasses import dataclass, asdict


TRANCHES = {
    "canonical",
    "ph_audit",
    "primary_d50",
    "primary_d100",
    "noise",
    "n_scaling",
}

PH_MODES = {
    "full_vr",
    "skip_vr",
    "event_driven",
    "landmark_vr",
    "online_landmark_dynamic_support",
    "fixed_knn_vr",
    "fixed_support_vr",
    "online_landmark_event",
}

GEOMETRIES = {
    "spiked_gaussian",
    "isotropic",
    "clustered_gaussian",
    "torus",
    "kplane",
    "hypercube",
    "sphere",
    "paraboloid",
    "swiss_roll",
    "gaussian",
}

MECHANISMS = {
    "linear_to_kplane",
    "radial_collapse",
    "cluster_tightening",
    "cluster_merging",
    "hole_fill",
    "projection",
    "radial",
    "topology_first",
}

SCHEDULES = {"linear", "exponential", "sigmoid"}
SEVERITIES = {"mild", "moderate", "strong", "severe"}

METADATA_COLUMNS = [
    "tranche",
    "geometry",
    "mechanism",
    "n",
    "d",
    "schedule",
    "severity",
    "mover_frac",
    "noise",
    "seed",
    "epoch",
    "ph_mode",
]


@dataclass
class ManifestRow:
    artifact_id: str
    artifact_type: str
    relative_path: str
    absolute_path: str
    extension: str
    size_bytes: int
    modified_utc: str
    sha256: str | None

    tranche: str | None = None
    geometry: str | None = None
    mechanism: str | None = None
    n: int | None = None
    d: int | None = None
    schedule: str | None = None
    severity: str | None = None
    mover_frac: float | None = None
    noise: float | None = None
    seed: int | None = None
    epoch_min: int | None = None
    epoch_max: int | None = None
    ph_mode: str | None = None

    rows: int | None = None
    columns: int | None = None
    column_names: str | None = None

    code_commit: str | None = None
    manifest_created_utc: str | None = None


def classify_artifact(path):
    name = path.name.lower()
    suffix = path.suffix.lower()
    parts = [p.lower() for p in path.parts]

    if suffix in {".yaml", ".json"} or "configs" in parts:
        return "config"

    if suffix in {".png", ".pdf"}:
        if "figure" in parts or "figures" in parts or name.startswith("fig"):
            return "figure"
        return "image_or_pdf"

    if suffix in {".csv", ".md", ".tex"}:
        if "manifest" in name:
            return "manifest"
        if "table" in parts or "tables" in parts or "summary" in name:
            return "table_or_summary"
        return "text_table"

    if suffix in {".parquet"}:
        if "metric" in name or "metrics" in parts or "metric_outputs" in parts:
            return "metric_trace"
        if "summary" in name:
            return "summary_table"
        return "columnar_data"

    if suffix in {".npy", ".npz", ".pt", ".pth", ".h5", ".hdf5", ".zarr"}:
        if "checkpoint" in parts or "trajectory" in parts or "trajectories" in parts:
            return "checkpointed_point_cloud"
        return "array_data"

    if suffix in {".py", ".sh", ".ipynb"}:
        return "script"

    return "other"


def normalize_token(token):
    return token.strip().strip("/").replace("-", "_")


def maybe_int(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(float(x))
    except Exception:
        return None


def maybe_float(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def infer_from_path(path):
    parts = [normalize_token(p) for p in path.parts]
    joined = "/".join(parts)
    meta = {}

    for part in parts:
        if part in TRANCHES:
            meta.setdefault("tranche", part)
        if part in PH_MODES:
            meta.setdefault("ph_mode", part)
        if part in GEOMETRIES:
            meta.setdefault("geometry", part)
        if part in MECHANISMS:
            meta.setdefault("mechanism", part)
        if part in SCHEDULES:
            meta.setdefault("schedule", part)
        if part in SEVERITIES:
            meta.setdefault("severity", part)

    patterns = {
        "n": r"(?:^|[_/\-])n[_=\-]?(\d+)(?:$|[_/\-])",
        "d": r"(?:^|[_/\-])d[_=\-]?(\d+)(?:$|[_/\-])",
        "seed": r"(?:^|[_/\-])seed[_=\-]?(\d+)(?:$|[_/\-])",
        "epoch": r"(?:^|[_/\-])epoch[_=\-]?(\d+)(?:$|[_/\-])",
        "mover_frac": r"(?:^|[_/\-])(?:mover_frac|moved_frac|frac)[_=\-]?([0-9.]+)(?:$|[_/\-])",
        "noise": r"(?:^|[_/\-])noise[_=\-]?([0-9.]+)(?:$|[_/\-])",
    }

    for key, pat in patterns.items():
        m = re.search(pat, joined)
        if m:
            val = m.group(1)
            if key in {"n", "d", "seed", "epoch"}:
                meta[key] = maybe_int(val)
            else:
                meta[key] = maybe_float(val)

    if "epoch" in meta:
        meta["epoch_min"] = meta["epoch"]
        meta["epoch_max"] = meta["epoch"]
        meta.pop("epoch", None)

    return meta


def read_tabular_metadata(path, max_rows_for_csv):
    meta = {}
    suffix = path.suffix.lower()

    try:
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path, nrows=max_rows_for_csv)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=max_rows_for_csv)
        else:
            return meta
    except Exception:
        return meta

    meta["rows"] = int(df.shape[0])
    meta["columns"] = int(df.shape[1])
    meta["column_names"] = ",".join(map(str, df.columns))

    for col in METADATA_COLUMNS:
        if col not in df.columns:
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        if col == "epoch":
            nums = pd.to_numeric(s, errors="coerce").dropna()
            if not nums.empty:
                meta["epoch_min"] = int(nums.min())
                meta["epoch_max"] = int(nums.max())
            continue

        unique_vals = s.unique()
        if len(unique_vals) == 1:
            val = unique_vals[0]
            if col in {"n", "d", "seed"}:
                meta[col] = maybe_int(val)
            elif col in {"mover_frac", "noise"}:
                meta[col] = maybe_float(val)
            else:
                meta[col] = str(val)

    return meta


def read_json_or_yaml_metadata(path):
    suffix = path.suffix.lower()
    meta = {}

    if suffix not in {".json", ".yaml", ".yml"}:
        return meta

    data = None
    try:
        if suffix == ".json":
            data = json.loads(path.read_text())
        else:
            try:
                data = yaml.safe_load(path.read_text())
            except Exception:
                data = None
    except Exception:
        data = None

    if not isinstance(data, dict):
        return meta

    candidates = [data]
    for v in data.values():
        if isinstance(v, dict):
            candidates.append(v)

    for key in METADATA_COLUMNS:
        values = [c.get(key) for c in candidates if key in c]
        value = next((v for v in values if v is not None), None)
        if value is None or key == "epoch":
            continue

        if key in {"n", "d", "seed"}:
            meta[key] = maybe_int(value)
        elif key in {"mover_frac", "noise"}:
            meta[key] = maybe_float(value)
        else:
            meta[key] = str(value)

    return meta


def safe_relpath(path, base_dir):
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def file_mtime_utc(path):
    return dt.datetime.fromtimestamp(
        path.stat().st_mtime, tz=dt.timezone.utc
    ).replace(microsecond=0).isoformat()


def build_row(
    path,
    base_dir,
    created_utc,
    commit,
    hash_files,
    hash_max_mb,
    artifact_prefix,
    index
) -> ManifestRow:
    rel = safe_relpath(path, base_dir)
    size = path.stat().st_size

    sha = None
    if hash_files:
        if hash_max_mb is None or size <= hash_max_mb * 1024 * 1024:
            sha = sha256_file(path)

    row = ManifestRow(
        artifact_id=f"{artifact_prefix}_{index:07d}",
        artifact_type=classify_artifact(path),
        relative_path=rel,
        absolute_path=str(path.resolve()),
        extension=path.suffix.lower(),
        size_bytes=int(size),
        modified_utc=file_mtime_utc(path),
        sha256=sha,
        code_commit=commit,
        manifest_created_utc=created_utc,
    )

    meta = {}
    meta.update(infer_from_path(path))
    meta.update(read_json_or_yaml_metadata(path))
    meta.update(read_tabular_metadata(path))

    for key, value in meta.items():
        if hasattr(row, key):
            setattr(row, key, value)

    return row


def should_skip(path, base_dir, exclude_patterns):
    rel = safe_relpath(path, base_dir)
    skip_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
    }

    if set(path.parts) & skip_dirs:
        return True

    for pat in exclude_patterns:
        if re.search(pat, rel):
            return True

    return False


def iter_files(roots, base_dir, exclude_patterns):
    seen = set()

    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            print(f"[WARN] root does not exist: {root}", file=sys.stderr)
            continue

        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]

        for path in files:
            rp = path.resolve()
            if rp in seen:
                continue
            if should_skip(rp, base_dir, exclude_patterns):
                continue
            seen.add(rp)
            yield rp


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ManifestRow.__dataclass_fields__.keys())

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_run_summary(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])

    if df.empty:
        df.to_csv(out_path, index=False)
        return

    group_cols = [
        "tranche",
        "geometry",
        "mechanism",
        "n",
        "d",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
        "ph_mode",
    ]

    for col in group_cols:
        if col not in df.columns:
            df[col] = None

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            artifact_count=("artifact_id", "count"),
            total_size_bytes=("size_bytes", "sum"),
            artifact_types=("artifact_type", lambda x: ",".join(sorted(set(map(str, x))))),
            path_examples=("relative_path", lambda x: " | ".join(list(map(str, x.head(3))))),
            epoch_min=("epoch_min", "min"),
            epoch_max=("epoch_max", "max"),
        )
        .reset_index()
        .sort_values(group_cols, na_position="last")
    )

    summary.to_csv(out_path, index=False)


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or None
    except Exception:
        return None

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a file-level release manifest for EvolveManifold assets."
    )
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("manifests/release_manifest.csv"))
    parser.add_argument("--run-summary", type=Path, default=None)
    parser.add_argument("--no-hash", action="store_true")
    parser.add_argument("--hash-max-mb", type=float, default=None)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--artifact-prefix", default="em")
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    created = utc_now()
    commit = git_commit(base_dir)

    paths = list(iter_files(args.roots, base_dir, args.exclude))
    print(f"[INFO] files discovered: {len(paths):,}")

    rows: list[ManifestRow] = []
    for i, path in enumerate(paths, start=1):
        if i % 500 == 0:
            print(f"[INFO] processed {i:,}/{len(paths):,}")
        try:
            rows.append(
                build_row(
                    path=path,
                    base_dir=base_dir,
                    created_utc=created,
                    commit=commit,
                    hash_files=not args.no_hash,
                    hash_max_mb=args.hash_max_mb,
                    artifact_prefix=args.artifact_prefix,
                    index=i,
                )
            )
        except Exception as exc:
            print(f"[WARN] failed to process {path}: {exc}", file=sys.stderr)

    write_csv(rows, args.out)
    print(f"[WROTE] {args.out}")

    if args.run_summary:
        write_run_summary(rows, args.run_summary)
        print(f"[WROTE] {args.run_summary}")

    if pd is not None and rows:
        df = pd.DataFrame([asdict(r) for r in rows])
        print("\n[SUMMARY] artifact_type counts")
        print(df["artifact_type"].value_counts(dropna=False).to_string())

        print("\n[SUMMARY] total size by artifact_type")
        size_summary = (
            df.groupby("artifact_type", dropna=False)["size_bytes"]
            .sum()
            .sort_values(ascending=False)
        )
        for k, v in size_summary.items():
            print(f"{k:28s} {v / (1024**3):9.3f} GiB")


if __name__ == "__main__":
    main()