"""
Config-driven Mapper experiment over EvolveManifold checkpoints.

Example:
    python giotto-exp.py --config mapper_canonical.json

The config controls:
- path/root resolution
- checkpoint discovery
- checkpoint filtering
- Mapper lens
- cover parameters
- clustering parameters
- output location
"""

from pathlib import Path
import argparse
import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from gtda.mapper import CubicalCover, make_mapper_pipeline


class NormLens(BaseEstimator, TransformerMixin):
    """Map each point to its Euclidean norm."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.linalg.norm(X, axis=1, keepdims=True)


class FixedPC1Lens(BaseEstimator, TransformerMixin):
    """Project onto PC1 fitted from the first selected checkpoint."""

    def __init__(self):
        self.pca_ = PCA(n_components=1)

    def fit(self, X, y=None):
        self.pca_.fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X):
        return self.pca_.transform(np.asarray(X, dtype=float))


class FixedPC2Lens(BaseEstimator, TransformerMixin):
    """Project onto first two PCs fitted from the first selected checkpoint."""

    def __init__(self):
        self.pca_ = PCA(n_components=2)

    def fit(self, X, y=None):
        self.pca_.fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X):
        return self.pca_.transform(np.asarray(X, dtype=float))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_base_root(paths_cfg):
    """Resolve base root using env var, fallback env var, then default path."""
    env_name = paths_cfg.get("base_root_env")
    fallback_env_name = paths_cfg.get("base_root_fallback_env")
    default = paths_cfg.get("base_root_default", ".")

    if env_name and os.getenv(env_name):
        return Path(os.getenv(env_name)).expanduser().resolve()

    if fallback_env_name and os.getenv(fallback_env_name):
        return Path(os.getenv(fallback_env_name)).expanduser().resolve()

    return Path(default).expanduser().resolve()


def resolve_checkpoint_root(cfg):
    paths_cfg = cfg.get("paths", {})
    base_root = resolve_base_root(paths_cfg)
    subdir = paths_cfg.get("checkpoint_root_subdir", "")
    return (base_root / subdir).expanduser().resolve()


def resolve_out_dir(cfg):
    paths_cfg = cfg.get("paths", {})
    base_root = resolve_base_root(paths_cfg)

    out_subdir = paths_cfg.get("mapper_output_subdir", "mapper_outputs")
    out_dir = Path(out_subdir).expanduser()

    if not out_dir.is_absolute():
        out_dir = base_root / out_dir

    return out_dir.resolve()


def load_point_cloud(path):
    """Load a point cloud from .npy, .npz, .csv, or .parquet."""
    path = Path(path)

    if path.suffix == ".npy":
        X = np.load(path)

    elif path.suffix == ".npz":
        data = np.load(path)
        if "x" in data:
            X = data["x"]
        elif "X" in data:
            X = data["X"]
        else:
            first_key = list(data.keys())[0]
            X = data[first_key]

    elif path.suffix == ".csv":
        X = pd.read_csv(path).to_numpy()

    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        numeric = df.select_dtypes(include=["number"])
        X = numeric.to_numpy()

    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D point cloud, got shape {X.shape} from {path}")

    return X


def parse_checkpoint_step(path):
    """Best-effort extraction of a numeric checkpoint step from filename."""
    stem = Path(path).stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if not digits:
        return None
    return int(digits[-1])


def path_contains_value(path_text, key, value):
    """
    Flexible path-token matcher.

    It tries common forms:
    - key=value
    - key_value
    - key-value
    - raw value as substring
    """
    value_str = str(value)
    key_str = str(key)

    candidates = [
        f"{key_str}={value_str}",
        f"{key_str}_{value_str}",
        f"{key_str}-{value_str}",
        value_str,
    ]

    return any(c in path_text for c in candidates)


def passes_measurement_filter(path, measurement_filter):
    """
    Filter checkpoint paths using config metadata.

    Logic:
    - any_of is OR over explicit selectors
    - fields outside any_of are global AND filters

    So this config:

        any_of:
          - geometry=torus, mechanism=hole_fill, seed=37
          - geometry=spiked_gaussian, mechanism=linear_to_kplane, seed=5
        n: [1000]
        d: [50]

    means:

        ((torus & hole_fill & seed 37) OR
         (spiked_gaussian & linear_to_kplane & seed 5))
        AND n=1000
        AND d=50
    """
    if not measurement_filter:
        return True

    path_text = str(path)

    any_of = measurement_filter.get("any_of")
    if any_of:
        if not any(selector_matches(path_text, selector) for selector in any_of):
            return False

    global_filter_keys = [
        "geometry",
        "mechanism",
        "n",
        "d",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
    ]

    for key in global_filter_keys:
        allowed_values = measurement_filter.get(key)
        if not allowed_values:
            continue

        if not any(path_contains_value(path_text, key, v) for v in allowed_values):
            return False

    return True


def slugify(value):
    """Make a filesystem-safe identifier."""
    value = str(value)
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def selector_run_id(selector):
    """Build stable run id from explicit run_id or selector fields."""
    if selector.get("run_id"):
        return slugify(selector["run_id"])

    parts = []
    for key in ["geometry", "mechanism", "seed", "severity", "noise", "n", "d"]:
        if key in selector and selector[key] is not None:
            parts.append(f"{key}_{selector[key]}")

    return slugify("__".join(parts))


def token_match(path_text, key, value):
    """
    Flexible path matcher.

    First tries structured forms like seed_5 / seed=5.
    Then falls back to plain substring matching.

    This is intentionally permissive because EvolveManifold paths may encode
    metadata inconsistently across directory/file names.
    """
    value_str = str(value)
    key_str = str(key)

    escaped_value = re.escape(value_str)
    escaped_key = re.escape(key_str)

    structured_patterns = [
        rf"(^|[/_\-]){escaped_key}[=:_\-]{escaped_value}($|[/_\-\.])",
        rf"(^|[/_\-]){escaped_value}($|[/_\-\.])",
    ]

    if any(re.search(p, path_text) for p in structured_patterns):
        return True

    return value_str in path_text


def selector_matches(path_text, selector):
    """
    Check whether all selector fields match the path.

    Ignores run_id because that is only an output name.
    """
    for key, value in selector.items():
        if key == "run_id":
            continue
        if value is None:
            continue
        if not token_match(path_text, key, value):
            return False
    return True


def global_filters_match(path_text, measurement_filter):
    """Apply filters outside any_of as global AND constraints."""
    global_filter_keys = [
        "geometry",
        "mechanism",
        "n",
        "d",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
    ]

    for key in global_filter_keys:
        allowed_values = measurement_filter.get(key)
        if not allowed_values:
            continue

        if not any(token_match(path_text, key, v) for v in allowed_values):
            return False

    return True


def parse_step_for_sort(path):
    """Best-effort numeric step for ordering checkpoints."""
    step = parse_checkpoint_step(path)
    if step is None:
        return 10**18
    return step


def extract_metadata_from_path(path):
    """
    Extract EvolveManifold metadata from checkpoint paths.

    Handles paths like:
        collapse_ph/cluster_merging/
        clustered_gaussian_n1000_d50_k16__linear__moderate__mp0.25__noise0.0__seed5/
        checkpoints/ckpt_epoch_0000.parquet
    """
    path = Path(path)
    text = str(path)

    meta = {}

    # Mechanism is usually the directory under collapse_ph.
    parts = path.parts
    if "collapse_ph" in parts:
        idx = parts.index("collapse_ph")
        if idx + 1 < len(parts):
            meta["mechanism"] = parts[idx + 1]

    # Find the experiment directory containing n/d/schedule/severity/etc.
    exp_part = None
    for part in parts:
        if "_n" in part and "_d" in part and "__" in part:
            exp_part = part
            break

    if exp_part is not None:
        chunks = exp_part.split("__")

        # First chunk: geometry_n1000_d50_k16
        head = chunks[0]

        m = re.search(r"_n(\d+)_d(\d+)", head)
        if m:
            meta["n"] = int(m.group(1))
            meta["d"] = int(m.group(2))
            meta["geometry"] = head[: m.start()]

        m = re.search(r"_k(\d+)", head)
        if m:
            meta["k"] = int(m.group(1))

        # Remaining chunks: linear, moderate, mp0.25, noise0.0, seed5
        for chunk in chunks[1:]:
            if chunk in {"linear", "exponential", "sigmoid"}:
                meta["schedule"] = chunk

            elif chunk in {"weak", "moderate", "strong"}:
                meta["severity"] = chunk

            elif chunk.startswith("mp"):
                try:
                    meta["mover_frac"] = float(chunk.replace("mp", ""))
                except ValueError:
                    meta["mover_frac"] = chunk.replace("mp", "")

            elif chunk.startswith("noise"):
                try:
                    meta["noise"] = float(chunk.replace("noise", ""))
                except ValueError:
                    meta["noise"] = chunk.replace("noise", "")

            elif chunk.startswith("seed"):
                try:
                    meta["seed"] = int(chunk.replace("seed", ""))
                except ValueError:
                    meta["seed"] = chunk.replace("seed", "")

    # Checkpoint step.
    step = parse_checkpoint_step(path)
    if step is not None:
        meta["step"] = step

    return meta


def value_matches(observed, allowed):
    """Compare parsed metadata values against config values robustly."""
    if observed is None:
        return False

    for value in allowed:
        if observed == value:
            return True

        # Numeric/string fallback.
        try:
            if float(observed) == float(value):
                return True
        except Exception:
            pass

        if str(observed) == str(value):
            return True

    return False


def metadata_matches_filter(meta, measurement_filter):
    """Apply global measurement_filter using parsed metadata, not path substrings."""
    filter_keys = [
        "geometry",
        "mechanism",
        "n",
        "d",
        "schedule",
        "severity",
        "mover_frac",
        "noise",
        "seed",
    ]

    for key in filter_keys:
        allowed = measurement_filter.get(key)
        if not allowed:
            continue

        if not value_matches(meta.get(key), allowed):
            return False

    return True


def selector_matches_metadata(meta, selector):
    """
    Check explicit any_of selector against parsed metadata.

    Ignores run_id.
    """
    for key, value in selector.items():
        if key == "run_id":
            continue

        if value is None:
            continue

        if not value_matches(meta.get(key), [value]):
            return False

    return True


def group_id_from_metadata(meta, group_by):
    """Build a stable run_id from parsed metadata."""
    parts = []
    for key in group_by:
        value = meta.get(key, "unknown")
        parts.append(f"{key}_{value}")
    return slugify("__".join(parts))


def discover_checkpoint_records(cfg):
    """
    Discover checkpoint paths and group them into Mapper runs.

    Supports two modes:

    1. Explicit selector mode:
        measurement_filter.any_of = [{geometry, mechanism, seed, ...}, ...]

    2. Group-by mode:
        checkpoint_selection.group_by = [
            "geometry", "mechanism", "schedule", "severity", "mover_frac", "noise", "seed"
        ]

    In group-by mode, every unique metadata tuple becomes a run.
    """
    selection = cfg.get("checkpoint_selection", {})
    measurement_filter = cfg.get("measurement_filter", {})
    any_of = measurement_filter.get("any_of")
    group_by = selection.get("group_by")

    checkpoint_root = resolve_checkpoint_root(cfg)

    explicit = selection.get("explicit_checkpoint_paths", [])
    if explicit:
        all_paths = [Path(p).expanduser().resolve() for p in explicit]
    else:
        checkpoint_globs = selection.get("checkpoint_globs", ["**/*.npy", "**/*.npz", "**/*.parquet"])
        all_paths = []

        for pattern in checkpoint_globs:
            full_pattern = str(checkpoint_root / pattern)
            all_paths.extend(Path(p).resolve() for p in glob.glob(full_pattern, recursive=True))

        all_paths = sorted(set(all_paths))

    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Discovered candidate files: {len(all_paths)}")

    max_per_selector = selection.get("max_checkpoints_per_selector")
    if max_per_selector is None:
        max_per_selector = selection.get("max_checkpoints")

    grouped = {}

    for path in all_paths:
        meta = extract_metadata_from_path(path)

        # First apply global filter.
        if not metadata_matches_filter(meta, measurement_filter):
            continue

        if any_of:
            matched_any = False

            for selector in any_of:
                if selector_matches_metadata(meta, selector):
                    run_id = selector_run_id(selector)
                    grouped.setdefault(run_id, []).append(
                        {
                            "path": path,
                            "run_id": run_id,
                            "selector": selector,
                            "metadata": meta,
                        }
                    )
                    matched_any = True

            if not matched_any:
                continue

        elif group_by:
            run_id = group_id_from_metadata(meta, group_by)
            grouped.setdefault(run_id, []).append(
                {
                    "path": path,
                    "run_id": run_id,
                    "selector": {},
                    "metadata": meta,
                }
            )

        else:
            run_id = "all"
            grouped.setdefault(run_id, []).append(
                {
                    "path": path,
                    "run_id": run_id,
                    "selector": {},
                    "metadata": meta,
                }
            )

    records = []

    for run_id, recs in sorted(grouped.items()):
        recs = sorted(recs, key=lambda r: (parse_step_for_sort(r["path"]), str(r["path"])))

        if max_per_selector is not None:
            recs = recs[: int(max_per_selector)]

        print(f"Run {run_id}: matched {len(recs)} checkpoints")

        records.extend(recs)

    return records


def graph_summary(graph):
    """Extract graph and node-size summaries from a giotto-tda Mapper graph."""
    summary = {}

    if hasattr(graph, "vcount"):
        n_nodes = graph.vcount()
        n_edges = graph.ecount()
        degrees = graph.degree() if n_nodes > 0 else []

        summary["mapper_nodes"] = n_nodes
        summary["mapper_edges"] = n_edges
        summary["mapper_mean_degree"] = float(np.mean(degrees)) if degrees else 0.0
        summary["mapper_max_degree"] = int(np.max(degrees)) if degrees else 0

        comps = graph.components() if hasattr(graph, "components") else []
        comp_sizes = [len(c) for c in comps] if comps else []

        summary["mapper_components"] = len(comp_sizes)
        summary["mapper_largest_component"] = max(comp_sizes) if comp_sizes else 0
        summary["mapper_largest_component_frac"] = (
            max(comp_sizes) / n_nodes if n_nodes > 0 and comp_sizes else 0.0
        )

        summary["mapper_beta1_graph"] = n_edges - n_nodes + len(comp_sizes)

        # giotto-tda Mapper nodes often store original point memberships
        # in vertex attributes. Attribute names vary slightly by version.
        node_sizes = []

        for v in graph.vs:
            attrs = v.attributes()

            if "node_elements" in attrs:
                elems = attrs["node_elements"]
                node_sizes.append(len(elems))

            elif "indices" in attrs:
                elems = attrs["indices"]
                node_sizes.append(len(elems))

            elif "pullback_set_label" in attrs:
                # Not a true node size, but useful for debugging.
                continue

        if node_sizes:
            summary["mapper_node_size_min"] = int(np.min(node_sizes))
            summary["mapper_node_size_median"] = float(np.median(node_sizes))
            summary["mapper_node_size_mean"] = float(np.mean(node_sizes))
            summary["mapper_node_size_max"] = int(np.max(node_sizes))
            summary["mapper_total_node_memberships"] = int(np.sum(node_sizes))
        else:
            summary["mapper_node_size_min"] = np.nan
            summary["mapper_node_size_median"] = np.nan
            summary["mapper_node_size_mean"] = np.nan
            summary["mapper_node_size_max"] = np.nan
            summary["mapper_total_node_memberships"] = np.nan

    else:
        for key in [
            "mapper_nodes",
            "mapper_edges",
            "mapper_mean_degree",
            "mapper_max_degree",
            "mapper_components",
            "mapper_largest_component",
            "mapper_largest_component_frac",
            "mapper_beta1_graph",
            "mapper_node_size_min",
            "mapper_node_size_median",
            "mapper_node_size_mean",
            "mapper_node_size_max",
            "mapper_total_node_memberships",
        ]:
            summary[key] = np.nan

    return summary


def build_mapper_pipeline(cfg, X0):
    mapper_cfg = cfg.get("mapper", {})

    standardize = bool(mapper_cfg.get("standardize", False))
    scaler = None

    if standardize:
        scaler = StandardScaler().fit(X0)
        X0_for_fit = scaler.transform(X0)
    else:
        X0_for_fit = X0

    lens_name = mapper_cfg.get("lens", "fixed_pc1")

    if lens_name == "norm":
        lens = NormLens().fit(X0_for_fit)
    elif lens_name == "fixed_pc1":
        lens = FixedPC1Lens().fit(X0_for_fit)
    elif lens_name == "fixed_pc2":
        lens = FixedPC2Lens().fit(X0_for_fit)
    else:
        raise ValueError(f"Unsupported lens: {lens_name}")

    cover_cfg = mapper_cfg.get("cover", {})
    cover = CubicalCover(
        n_intervals=int(cover_cfg.get("n_intervals", 8)),
        overlap_frac=float(cover_cfg.get("overlap_frac", 0.35)),
    )

    cluster_cfg = mapper_cfg.get("clusterer", {})
    cluster_name = cluster_cfg.get("name", "dbscan")

    if cluster_name != "dbscan":
        raise ValueError(f"Unsupported clusterer: {cluster_name}")

    clusterer = DBSCAN(
        eps=float(cluster_cfg.get("eps", 0.5)),
        min_samples=int(cluster_cfg.get("min_samples", 3)),
    )

    mapper = make_mapper_pipeline(
        filter_func=lens,
        cover=cover,
        clusterer=clusterer,
        verbose=False,
    )

    return mapper, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)

    out_dir = resolve_out_dir(cfg)
    graph_dir = out_dir / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    records = discover_checkpoint_records(cfg)

    if not records:
        checkpoint_root = resolve_checkpoint_root(cfg)
        raise FileNotFoundError(
            "No checkpoints matched the config.\n"
            f"checkpoint_root={checkpoint_root}\n"
            "Try loosening measurement_filter or using explicit_checkpoint_paths."
        )

    print(f"Matched {len(records)} total checkpoints.")
    print(f"Output directory: {out_dir}")

    all_rows = []

    records_by_run = {}
    for rec in records:
        records_by_run.setdefault(rec["run_id"], []).append(rec)

    for run_id, run_records in records_by_run.items():
        run_dir = out_dir / "runs" / run_id
        graph_dir = run_dir / "graphs"
        run_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Run: {run_id} ===")
        print(f"Checkpoints: {len(run_records)}")
        print(f"Run output: {run_dir}")

        # Fit scaler/lens separately per run using that run's first checkpoint.
        X0 = load_point_cloud(run_records[0]["path"])
        mapper, scaler = build_mapper_pipeline(cfg, X0)

        mapper_cfg = cfg.get("mapper", {})
        cover_cfg = mapper_cfg.get("cover", {})
        cluster_cfg = mapper_cfg.get("clusterer", {})
        save_graphs = bool(mapper_cfg.get("save_graphs", False))

        run_rows = []

        for i, rec in enumerate(run_records):
            path = rec["path"]
            selector = rec["selector"]

            X = load_point_cloud(path)
            X_use = scaler.transform(X) if scaler is not None else X

            try:
                graph = mapper.fit_transform(X_use)
                mapper_failed = False
                mapper_error = ""
            except Exception as e:
                graph = None
                mapper_failed = True
                mapper_error = str(e)
                print(f"[WARNING] Mapper failed for {path}: {e}")

            step = parse_checkpoint_step(path)
            step_label = "none" if step is None else f"{int(step):03d}"

            row = {
                "experiment_name": cfg.get("name"),
                "run_id": run_id,
                "checkpoint_index": i,
                "checkpoint_path": str(path),
                "checkpoint_step_guess": step,
                "n_points": X.shape[0],
                "ambient_dim": X.shape[1],
                "lens": mapper_cfg.get("lens", "fixed_pc1"),
                "standardize": bool(mapper_cfg.get("standardize", False)),
                "cover_intervals": int(cover_cfg.get("n_intervals", 8)),
                "cover_overlap": float(cover_cfg.get("overlap_frac", 0.35)),
                "clusterer": cluster_cfg.get("name", "dbscan"),
                "dbscan_eps": float(cluster_cfg.get("eps", 0.5)),
                "dbscan_min_samples": int(cluster_cfg.get("min_samples", 3)),
            }

            # Store selector fields as columns.
            # Store parsed metadata as columns.
            meta = rec.get("metadata", {})
            for key, value in meta.items():
                row[key] = value

            # Store selector fields as columns when using explicit any_of mode.
            for key, value in selector.items():
                if key != "run_id":
                    row[f"selector_{key}"] = value

            if graph is None:
                row.update({
                    "mapper_failed": True,
                    "mapper_error": mapper_error,
                    "mapper_nodes": np.nan,
                    "mapper_edges": np.nan,
                    "mapper_mean_degree": np.nan,
                    "mapper_max_degree": np.nan,
                    "mapper_components": np.nan,
                    "mapper_largest_component": np.nan,
                    "mapper_largest_component_frac": np.nan,
                    "mapper_beta1_graph": np.nan,
                    "mapper_node_size_min": np.nan,
                    "mapper_node_size_median": np.nan,
                    "mapper_node_size_mean": np.nan,
                    "mapper_node_size_max": np.nan,
                    "mapper_total_node_memberships": np.nan,
                })
            else:
                row.update(graph_summary(graph))
                row["mapper_failed"] = False
                row["mapper_error"] = ""


            if save_graphs and graph is not None:
                graph_name = f"{run_id}__frame_{i:03d}__step_{step_label}.pkl"
                graph_path = graph_dir / graph_name
                with open(graph_path, "wb") as f:
                    pickle.dump(graph, f)
                row["graph_pickle"] = str(graph_path)
            else:
                row["graph_pickle"] = ""

            run_rows.append(row)
            all_rows.append(row)

            if mapper_failed:
                print(f"[{i + 1}/{len(run_records)}] {path} MAPPER_FAILED")
            else:
                print(
                    f"[{i + 1}/{len(run_records)}] {path} "
                    f"nodes={row['mapper_nodes']} "
                    f"edges={row['mapper_edges']} "
                    f"beta1={row['mapper_beta1_graph']}"
                )

        run_summary = pd.DataFrame(run_rows)
        run_summary_path = run_dir / "mapper_summary.csv"
        run_summary.to_csv(run_summary_path, index=False)

        resolved_config_path = run_dir / "resolved_mapper_config.json"
        with open(resolved_config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"Wrote run summary: {run_summary_path}")

    all_summary = pd.DataFrame(all_rows)
    all_summary_path = out_dir / "mapper_summary_all.csv"
    all_summary.to_csv(all_summary_path, index=False)

    resolved_config_path = out_dir / "resolved_mapper_config.json"
    with open(resolved_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nWrote all-run summary: {all_summary_path}")
    print(f"Wrote config copy: {resolved_config_path}")


if __name__ == "__main__":
    main()
