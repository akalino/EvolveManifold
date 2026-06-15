"""
Fragile-hardware runner redesign.

Goals:
- one run directory per trajectory
- one manifest.json per run
- parquet checkpoints (not pickle)
- atomic writes for manifests / checkpoints
- no reliance on output-directory scans
- safer skip semantics on fragile disks

Note:
This keeps your experiment-building and step-function logic mostly intact.
The only likely integration point that may need a tiny follow-up patch is the
exact callback method that `trajectory.dynamics(...)` expects on the checkpoint
writer. To hedge against that, ParquetCheckpointWriter exposes several common
methods (`save`, `write`, `checkpoint`, and `__call__`) that all do the same
thing.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cluster_mechanic import step_cluster_collapse, cluster_params_from_severity
from clustering import make_clustered_gaussian, get_cluster_labels_for_geometry
from experiment_dataclasses import TrajectoryExperiment
from geometry import get_geometry
from linear_mechanic import LinearSpectralParams, step_linear_spectral
from nonlinear_mechanic import NonLinearParams, step_nonlinear_projection
from projectors import (
    proj_to_k_plane,
    proj_to_sphere,
    proj_to_torus,
    proj_to_paraboloid,
)
from radial_mechanic import step_radial, radial_params_from_severity
from topological_mechanisms import (
    HoleFillParams,
    PinchParams,
    BridgeParams,
    step_hole_fill,
    step_loop_pinch,
    step_bridge_across_hole,
)
from trajectory import dynamics


# Default to the stable mount point we just created. You can override this
# without editing the script:
#   EVOLVE_COLLAPSE_ROOT=/some/other/path python run_parquet_manifest_safe.py
LOCAL_ROOT = os.path.expanduser("~/evolve_local/evolve_collapse")

EXTERNAL_ROOT = os.environ.get("EVOLVE_ROOT", LOCAL_ROOT)

CHECKPOINT_ROOT = os.path.join(EXTERNAL_ROOT, "evolve_checkpoints")
METRIC_ROOT = os.path.join(EXTERNAL_ROOT, "metric_outputs")
SUMMARY_ROOT = os.path.join(EXTERNAL_ROOT, "metric_summaries")
ASSET_ROOT = os.path.join(EXTERNAL_ROOT, "summary_assets")
CLUSTER_LABEL_ROOT = os.path.join(EXTERNAL_ROOT, "cluster_labels")

# Append-friendly run index. This is not the source of truth for a run;
# the per-run manifest is. This file is just a convenient catalog.
GLOBAL_RUN_INDEX = os.path.join(CHECKPOINT_ROOT, "run_index.jsonl")

PAPER_FOCUSED_GRID = {
    "geometries": [
        "clustered_gaussian",
        "torus",
        "isotropic",
        "spiked_gaussian"
    ],
    "mechanisms": [
        "projection",
        "linear_to_kplane",
        "radial_collapse",
        "cluster_tightening",
        "cluster_merging",
        "hole_fill"
    ],
    "schedules": [
        "linear"
    ],
    "severities": [
        "moderate",
        "strong"
    ],
    "mover_fracs": [
        1.0
    ],
    "noises": [
        0.0,
    ],
    "n_values": [
        1000,
    ],
    "d_values": [
        50
    ],
    "seeds": [
        5,
        17,
        26,
        31,
        37,
        51,
        123,
        821,
        1111,
        1823
    ],
    "num_steps": 50,
    "checkpoint_every": 2,
}


PRIMARY_D50_GRID = {
    "geometries": [
        "clustered_gaussian",
        "torus",
        "isotropic",
        "spiked_gaussian",
    ],
    "mechanisms": [
        "projection",
        "linear_to_kplane",
        "radial_collapse",
        "cluster_tightening",
        "cluster_merging",
        "hole_fill",
    ],
    "schedules": [
        "linear",
        "exponential",
        "sigmoid",
    ],
    "severities": [
        "weak",
        "moderate",
        "strong",
    ],
    "mover_fracs": [
        0.25,
        0.5,
        1.0,
    ],
    "noises": [
        0.0,
    ],
    "n_values": [
        1000,
    ],
    "d_values": [
        50,
    ],
    "seeds": [
        5,
        17,
        26,
        31,
        37,
        51,
        123,
        821,
        1111,
        1823,
    ],
    "num_steps": 50,
    "checkpoint_every": 2,
}

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fsync_parent(path: str | Path) -> None:
    """Best-effort directory fsync so rename metadata reaches disk."""
    path = Path(path)
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        # Some filesystems or mounts do not allow directory fsync. The atomic
        # replace still protects readers from half-written files.
        pass


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_parent(path)


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def make_run_stem(exp: TrajectoryExperiment) -> str:
    return (
        f"{exp.base_geometry}"
        f"_n{exp.n}"
        f"_d{exp.d}"
        f"_k{exp.k}"
        f"__{exp.schedule}"
        f"__{exp.severity}"
        f"__mp{exp.mover_frac}"
        f"__noise{exp.noise}"
        f"__seed{exp.seed}"
    )


def get_label_root_from_checkpoint_root(checkpoint_root: str) -> str:
    parent = os.path.dirname(checkpoint_root.rstrip(os.sep))
    return os.path.join(parent, "cluster_label_outputs")


def save_cluster_labels(exp: TrajectoryExperiment, labels, label_root: str) -> None:
    os.makedirs(label_root, exist_ok=True)

    stem = make_run_stem(exp)
    labels_path = os.path.join(label_root, f"{stem}_labels.npy")
    meta_path = os.path.join(label_root, f"{stem}_labels_meta.json")

    labels_tmp = labels_path + f".tmp.{os.getpid()}.npy"
    np.save(labels_tmp, np.asarray(labels, dtype=int))
    os.replace(labels_tmp, labels_path)
    fsync_parent(labels_path)

    meta = {
        "base_geometry": exp.base_geometry,
        "n": exp.n,
        "d": exp.d,
        "k": exp.k,
        "schedule": exp.schedule,
        "severity": exp.severity,
        "noise": exp.noise,
        "seed": exp.seed,
        "mechanism": exp.mechanism,
        "num_clusters": int(len(np.unique(labels))),
        "saved_at": utc_now_iso(),
    }
    atomic_write_json(meta_path, meta)


def get_mechanism_params(mechanism: str, severity: str) -> Dict[str, float]:
    if mechanism == "linear_to_kplane":
        if severity == "weak":
            return {"alpha_0": 1.0, "alpha_t": 0.5}
        if severity == "moderate":
            return {"alpha_0": 1.0, "alpha_t": 0.2}
        return {"alpha_0": 1.0, "alpha_t": 0.05}

    if mechanism == "hole_fill":
        if severity == "weak":
            return {"r0": 0.0, "rt": 0.2}
        if severity == "moderate":
            return {"r0": 0.0, "rt": 0.5}
        return {"r0": 0.0, "rt": 0.8}

    if mechanism == "loop_pinch":
        if severity == "weak":
            return {"strength_0": 0.0, "strength_t": 0.3}
        if severity == "moderate":
            return {"strength_0": 0.0, "strength_t": 0.6}
        return {"strength_0": 0.0, "strength_t": 1.0}

    if mechanism == "bridge_across_hole":
        if severity == "weak":
            return {"strength_0": 0.0, "strength_t": 0.3}
        if severity == "moderate":
            return {"strength_0": 0.0, "strength_t": 0.6}
        return {"strength_0": 0.0, "strength_t": 1.0}

    if severity == "weak":
        return {"eps_0": 0.5, "eps_t": 0.05, "relax": 0.2}
    if severity == "moderate":
        return {"eps_0": 0.5, "eps_t": 0.02, "relax": 0.5}
    return {"eps_0": 0.5, "eps_t": 0.005, "relax": 1.0}


def is_valid_combo(geometry: str, mechanism: str) -> bool:
    cluster_mechanisms = {"cluster_tightening", "cluster_merging"}
    topology_mechanisms = {"hole_fill", "loop_pinch", "bridge_across_hole"}

    if mechanism in cluster_mechanisms:
        return geometry == "clustered_gaussian"
    if mechanism in topology_mechanisms:
        return geometry == "torus"
    if mechanism == "linear_to_kplane" and geometry == "torus":
        return False
    return True


def build_experiments(
    n: int,
    d: int,
    num_steps: int,
    checkpoint_every: int,
    seed: int,
    k: int,
    grid: Optional[Dict[str, Any]],
):
    if grid is None:
        grid = PAPER_FOCUSED_GRID

    exps = []
    for geom, mech, sched, sev, mp, noise in product(
        grid["geometries"],
        grid["mechanisms"],
        grid["schedules"],
        grid["severities"],
        grid["mover_fracs"],
        grid["noises"],
    ):
        if not is_valid_combo(geom, mech):
            continue
        exps.append(
            TrajectoryExperiment(
                base_geometry=geom,
                mechanism=mech,
                n=n,
                d=d,
                k=k,
                total_steps=num_steps,
                checkpoint_every=checkpoint_every,
                seed=seed,
                mover_frac=mp,
                noise=noise,
                schedule=sched,
                severity=sev,
                mechanism_params=get_mechanism_params(mech, sev),
            )
        )
    return exps


def build_step(exp: TrajectoryExperiment, x0=None):
    if exp.mechanism == "cluster_tightening":
        labels = get_cluster_labels_for_geometry(exp, x0)
        p = cluster_params_from_severity(
            severity=exp.severity,
            schedule=exp.schedule,
            finish=exp.total_steps,
            cluster_labels=labels,
            mover_frac=exp.mover_frac,
            mode="tighten",
            seed=exp.seed,
        )
        return step_cluster_collapse(p)

    if exp.mechanism == "cluster_merging":
        labels = get_cluster_labels_for_geometry(exp, x0)
        p = cluster_params_from_severity(
            severity=exp.severity,
            schedule=exp.schedule,
            finish=exp.total_steps,
            cluster_labels=labels,
            mover_frac=exp.mover_frac,
            mode="merge",
            seed=exp.seed,
        )
        return step_cluster_collapse(p)

    if exp.mechanism == "linear_to_kplane":
        return step_linear_spectral(
            LinearSpectralParams(
                k=exp.k,
                alpha_0=exp.mechanism_params["alpha_0"],
                alpha_t=exp.mechanism_params["alpha_t"],
                noise=exp.noise,
                schedule=exp.schedule,
            ),
            exp.total_steps,
        )

    if exp.mechanism == "hole_fill":
        return step_hole_fill(
            HoleFillParams(
                r0=exp.mechanism_params["r0"],
                rt=exp.mechanism_params["rt"],
                schedule=exp.schedule,
                noise=exp.noise,
            ),
            exp.total_steps,
        )

    if exp.mechanism == "loop_pinch":
        return step_loop_pinch(
            PinchParams(
                strength_0=exp.mechanism_params["strength_0"],
                strength_t=exp.mechanism_params["strength_t"],
                schedule=exp.schedule,
                noise=exp.noise,
            ),
            exp.total_steps,
        )

    if exp.mechanism == "bridge_across_hole":
        return step_bridge_across_hole(
            BridgeParams(
                strength_0=exp.mechanism_params["strength_0"],
                strength_t=exp.mechanism_params["strength_t"],
                schedule=exp.schedule,
                noise=exp.noise,
            ),
            exp.total_steps,
        )

    if exp.mechanism == "nonlinear_to_kplane":
        return step_nonlinear_projection(
            _proj_fn=lambda x: proj_to_k_plane(x, _k=exp.k),
            _params=NonLinearParams(
                eps_0=exp.mechanism_params["eps_0"],
                eps_t=exp.mechanism_params["eps_t"],
                relax=exp.mechanism_params["relax"],
                schedule=exp.schedule,
            ),
            _t=exp.total_steps,
        )

    if exp.mechanism == "nonlinear_to_sphere":
        return step_nonlinear_projection(
            _proj_fn=lambda x: proj_to_sphere(x, _r=1.0),
            _params=NonLinearParams(
                eps_0=exp.mechanism_params["eps_0"],
                eps_t=exp.mechanism_params["eps_t"],
                relax=exp.mechanism_params["relax"],
                schedule=exp.schedule,
            ),
            _t=exp.total_steps,
        )

    if exp.mechanism == "nonlinear_to_torus":
        return step_nonlinear_projection(
            _proj_fn=lambda x: proj_to_torus(x, _r_major=2.0, _r_minor=0.5),
            _params=NonLinearParams(
                eps_0=exp.mechanism_params["eps_0"],
                eps_t=exp.mechanism_params["eps_t"],
                relax=exp.mechanism_params["relax"],
                schedule=exp.schedule,
            ),
            _t=exp.total_steps,
        )

    if exp.mechanism == "radial_collapse":
        p = radial_params_from_severity(
            severity=exp.severity,
            schedule=exp.schedule,
            finish=exp.total_steps,
            mover_frac=exp.mover_frac,
            center_mode="centroid",
            target_radius=0.0,
            mode="contract_to_center",
            seed=exp.seed,
        )
        return step_radial(p)

    if exp.mechanism == "radial_shell_collapse":
        p = radial_params_from_severity(
            severity=exp.severity,
            schedule=exp.schedule,
            finish=exp.total_steps,
            mover_frac=exp.mover_frac,
            center_mode="centroid",
            target_radius=0.5,
            mode="to_radius",
            seed=exp.seed,
        )
        return step_radial(p)

    return step_nonlinear_projection(
        _proj_fn=lambda x: proj_to_paraboloid(x),
        _params=NonLinearParams(
            eps_0=exp.mechanism_params["eps_0"],
            eps_t=exp.mechanism_params["eps_t"],
            relax=exp.mechanism_params["relax"],
            schedule=exp.schedule,
        ),
        _t=exp.total_steps,
    )


def append_run_index(manifest: Dict[str, Any]) -> None:
    checkpoint_root = manifest.get("checkpoint_root", CHECKPOINT_ROOT)
    index_path = os.path.join(checkpoint_root, "run_index.jsonl")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    line = json.dumps({
        "run_id": manifest["run_id"],
        "run_dir": manifest["run_dir"],
        "status": manifest["status"],
        "updated_at": manifest["updated_at"],
    }) + "\n"
    with open(index_path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    fsync_parent(index_path)


def x_to_parquet(x: np.ndarray, out_path: str) -> None:
    """
    Save point cloud to parquet in a shareable columnar format.
    Stores one row per point and one column per coordinate.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected 2D array for checkpoint, got shape={x.shape}")

    out_path = str(out_path)
    columns = {f"dim_{j:04d}": x[:, j] for j in range(x.shape[1])}
    df = pd.DataFrame(columns)
    tmp_path = out_path + f".tmp.{os.getpid()}.parquet"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    fsync_parent(out_path)


class ParquetCheckpointWriter:
    """
    Drop-in-ish checkpoint writer for fragile storage.

    Per run directory layout:
      run_dir/
        manifest.json
        metadata.json
        checkpoints/
          ckpt_epoch_0000.parquet
          ckpt_epoch_0002.parquet
          ...

    The manifest is the source of truth for the run.
    """

    def __init__(
        self,
        root_dir: str,
        experiment: str,
        mechanism: str,
        model: str,
        every: int,
        overwrite: bool = False,
        extra_payload: Optional[Dict[str, Any]] = None,
    ):
        self.root_dir = root_dir
        self.experiment = experiment
        self.mechanism = mechanism
        self.model = model
        self.every = every
        self.overwrite = overwrite
        self.extra_payload = extra_payload or {}

        self.run_dir = os.path.join(root_dir, experiment, mechanism, model)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.metadata_path = os.path.join(self.run_dir, "metadata.json")

        if os.path.exists(self.run_dir) and not overwrite:
            raise FileExistsError(
                f"run directory already exists and overwrite=False: {self.run_dir}"
            )

        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.manifest = {
            "run_id": model,
            "checkpoint_root": root_dir,
            "run_dir": self.run_dir,
            "experiment": experiment,
            "mechanism": mechanism,
            "model": model,
            "checkpoint_dir": self.ckpt_dir,
            "checkpoint_every": every,
            "status": "running",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "last_epoch_written": None,
            "num_checkpoints_written": 0,
            "checkpoints": [],
        }
        atomic_write_json(self.manifest_path, self.manifest)
        atomic_write_json(self.metadata_path, self.extra_payload)
        append_run_index(self.manifest)

    def checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.ckpt_dir, f"ckpt_epoch_{epoch:04d}.parquet")

    def _record_checkpoint(self, epoch: int, x: np.ndarray) -> None:
        path = self.checkpoint_path(epoch)
        x_to_parquet(x, path)

        self.manifest["last_epoch_written"] = int(epoch)
        self.manifest["num_checkpoints_written"] += 1
        self.manifest["updated_at"] = utc_now_iso()
        self.manifest["checkpoints"].append({
            "epoch": int(epoch),
            "path": path,
            "n": int(x.shape[0]),
            "d": int(x.shape[1]),
            "written_at": utc_now_iso(),
        })
        atomic_write_json(self.manifest_path, self.manifest)

    # Expose several method names to maximize compatibility with trajectory.dynamics.
    def save(self, epoch: int, x: np.ndarray, **kwargs) -> None:
        if epoch % self.every == 0:
            self._record_checkpoint(epoch, x)

    def soft_save(self, x: np.ndarray, epoch: int, force: bool = False, **kwargs) -> None:
        """Compatibility adapter for trajectory.dynamics.

        trajectory.dynamics calls soft_save(x, epoch, force).
        Write when force=True, or when epoch matches checkpoint cadence.
        """
        if force or epoch % self.every == 0:
            self._record_checkpoint(epoch, x)

    def write(self, epoch: int, x: np.ndarray, **kwargs) -> None:
        self.save(epoch, x, **kwargs)

    def checkpoint(self, epoch: int, x: np.ndarray, **kwargs) -> None:
        self.save(epoch, x, **kwargs)

    def __call__(self, epoch: int, x: np.ndarray, **kwargs) -> None:
        self.save(epoch, x, **kwargs)

    def mark_completed(self) -> None:
        self.manifest["status"] = "completed"
        self.manifest["updated_at"] = utc_now_iso()
        atomic_write_json(self.manifest_path, self.manifest)
        append_run_index(self.manifest)

    def mark_failed(self, error: str) -> None:
        self.manifest["status"] = "failed"
        self.manifest["updated_at"] = utc_now_iso()
        self.manifest["error"] = error
        atomic_write_json(self.manifest_path, self.manifest)
        append_run_index(self.manifest)


def require_safe_checkpoint_root(path: str | os.PathLike[str]) -> None:
    """
    Refuse obviously dangerous roots, but allow local/HPC roots.

    Previously this only allowed /mnt/wd_black, which is now too brittle.
    """
    resolved = os.path.abspath(os.path.expanduser(str(path)))

    forbidden = {
        "/",
        "/home",
        os.path.expanduser("~"),
        "/tmp",
        "/var",
        "/usr",
        "/etc",
    }

    if resolved in forbidden:
        raise RuntimeError(f"Refusing unsafe checkpoint root: {resolved}")

    allowed_prefixes = [
        os.path.abspath(os.path.expanduser("~/evolve_local")),
        "/mnt/wd_black",
    ]

    for env_name in ("EVOLVE_ROOT", "EVOLVE_COLLAPSE_ROOT", "SCRATCH", "PROJECT", "SLURM_TMPDIR"):
        value = os.environ.get(env_name)
        if value:
            allowed_prefixes.append(os.path.abspath(os.path.expanduser(value)))

    if not any(resolved.startswith(prefix) for prefix in allowed_prefixes):
        raise RuntimeError(
            "Checkpoint root is outside known safe roots: "
            f"{resolved}. Set EVOLVE_ROOT or EVOLVE_COLLAPSE_ROOT intentionally."
        )

def run_experiment(exp: TrajectoryExperiment, root_dir: str = CHECKPOINT_ROOT, label_root: Optional[str] = None) -> None:
    if label_root is None:
        label_root = os.path.join(os.path.dirname(root_dir), "cluster_labels")

    model_name = make_run_stem(exp)
    run_dir = os.path.join(root_dir, "collapse_ph", exp.mechanism, model_name)
    manifest_path = os.path.join(run_dir, "manifest.json")

    # Safe skip semantics on fragile storage: if there is already a completed
    # manifest, skip. If there is a partial/failed run, leave it in place and
    # force the user to decide what to do instead of silently overwriting.
    if os.path.exists(manifest_path):
        try:
            manifest = json.loads(Path(manifest_path).read_text())
            status = manifest.get("status", "unknown")
        except Exception:
            status = "unknown"

        if status == "completed":
            print(f"skipping completed run: {run_dir}")
            return

        print(f"found non-completed existing run, refusing to overwrite automatically: {run_dir}")
        return

    if exp.base_geometry == "clustered_gaussian":
        x0, labels = make_clustered_gaussian(
            n=exp.n,
            d=exp.d,
            num_clusters=getattr(exp, "num_clusters", 4),
            seed=exp.seed,
        )
        object.__setattr__(exp, "cluster_labels", labels)
        save_cluster_labels(exp, labels, label_root)
    else:
        x0 = get_geometry(
            exp.base_geometry,
            exp.n,
            exp.d,
            _seed=exp.seed,
            _k=exp.k,
        )
        labels = None

    step_fn = build_step(exp, x0)

    extra_payload = {
        **asdict(exp),
        "base_geometry": exp.base_geometry,
        "schedule": exp.schedule,
        "severity": exp.severity,
        "mover_frac": exp.mover_frac,
        "noise": exp.noise,
        "n": exp.n,
        "d": exp.d,
        "k": exp.k,
        "seed": exp.seed,
        "created_at": utc_now_iso(),
    }
    if labels is not None:
        extra_payload["cluster_labels_path"] = os.path.join(
            label_root,
            f"{make_run_stem(exp)}_labels.npy",
        )

    writer = ParquetCheckpointWriter(
        root_dir=root_dir,
        experiment="collapse_ph",
        mechanism=exp.mechanism,
        model=model_name,
        every=exp.checkpoint_every,
        overwrite=False,
        extra_payload=extra_payload,
    )

    try:
        dynamics(
            x0,
            exp.total_steps,
            step_fn,
            exp.mover_frac,
            exp.seed,
            writer,
        )
        writer.mark_completed()
    except Exception as exc:
        writer.mark_failed(repr(exc))
        raise


def run_all(
    n: int,
    d: int,
    num_steps: int,
    checkpoint_every: int,
    seed: int,
    k: int = 8,
    root_dir: str = CHECKPOINT_ROOT,
    grid: Optional[Dict[str, Any]] = None,
):
    require_safe_checkpoint_root(root_dir)

    exps = build_experiments(
        n=n,
        d=d,
        num_steps=num_steps,
        checkpoint_every=checkpoint_every,
        seed=seed,
        k=k,
        grid=grid,
    )

    for i, exp in enumerate(exps, start=1):
        print(
            f"[{i}/{len(exps)}] "
            f"{exp.base_geometry} | {exp.mechanism} | "
            f"n={n} | d={d} | "
            f"{exp.schedule} | {exp.severity} | "
            f"mp={exp.mover_frac} | noise={exp.noise} | seed={exp.seed}"
        )
        run_experiment(exp, root_dir=root_dir)


def main():
    grid = PRIMARY_D50_GRID
    require_safe_checkpoint_root(CHECKPOINT_ROOT)
    print(f"Writing checkpoints under: {CHECKPOINT_ROOT}")

    for num_p in grid["n_values"]:
        for di in grid["d_values"]:
            for seed in grid["seeds"]:
                proj_k = int(di / 3)
                run_all(
                    n=num_p,
                    d=di,
                    num_steps=grid["num_steps"],
                    checkpoint_every=grid["checkpoint_every"],
                    seed=seed,
                    k=proj_k,
                    root_dir=CHECKPOINT_ROOT,
                    grid=grid,
                )


if __name__ == "__main__":
    main()
