# EvolveManifold

Synthetic evolving point-cloud benchmarks for studying geometric, spectral, and 
topological signatures of collapse.

`EvolveManifold` generates controlled point-cloud trajectories in which a 
known geometry is progressively transformed by a known collapse mechanism. 
These trajectories can then be measured with geometric, spectral, and 
persistent-homology-based metrics to study which detectors respond earliest, 
most reliably, and most specifically to different forms of collapse.

The project is designed around a parquet-first workflow:

1. Generate synthetic point-cloud checkpoints.
2. Store checkpoints and metadata in a manifest-driven directory layout.
3. Measure collapse metrics over each trajectory.
4. Aggregate metric outputs.
5. Generate tables and figures for analysis and reporting.

The motivating use case is collapse detection in high-dimensional representations, 
including neural-network and language-model hidden states, but the benchmark 
itself is intentionally synthetic and controlled.

## Repository status

This repository is under active development. The current focus is cleaning and 
stabilizing the benchmark pipeline for reproducible local and larger-scale runs.

## Core idea

A benchmark run consists of:

* a starting geometry, such as an isotropic cloud, clustered Gaussian, torus, sphere, cube, or spiked Gaussian;
* a collapse mechanism, such as projection, radial collapse, cluster tightening, cluster merging, or hole filling;
* a schedule controlling how collapse severity changes over time;
* a sequence of checkpoints;
* a collection of metrics computed at each checkpoint.

The result is a controlled trajectory where the ground-truth collapse process 
is known. 
This makes it possible to compare metric families by detection time, robustness, 
and failure mode.

## Canonical workflow

The intended pipeline is:

```bash
# 1. Generate parquet checkpoints and a manifest.
python run_parquet_manifest.py

# 2. Measure checkpoints in parallel.
python measure_checkpoints_parallel_parquet.py

# 3. Summarize metric outputs.
python summarize_metric_results.py

# 4. Compute detection-time summaries.
python compute_detection_times.py

# 5. Generate paper-facing artifacts.
python make_canonical_detection_artifacts.py
python make_canonical_fidelity_comparison.py
```

## Expected output structure

A typical run should produce checkpoint directories, per-run metric outputs, 
combined metric tables, and paper-facing artifacts.


Output categories:

```text
checkpoints/          # generated parquet checkpoint trajectories
metric_outputs/      # per-run metric outputs
metric_summaries/    # aggregate metric summaries
figures/             # curated paper-facing figures
tables/              # curated paper-facing tables
logs/                # run logs
```

Large benchmark assets should be treated as external artifacts rather than 
ordinary source files.

## Important scripts

### Checkpoint generation

`run_parquet_manifest.py` is the canonical checkpoint-generation entry point.

It is responsible for generating synthetic trajectories, writing checkpoint files,
and recording metadata needed for downstream measurement.

### Metric measurement

`measure_checkpoints_parallel_parquet.py` is the canonical metric runner.

It discovers runs from a manifest, measures each checkpoint trajectory, 
skips completed outputs when possible, and writes per-run metric files. 
The output format is parquet.

### Tranche definitions

`measurement_tranched.py` defines benchmark tranches or slices used to organize 
larger experiments.

Tranches are useful when the full benchmark grid is too large to run in a single 
local pass.
We are grateful to the NSF ACCESS program that has allowed for scaling to build 
meaningful high-dimensional benchmarks.

### Metric summaries

`summarize_metric_results.py` aggregates per-run metric outputs into summary 
tables.

### Detection-time analysis

`compute_detection_times.py` and `detection_time.py` compute when each metric 
detects a controlled collapse event under a specified thresholding or comparison 
rule.

### Persistent-homology workflow

`ph_workflow.py` contains persistent-homology workflow modes, including full 
Vietoris--Rips, landmark, skip, fixed-support, k-nearest-neighbor, and 
event-driven variants.

This file is central to experiments comparing scalable persistent-homology 
approximations.

### Paper artifacts

The paper-facing artifact scripts generate curated tables and figures from 
measured outputs.

Current canonical artifact scripts include:

```text
make_canonical_detection_artifacts.py
make_canonical_fidelity_comparison.py
generate_trajectory_panels.py
make_figure2_trajectories.py
```

Figure-generation scripts may eventually move under `scripts/figures/`.

## Experiment plan

The active experiment plan is documented in:

```text
EXPERIMENTS.md
```

The benchmark is organized around several experiment families:

1. Stability calibration.
2. Schedule calibration.
3. Controlled benchmark runs.
4. Topology-specific experiments.
5. Persistent-homology workflow comparisons.

The main benchmark questions are:

* Which metric families detect collapse earliest?
* Which metrics are robust to noise?
* Which metrics are specific to particular collapse mechanisms?
* Which persistent-homology approximations preserve enough signal to be useful at scale?
* Which workflows are feasible for high-dimensional, high-cardinality point clouds?

## Metric families

The benchmark compares several broad metric families.

### Geometric metrics

These measure changes in distances, local structure, cluster separation, 
covariance geometry, or intrinsic dimension.

### Spectral metrics

These measure changes in covariance spectra, anisotropy, effective rank, 
or singular-value structure.

### Topological metrics

These measure changes in persistent-homology summaries, including Betti curves, 
persistence diagrams, and derived statistics.

### Workflow diagnostics

For persistent-homology workflows, the benchmark may also track approximation 
fidelity, support stability, edge precision, and event-trigger behavior.


## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the repository is later converted into an installable package, 
the preferred development install will be:

```bash
pip install -e .
```

## Development conventions

The preferred docstring style is reStructuredText:

```python
def example_metric(x):
    """Compute an example collapse metric.

    :param x: Point cloud with shape ``(n_points, ambient_dim)``.
    :return: Scalar metric value.
    """
```

Explicit typing should be used sparingly. 
Type annotations are appropriate for dataclass fields, unclear interfaces, 
and places where they improve maintainability. 
Ordinary internal functions should prefer clear names and reStructuredText 
docstrings.


## Future structure

The repository should be reorganized:

```text
evolve_manifold/
  geometry.py
  trajectory.py
  checkpoint.py
  metrics.py
  ph_workflow.py
  mechanisms/
  io/
  analysis/

scripts/
  generate_checkpoints.py
  measure_checkpoints.py
  summarize_results.py
  figures/

tests/
figures/
tables/
```

For now, the highest priority is keeping the canonical workflow stable before 
doing a large package migration.

## Citation and reproducibility

This repository is intended to support reproducible benchmark experiments. 
When using results from this repository, record:

* repository URL;
* branch name;
* commit hash;
* experiment tranche;
* checkpoint-generation configuration;
* metric-measurement configuration;
* persistent-homology workflow mode;
* output artifact path.

A formal citation will be added once the associated benchmark paper and stable 
artifact release are available.

