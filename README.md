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

This repository release is stable for the benchmark pipeline, both local and larger-scale runs.
Continued expansions using optimal transport metrics, distance-to-measure filtrations, and 
additional validation assets are under development.

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
python run_cloud_manifest.py

# 2. Measure checkpoints in parallel.
python run_measurement_tranched.py

# 3. Summarize metric outputs.
python scripts/summarize_metric_results.py

# 4. Compute detection-time summaries.
python analysis/compute_detection_times.py

# 5. Generate paper-facing artifacts.
python scripts/make_canonical_detection_artifacts.py
python scripts/make_canonical_fidelity_comparison.py
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

For persistent-homology workflows, the benchmark also tracks approximation 
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

## Smoke test

After installing the dependencies, run the minimal end-to-end smoke test:

```bash
python scripts/smoke_test.py
```

A formal citation will be added once the associated benchmark paper and stable 
artifact release are available.

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



## ACCESS Migration

```angular2html
mkdir -p ~/evolve_local/evolve_collapse
mkdir -p ~/evolve_local/evolve_collapse/evolve_checkpoints
mkdir -p ~/evolve_local/evolve_collapse/metric_outputs
mkdir -p ~/evolve_local/evolve_collapse/logs

export EVOLVE_ROOT="$HOME/evolve_local/evolve_collapse"
```

```angular2html
python run_parquet_mainfest.py \
  --root-dir "$EVOLVE_ROOT/evolve_checkpoints" \
  --tranche canonical
```

```angular2html
python measure_checkpoints_parallel_parquet_tranched.py \
  --root-dir "$EVOLVE_ROOT/evolve_checkpoints" \
  --out-dir "$EVOLVE_ROOT/metric_outputs" \
  --ph-mode online_landmark_dynamic_support \
  --tranche canonical \
  --workers 2 \
  --no-csv
```
