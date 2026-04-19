# EvolveManifold Experiment Plan

The goal is to:

- Ensure **numerical stability**
- Validate **new mechanisms (radial + cluster)**
- Produce **interpretable visualizations**
- Build toward **benchmark-quality results**

---

# Overview

Experiments happen in **four phases**:

1. **Stability Calibration**
2. **Schedule Calibration**
3. **Benchmark Runs (with baselines)**
4. **Topology Experiments**

---

# Phase 1: Stability Calibration

## Goal
Verify that new mechanisms behave numerically and visually under controlled conditions.

## Geometry
- ✅ `clustered_gaussian` (preferred)
- ❌ Avoid `kcube` initially (UMAP + high-d issues)

## Mechanisms (in order)

1. `radial_collapse`
2. `radial_shell_collapse`
3. `cluster_tightening`
4. `cluster_merging` (only if others are stable)

## Parameters

| Parameter       | Value                |
|----------------|----------------------|
| schedule       | `linear`             |
| severity       | `weak`               |
| mover_frac     | `0.25`, `0.5`, `1.0` |
| noise          | `0.0`                |
| seeds          | `1`                  |
| n              | `1000`               |
| d              | `50`                 |
| k              | `~16` (dim/3)        |
| steps          | `50`                 |

## Checks

- No exploding values (min/max/std stable)
- No NaN / Inf values
- PCA plots show structure (not a single point)
- Labels align with visible clusters

---

# Phase 2: Schedule Calibration

## Goal
Determine which schedules produce stable and interpretable dynamics.

## Compare schedules

- `linear` (baseline)
- `sigmoid`
- `exponential` (last)

## Use only stable mechanisms from Phase 1:

- `radial_collapse`
- `cluster_tightening`

## Keep all other parameters fixed.

## Output

Build a table like:

| Mechanism          | Schedule     | Stability | Notes |
|-------------------|-------------|----------|------|
| radial_collapse   | linear      | ✅        | stable |
| radial_collapse   | exponential | ❌        | explodes |
| cluster_tightening| sigmoid     | ✅        | smooth |

---

# Phase 3: Controlled Benchmark Runs

## Goal
Compare new mechanisms against known baselines.

## Geometries

- `clustered_gaussian` ✅
- `kcube`
- `kplane`

## Mechanisms

- `linear_to_kplane` (baseline)
- `radial_collapse`
- `cluster_tightening`

## Parameters

| Parameter       | Values                |
|----------------|----------------------|
| schedule       | best from Phase 2 + `linear` |
| severity       | `weak`, `moderate`   |
| mover_frac     | `0.25`, `0.5`        |
| noise          | `0.0`, `0.1`         |
| seeds          | `3`                  |

## Outputs

- Metric CSVs
- Summary tables
- PCA visualizations (primary)
- UMAP (secondary, after PCA pre-reduction)

---

# Phase 4: Topology Experiments

## Goal
Evaluate topological sensitivity (PH metrics vs spectral metrics).

## Geometries

- `torus`
- `swiss`
- `kcube`

## Mechanisms

- `hole_fill`
- `loop_pinch`
- `bridge_across_hole`

## Notes

- Run separately from collapse experiments
- Focus on PH metrics (Betti curves, persistence)

---

# Visualization Guidelines

## Preferred pipeline

### PCA (primary)

- Use **global PCA alignment**
- Compare across epochs directly

### UMAP (secondary)

Always use:

```python
x = sanitize_points(x)
x = standardize_points(x)
x = PCA(n_components=15).fit_transform(x)
z = UMAP(n_neighbors=100).fit_transform(x)