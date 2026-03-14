# Forcing Collapse via Evolution

Generating synthetic point-clouds and applying "collapse trajectories"
that are applied in an "evolution-like" process. 
This forces a variety of geometric collapses over time, providing
checkpoints that can be loaded to apply statistical persistent homology metrics to.

Currently, the main objective is to apply collapse mechanisms in order to create checkpoints (plk files)
for later analysis.

We split the experiments into six main categories (presented below):
starting geometry, collapse mechanism, collapse schedule, collapse severity, fraction of moved points, and noise.

## Geometry

- kcube
- kplane
- sphere
- torus
- swiss roll
- paraboloid
- spiked gaussian

for a total of (7) to test. 

## Collapse Mechanism

- linear to kplane
- nonlinear to kplane
- nonlinear to sphere
- nonlinear to sphere
- nonlinear to torus
- nonlinear to paraboloid

for a total of (6) to test.

TODO: radial collapse, cluster collapse (select anchors and provide attraction)

## Schedule Types

- linear
- exponential
- sigmoid/delayed

for a total of (3) to test.

## Severity Parameters

- total steps
- initial collapse strength
- final collapse strength
- type of schedule

Severity should be defined per each mechanism type:

Linear endpoint shrinkage:
- weak alpha_t: 0.5
- med alpha_t = 0.2
- strong alpha_t = 0.05

Nonlinear projections:
- weak relax=0.2, eps_t=0.05
- med relax=0.5, eps_t=0.02
- string relax=1.0, eps_t=0.005

## Points Moved

- Self-explanatory, just sweep
- try {0.25, 0.5, 1.0} as an initial small test on synthetics

for a total of (3) to test.

## Noise

- Also self-explanatory, try {0.0, 0.005, 0.01, 0.03, 0.05}

for a total of (5) to test.


# Total Checkpoints (at a single seed)

Thus far, (5 x 3 x 3 x 3 x x 6 x 7) = 5,670.
