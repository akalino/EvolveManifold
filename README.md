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

# Expansion Roadmap

- Scale up ambient dimension and sample size, run calibration pipeline
- Track collapse dynamically on snapshots along controlled collapse trajectories

## Idea

Two objectives:

### Objective 1

See if the mechanism taxonomy stats consistent with the original results.
Questions:

- Does DTM continue to improve robustness relative to VR?
- Is MTE still more sensitive than TP arcoss mechanisms?
- Does calibration remain near nominal over the expanded null suite?
- Does power improve or degrade in a predictive fashion w.r.t $n, d, \varepsilon$?

### Objective 2

Instead of estimating only on the `bookends', we can sample along the collapse trajectory

$X^{(0)}, X^{(1)}, \ldots, X^{(T)}$

where $X^{(0)}$ is healthy and $X^{(T)}$ is the final collapsed state.
Questions:

- How early in the process does the PH summary detect collapse?
- Which summaries respond monotonically (or near monotonically)?
- Do the PH summaries fire before the standard spectral metrics?

## Experiments

### Scaling 

We keep the same mechanisms as the original paper:
- Mechanism A: linear/spectral collapse,
- Mechanism B: nonlinear support collapse, and
- Mechanism C: contamination.

We use the same PH filtrations:
- VR
- DTM
- Consider witness complex again for speed/robustness comparison.

We can expand the PH metrics:
- Total persistence (TP)
- Mean tail excess (MTE)
- Max persistence (MP)
- Top five persistence (TFP)
- Betti curve area (BCA)
- Betti curve peak (BCP)
- Betti curve delta (BCD)

still computed over homology dimensions $q \in \{0,1,2\}.

Expansions to point cloud sizes:
- Initial aim:
-- $n \in \{250, 500, 1000, 2000, 5000\}$
-- $d \in \{25, 50, 100, 200, 300\}$
-- stress test at $n=10000, d \in \{200, 500\}

For each mechanism class and null, start with 100 replications, then progress to 500 for finalization.

For each alternative collapse condition $\theta$, estimate the power
Compute a regression on performance as a function

$\widehat{\pi}(\theta) \sim f(n,d,\varepsilon,\text{mechanism, filtration, metrics})$

### Snapshot / trajectory study

For each geometry family, define a trajectory

$$
X^{(t)} = \Phi_t(X^{(0)}), \qquad t=0,1,\dots,T,
$$

where $\Phi_0$ is the identity and $\Phi_T$ produces the target collapsed configuration.

#### Linear collapse

Let $X^{(0)} \subset \mathbb{R}^d$ be healthy. Define

$$
X^{(t)} = A_t X^{(0)},
$$

where $A_t$ gradually suppresses variance along selected directions.

For example,

$$
A_t = \mathrm{diag}(1,\dots,1,\lambda_t,\dots,\lambda_t),
\qquad
\lambda_t \downarrow 0.
$$

#### Nonlinear-support collapse

Map points toward a lower-dimensional nonlinear set $M$ by

$$
X^{(t)} = (1-\gamma_t)X^{(0)} + \gamma_t \,\Pi_M(X^{(0)}),
$$

where $\Pi_M$ is a projection or nearest-point map onto $M$, and

$$
0=\gamma_0 < \gamma_1 < \cdots < \gamma_T = 1.
$$

#### Contamination / heterogeneity collapse

Increase contamination or mixture imbalance over time:

$$
X^{(t)} \sim (1-\rho_t)P_{\mathrm{healthy}} + \rho_t P_{\mathrm{contam}},
\qquad
\rho_t \uparrow 1.
$$

### Detection-time metrics

Define the first detection time of a test statistic $T$ as

$$
\tau_{\mathrm{det}}(T)
=
\min\{t : T(X^{(t)}) > c_\alpha(T;N)\},
$$

with $\tau_{\mathrm{det}}(T)=\infty$ if no detection occurs.

To compare methods, summarize:

$$
\mathbb{E}[\tau_{\mathrm{det}}(T)],
\qquad
\Pr\big(\tau_{\mathrm{det}}(T) < \infty\big),
$$

and the distribution of $\tau_{\mathrm{det}}(T)$ across replicates.

A useful normalized version is

$$
\widetilde{\tau}_{\mathrm{det}}(T)
=
\frac{\tau_{\mathrm{det}}(T)}{T},
$$

so values lie in $[0,1]$.

### Monotonicity and smoothness

To understand whether a statistic behaves predictably during collapse, measure whether $T_t$ is approximately monotone in $t$.

Possible diagnostics:

- Spearman correlation between $t$ and $T_t$,
- number of sign changes in first differences,
- total variation

$$
\mathrm{TV}(T_\bullet)=\sum_{t=1}^{T} |T_t - T_{t-1}|.
$$

This helps distinguish stable early-warning signals from noisy endpoint-only detectors.

