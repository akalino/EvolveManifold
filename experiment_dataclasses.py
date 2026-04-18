from dataclasses import dataclass

@dataclass(frozen=True)
class TrajectoryExperiment:
    base_geometry: str
    mechanism: str
    n: int
    d: int
    k: int
    total_steps: int
    checkpoint_every: int
    seed: int
    mover_frac: float
    noise: float
    schedule: str
    severity: str
    mechanism_params: dict
