""" Runner for forcing collapse. """
import os
import numpy as np

# from contamination_mechanic import ContaminationParams, step_with_contamination
from itertools import product

from checkpoint import CheckpointManager
from geometry import get_geometry
from experiment_dataclasses import TrajectoryExperiment
from linear_mechanic import LinearSpectralParams, step_linear_spectral
from nonlinear_mechanic import NonLinearParams, step_nonlinear_projection
from projectors import (proj_to_k_plane, proj_to_sphere,
                        proj_to_torus, proj_to_paraboloid)
from topological_mechanisms import (HoleFillParams, PinchParams, BridgeParams,
                                    step_hole_fill, step_loop_pinch,
                                    step_bridge_across_hole)
from trajectory import dynamics

EXTERNAL_ROOT = "/media/alkal/WD_BLACK/evolve_collapse"
CHECKPOINT_ROOT = os.path.join(EXTERNAL_ROOT, "evolve_checkpoints")
METRIC_ROOT = os.path.join(EXTERNAL_ROOT, "metric_outputs")
SUMMARY_ROOT = os.path.join(EXTERNAL_ROOT, "metric_summaries")
ASSET_ROOT = os.path.join(EXTERNAL_ROOT, "summary_assets")


def get_mechanism_params(_mechanism, _severity):
    """

    :param _mechanism:
    :param _severity:
    :return:
    """
    if _mechanism == "linear_to_kplane":
        if _severity == "weak":
            return {"alpha_0": 1.0, "alpha_t": 0.5}
        if _severity == "med":
            return {"alpha_0": 1.0, "alpha_t": 0.2}
        return {"alpha_0": 1.0, "alpha_t": 0.05}

    if _mechanism == "hole_fill":
        if _severity == "weak":
            return {"r0": 0.0, "rt": 0.2}
        if _severity == "med":
            return {"r0": 0.0, "rt": 0.5}
        return {"r0": 0.0, "rt": 0.8}
    if _mechanism == "loop_pinch":
        if _severity == "weak":
            return {"strength_0": 0.0, "strength_t": 0.3}
        if _severity == "med":
            return {"strength_0": 0.0, "strength_t": 0.6}
        return {"strength_0": 0.0, "strength_t": 1.0}
    if _mechanism == "bridge_across_hole":
        if _severity == "weak":
            return {"strength_0": 0.0, "strength_t": 0.3}
        if _severity == "med":
            return {"strength_0": 0.0, "strength_t": 0.6}
        return {"strength_0": 0.0, "strength_t": 1.0}

    if _severity == "weak":
        return {"eps_0": 0.5, "eps_t": 0.05, "relax": 0.2}
    if _severity == "med":
        return {"eps_0": 0.5, "eps_t": 0.02, "relax": 0.5}
    return {"eps_0": 0.5, "eps_t": 0.005, "relax": 1.0}


def build_experiments(_n, _d, _num_steps, _checkpoint_every, _seed, _k):
    """

    :param _n:
    :param _d:
    :param _num_steps:
    :param _checkpoint_every:
    :param _seed:
    :param _k:
    :return:
    """
    geometries = [
        "kcube",
        #"kplane",
        #"sphere",
        #"torus",
        #"swiss",
        #"paraboloid",
        "spiked_gaussian",
    ]

    mechanisms = [
        "linear_to_kplane",
        "nonlinear_to_kplane",
        #"nonlinear_to_sphere",
        #"nonlinear_to_torus",
        #"nonlinear_to_paraboloid",
        #"hole_fill"
    ]

    schedules = ["linear", "exponential", "sigmoid"]
    severities = ["weak", "moderate", "strong"]
    mover_fracs = [0.25, 0.5, 1.0]
    noises = [0.0, 0.1, 0.2, 0.3]

    exps = []
    for geom, mech, sched, sev, mp, noise in product(
        geometries,
        mechanisms,
        schedules,
        severities,
        mover_fracs,
        noises,
    ):
        exps.append(
            TrajectoryExperiment(
                base_geometry=geom,
                mechanism=mech,
                n=_n,
                d=_d,
                k=_k,
                total_steps=_num_steps,
                checkpoint_every=_checkpoint_every,
                seed=_seed,
                mover_frac=mp,
                noise=noise,
                schedule=sched,
                severity=sev,
                mechanism_params=get_mechanism_params(mech, sev),
            )
        )
    return exps


def build_step(_exp):
    """

    :param _exp:
    :return:
    """
    if _exp.mechanism == "linear_to_kplane":
        return step_linear_spectral(
            LinearSpectralParams(
                k=_exp.k,
                alpha_0=_exp.mechanism_params["alpha_0"],
                alpha_t=_exp.mechanism_params["alpha_t"],
                noise=_exp.noise,
                schedule=_exp.schedule,
            ),
            _exp.total_steps,
        )
    if _exp.mechanism == "hole_fill":
        return step_hole_fill(
            HoleFillParams(
                r0=_exp.mechanism_params["r0"],
                rt=_exp.mechanism_params["rt"],
                schedule=_exp.schedule,
                noise=_exp.noise,
            ),
            _exp.total_steps,
        )

    if _exp.mechanism == "loop_pinch":
        return step_loop_pinch(
            PinchParams(
                strength_0=_exp.mechanism_params["strength_0"],
                strength_t=_exp.mechanism_params["strength_t"],
                schedule=_exp.schedule,
                noise=_exp.noise,
            ),
            _exp.total_steps,
        )

    if _exp.mechanism == "bridge_across_hole":
        return step_bridge_across_hole(
            BridgeParams(
                strength_0=_exp.mechanism_params["strength_0"],
                strength_t=_exp.mechanism_params["strength_t"],
                schedule=_exp.schedule,
                noise=_exp.noise,
            ),
            _exp.total_steps,
        )

    if _exp.mechanism == "nonlinear_to_kplane":
        return step_nonlinear_projection(
            _proj_fn=lambda x: proj_to_k_plane(x, _k=_exp.k),
            _params=NonLinearParams(
                eps_0=_exp.mechanism_params["eps_0"],
                eps_t=_exp.mechanism_params["eps_t"],
                relax=_exp.mechanism_params["relax"],
                schedule=_exp.schedule,
            ),
            _t=_exp.total_steps,
        )

    if _exp.mechanism == "nonlinear_to_sphere":
        return step_nonlinear_projection(
            _proj_fn=lambda x: proj_to_sphere(x, _r=1.0),
            _params=NonLinearParams(
                eps_0=_exp.mechanism_params["eps_0"],
                eps_t=_exp.mechanism_params["eps_t"],
                relax=_exp.mechanism_params["relax"],
                schedule=_exp.schedule,
            ),
            _t=_exp.total_steps,
        )

    if _exp.mechanism == "nonlinear_to_torus":
        return step_nonlinear_projection(
            _proj_fn=lambda x: proj_to_torus(x, _r_major=2.0, _r_minor=0.5),
            _params=NonLinearParams(
                eps_0=_exp.mechanism_params["eps_0"],
                eps_t=_exp.mechanism_params["eps_t"],
                relax=_exp.mechanism_params["relax"],
                schedule=_exp.schedule,
            ),
            _t=_exp.total_steps,
        )

    return step_nonlinear_projection(
        _proj_fn=lambda x: proj_to_paraboloid(x),
        _params=NonLinearParams(
            eps_0=_exp.mechanism_params["eps_0"],
            eps_t=_exp.mechanism_params["eps_t"],
            relax=_exp.mechanism_params["relax"],
            schedule=_exp.schedule,
        ),
        _t=_exp.total_steps,
    )

def run_experiment(_exp, _root_dir="evolve_checkpoints"):
    """

    :param _exp:
    :return:
    """
    x0 = get_geometry(_exp.base_geometry,
                      _exp.n,
                      _exp.d,
                      _seed=_exp.seed,
                      _k=_exp.k)

    step_fn = build_step(_exp)

    model_name = (
        f"{_exp.base_geometry}"
        f"_n{_exp.n}"
        f"_d{_exp.d}"
        f"_k{_exp.k}"
        f"__{_exp.schedule}"
        f"__{_exp.severity}"
        f"__mp{_exp.mover_frac}"
        f"__noise{_exp.noise}"
        f"__seed{_exp.seed}"
    )

    root_dir = _root_dir
    experiment = "collapse_ph"
    mechanism = _exp.mechanism
    run_dir = os.path.join(root_dir, experiment, mechanism, model_name)

    if os.path.exists(run_dir):
        print(f"skipping existing run: {run_dir}")
        return

    ckpt = CheckpointManager(
        _root_dir=_root_dir,
        _experiment="collapse_ph",
        _mechanism=_exp.mechanism,
        _model=model_name,
        _every=_exp.checkpoint_every,
        _overwrite=False
    )

    dynamics(x0,
             _exp.total_steps,
             step_fn,
             _exp.mover_frac,
             _exp.seed,
             ckpt)


def run_all(_n, _d, _num_steps, _checkpoint_every,
            _seed=None, _k=8, _root_dir="evolve_checkpoints"):
    """

    :param _n:
    :param _d:
    :param _num_steps:
    :param _checkpoint_every:
    :param _seed:
    :param _k:
    :return:
    """
    exps = build_experiments(_n,
                             _d,
                             _num_steps,
                             _checkpoint_every,
                             _seed,
                             _k)

    for i, exp in enumerate(exps, start=1):
        print(f"[{i}/{len(exps)}] "
              f"{exp.base_geometry} | {exp.mechanism} | "
              f"{exp.schedule} | {exp.severity} | "
              f"mp={exp.mover_frac} | noise={exp.noise}")
        run_experiment(exp, _root_dir)


if __name__ == "__main__":
    for np in [1000, 2000, 5000]:
        for di in [50, 100]:
            proj_k = int(di / 3)
            run_all(np, di, 50, 2,
                    _seed=17, _k=proj_k, _root_dir=CHECKPOINT_ROOT)
