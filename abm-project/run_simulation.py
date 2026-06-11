"""Reproducible driver for the three Green Stock ABM scenarios.

Running this script with the default arguments seeds each scenario with a
deterministic value derived from ``--seed`` and writes the following to
``abm-project/results/``:

- ``aggressive_history.npz``
- ``conservative_history.npz``
- ``mixed_history.npz``
- ``price_trajectories.png``
- ``belief_trajectory.png`` (agent 0 of the Mixed scenario)

The ``.npz`` files contain ``beliefs``, ``actions``, ``price``, ``theta_l``,
``theta_u``, and ``seed`` so each run can be reproduced or post-analyzed.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np

from abm import (
    GreenStockABM,
    make_aggressive,
    make_conservative,
    make_mixed,
    plot_belief_trajectory,
    plot_price_trajectories,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_SEED = 20260610
DEFAULT_N = 1000
DEFAULT_T = 50
DEFAULT_P0 = 10000.0


def _save_history(model: GreenStockABM, path: str, scenario_seed: int) -> None:
    np.savez(
        path,
        beliefs=model.beliefs,
        actions=model.actions,
        price=model.price,
        theta_l=model.theta_l,
        theta_u=model.theta_u,
        seed=np.array(scenario_seed),
        n=np.array(model.n),
        T=np.array(model.T),
        P0=np.array(model.P0),
    )


def run(seed: int = DEFAULT_SEED,
        n: int = DEFAULT_N,
        T: int = DEFAULT_T,
        P0: float = DEFAULT_P0,
        results_dir: str = RESULTS_DIR) -> Dict[str, GreenStockABM]:
    """Run the three canonical scenarios and persist the outputs."""
    os.makedirs(results_dir, exist_ok=True)

    scenarios = {
        "Aggressive": (make_aggressive, seed + 0, "aggressive_history.npz"),
        "Conservative": (make_conservative, seed + 1, "conservative_history.npz"),
        "Mixed": (make_mixed, seed + 2, "mixed_history.npz"),
    }

    models: Dict[str, GreenStockABM] = {}
    for label, (factory, scenario_seed, fname) in scenarios.items():
        print(f"[run_simulation] running {label} (seed={scenario_seed}, n={n}, T={T})")
        model = factory(seed=scenario_seed, n=n, T=T, P0=P0).run()
        models[label] = model
        history_path = os.path.join(results_dir, fname)
        _save_history(model, history_path, scenario_seed)
        print(f"[run_simulation]   saved {history_path}")

    price_path = os.path.join(results_dir, "price_trajectories.png")
    plot_price_trajectories(models, out_path=price_path)
    print(f"[run_simulation] saved {price_path}")

    belief_path = os.path.join(results_dir, "belief_trajectory.png")
    plot_belief_trajectory(models["Mixed"], agent_idx=0, out_path=belief_path,
                           title="Belief trajectory: agent 0 (Mixed scenario)")
    print(f"[run_simulation] saved {belief_path}")

    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Green Stock ABM scenarios.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Master seed; per-scenario seeds are seed+0/1/2.")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="Number of traders per scenario.")
    parser.add_argument("--T", type=int, default=DEFAULT_T,
                        help="Number of simulation time steps.")
    parser.add_argument("--P0", type=float, default=DEFAULT_P0,
                        help="Initial price.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help="Directory for .npz histories and figures.")
    args = parser.parse_args()

    run(seed=args.seed, n=args.n, T=args.T, P0=args.P0,
        results_dir=args.results_dir)


if __name__ == "__main__":
    main()
