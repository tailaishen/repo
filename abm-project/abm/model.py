"""Green Stock agent-based model.

The ``GreenStockABM`` class owns a population of ``Agent`` traders, the
information signals ``I(t)`` and ``N(t)``, the price series ``P(t)``, and the
seeded RNG. Scenario factories (``make_aggressive``, ``make_conservative``,
``make_mixed``) configure the threshold sampler for the three simulation
scenarios described in the prompt.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from .agent import Agent


ThresholdSampler = Callable[
    [np.random.Generator, int], Tuple[np.ndarray, np.ndarray]
]


def default_info_fn(t: int) -> float:
    """Positive green-energy information signal ``I(t) = 0.2 + 0.05 t``."""
    return 0.2 + 0.05 * t


def default_noise_fn(t: int) -> float:
    """Negative green-energy information signal ``N(t)``.

    ``N(t) = (1 + 0.3 sin(pi t / 10)) / 2`` per the prompt.
    """
    return (1.0 + 0.3 * math.sin(math.pi * t / 10.0)) / 2.0


class GreenStockABM:
    """Agent-based model of green-energy stock price formation.

    The model preallocates history arrays of shape ``(T+1, n)`` for beliefs
    and actions, and ``(T+1,)`` for the price, with the row ``t = 0`` storing
    the initial state. Calling :meth:`run` fills rows ``1 .. T``.
    """

    def __init__(
        self,
        n: int = 1000,
        T: int = 50,
        P0: float = 10000.0,
        gamma: float = 0.9,
        delta: float = 0.3,
        sigma: float = 0.02,
        threshold_sampler: Optional[ThresholdSampler] = None,
        info_fn: Callable[[int], float] = default_info_fn,
        noise_fn: Callable[[int], float] = default_noise_fn,
        seed: Optional[int] = None,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma must lie in [0, 1)")
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative")

        self.n = n
        self.T = T
        self.P0 = float(P0)
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma
        self.info_fn = info_fn
        self.noise_fn = noise_fn
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if threshold_sampler is None:
            threshold_sampler = uniform_threshold_sampler(
                theta_u_low=0.2, theta_u_high=0.4,
                theta_l_low=-0.8, theta_l_high=-0.6,
            )
        self._threshold_sampler = threshold_sampler

        self.beliefs = np.zeros((T + 1, n), dtype=float)
        self.actions = np.zeros((T + 1, n), dtype=np.int8)
        self.price = np.zeros(T + 1, dtype=float)
        self.price[0] = self.P0

        self.theta_l, self.theta_u = self._threshold_sampler(self.rng, n)
        self._validate_thresholds()

        self.agents = [
            Agent(
                theta_l=float(self.theta_l[i]),
                theta_u=float(self.theta_u[i]),
                gamma=gamma,
                delta=delta,
                sigma=sigma,
            )
            for i in range(n)
        ]

    def _validate_thresholds(self) -> None:
        """Ensure thresholds satisfy ``-1 < theta_l < 0 < theta_u < 1``."""
        if self.theta_l.shape != (self.n,) or self.theta_u.shape != (self.n,):
            raise ValueError("threshold sampler returned wrong-shaped arrays")
        if not np.all((self.theta_l > -1.0) & (self.theta_l < 0.0)):
            raise ValueError("theta_l must lie in (-1, 0)")
        if not np.all((self.theta_u > 0.0) & (self.theta_u < 1.0)):
            raise ValueError("theta_u must lie in (0, 1)")

    def step(self, t: int) -> None:
        """Advance the model from time ``t`` to ``t + 1``.

        Reads `I(t)` and `N(t)`, updates every agent's belief and action, then
        applies ``P(t+1) = P(t) + sum_i B_i(t+1)``.
        """
        I_t = self.info_fn(t)
        N_t = self.noise_fn(t)
        beliefs_row = self.beliefs[t + 1]
        actions_row = self.actions[t + 1]
        for i, agent in enumerate(self.agents):
            beliefs_row[i] = agent.update_belief(I_t, N_t, self.rng)
            actions_row[i] = agent.take_action()
        self.price[t + 1] = self.price[t] + int(actions_row.sum())

    def run(self) -> "GreenStockABM":
        """Run the full simulation for ``T`` steps and return ``self``."""
        for t in range(self.T):
            self.step(t)
        return self

    def aggregate_behavior(self) -> np.ndarray:
        """Return ``B_total(t)`` for ``t = 0 .. T`` as a 1D array."""
        return self.actions.sum(axis=1)


def uniform_threshold_sampler(
    theta_u_low: float,
    theta_u_high: float,
    theta_l_low: float,
    theta_l_high: float,
) -> ThresholdSampler:
    """Return a sampler that draws thresholds i.i.d. from two uniforms."""

    def sample(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
        theta_l = rng.uniform(theta_l_low, theta_l_high, size=n)
        theta_u = rng.uniform(theta_u_low, theta_u_high, size=n)
        return theta_l, theta_u

    return sample


def mixed_threshold_sampler() -> ThresholdSampler:
    """Half-aggressive, half-conservative threshold sampler.

    Each agent is independently assigned to the aggressive or conservative
    profile with probability ``0.5``. Thresholds are then drawn from the
    corresponding uniform distributions.
    """
    aggressive = uniform_threshold_sampler(
        theta_u_low=0.2, theta_u_high=0.4,
        theta_l_low=-0.8, theta_l_high=-0.6,
    )
    conservative = uniform_threshold_sampler(
        theta_u_low=0.6, theta_u_high=0.8,
        theta_l_low=-0.4, theta_l_high=-0.2,
    )

    def sample(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
        theta_l_agg, theta_u_agg = aggressive(rng, n)
        theta_l_con, theta_u_con = conservative(rng, n)
        is_aggressive = rng.random(n) < 0.5
        theta_l = np.where(is_aggressive, theta_l_agg, theta_l_con)
        theta_u = np.where(is_aggressive, theta_u_agg, theta_u_con)
        return theta_l, theta_u

    return sample


def make_aggressive(
    seed: Optional[int] = None,
    n: int = 1000,
    T: int = 50,
    P0: float = 10000.0,
) -> GreenStockABM:
    """Aggressive buyers: low buying thresholds, high selling thresholds."""
    return GreenStockABM(
        n=n,
        T=T,
        P0=P0,
        seed=seed,
        threshold_sampler=uniform_threshold_sampler(
            theta_u_low=0.2, theta_u_high=0.4,
            theta_l_low=-0.8, theta_l_high=-0.6,
        ),
    )


def make_conservative(
    seed: Optional[int] = None,
    n: int = 1000,
    T: int = 50,
    P0: float = 10000.0,
) -> GreenStockABM:
    """Conservative buyers: high buying thresholds, low selling thresholds."""
    return GreenStockABM(
        n=n,
        T=T,
        P0=P0,
        seed=seed,
        threshold_sampler=uniform_threshold_sampler(
            theta_u_low=0.6, theta_u_high=0.8,
            theta_l_low=-0.4, theta_l_high=-0.2,
        ),
    )


def make_mixed(
    seed: Optional[int] = None,
    n: int = 1000,
    T: int = 50,
    P0: float = 10000.0,
) -> GreenStockABM:
    """Mixed buyers: random half-aggressive, half-conservative assignment."""
    return GreenStockABM(
        n=n,
        T=T,
        P0=P0,
        seed=seed,
        threshold_sampler=mixed_threshold_sampler(),
    )
