"""Individual trader agent for the Green Stock ABM.

An ``Agent`` keeps its own latent state ``x``, derived belief ``b = tanh(x)``,
and personal thresholds ``theta_l, theta_u``. All randomness is owned by the
caller (the model) via a ``numpy.random.Generator`` passed into
``update_belief``, so simulations are reproducible from a single seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Agent:
    """A single trader in the Green Stock ABM.

    Attributes:
        theta_l: Selling threshold in ``(-1, 0)``. Agent sells when ``b <= theta_l``.
        theta_u: Buying threshold in ``(0, 1)``. Agent buys when ``b > theta_u``.
        gamma: Belief retention rate.
        delta: Information sensitivity.
        sigma: Standard deviation of the per-step belief noise.
        x: Latent belief state. Initialized to 0 so ``b = tanh(0) = 0``.
        b: Current belief, equals ``tanh(x)``; lies in ``(-1, 1)``.
    """

    theta_l: float
    theta_u: float
    gamma: float = 0.9
    delta: float = 0.3
    sigma: float = 0.02
    x: float = 0.0
    b: float = 0.0

    def update_belief(self, I_t: float, N_t: float, rng: np.random.Generator) -> float:
        """Advance belief one step using the spec's recursion.

        Computes (in order):
            eps_t = N(0, sigma^2)
            x(t+1) = gamma * x(t) + delta * (I(t) - N(t)) + eps_t
            b(t+1) = tanh(x(t+1))

        Returns the new belief ``b(t+1)``.
        """
        eps_t = rng.normal(0.0, self.sigma) if self.sigma > 0.0 else 0.0
        self.x = self.gamma * self.x + self.delta * (I_t - N_t) + eps_t
        self.b = math.tanh(self.x)
        return self.b

    def take_action(self) -> int:
        """Return the trader's action ``B_i(t) in {-1, 0, 1}``.

        Implements the spec's piecewise rule exactly, including the ``<=``
        boundary on the sell side:
            +1 if b > theta_u
            -1 if b <= theta_l
             0 otherwise
        """
        if self.b > self.theta_u:
            return 1
        if self.b <= self.theta_l:
            return -1
        return 0
