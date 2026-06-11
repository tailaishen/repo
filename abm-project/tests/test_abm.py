"""Test suite for the Green Stock ABM.

These tests verify that beliefs and actions stay in bounds, that thresholds
are sampled from the correct ranges, that the price update rule matches the
specification, and that simulations are reproducible from a single seed.
Two reduced-model tests additionally save figures into
``results/tests/`` so the simplified dynamics can be inspected by eye.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from abm import (
    GreenStockABM,
    make_aggressive,
    make_conservative,
    make_mixed,
    plot_belief_trajectory,
    plot_price_trajectories,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS_TESTS_DIR = os.path.join(ROOT, "results", "tests")


@pytest.fixture(scope="session", autouse=True)
def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_TESTS_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def aggressive_model() -> GreenStockABM:
    return make_aggressive(seed=123, n=50, T=30).run()


@pytest.fixture(scope="module")
def conservative_model() -> GreenStockABM:
    return make_conservative(seed=124, n=50, T=30).run()


@pytest.fixture(scope="module")
def mixed_model() -> GreenStockABM:
    return make_mixed(seed=125, n=80, T=30).run()


def test_beliefs_in_bounds(aggressive_model, conservative_model, mixed_model):
    for model in (aggressive_model, conservative_model, mixed_model):
        assert np.all(model.beliefs > -1.0)
        assert np.all(model.beliefs < 1.0)


def test_actions_in_set(aggressive_model, conservative_model, mixed_model):
    for model in (aggressive_model, conservative_model, mixed_model):
        unique_actions = set(np.unique(model.actions).tolist())
        assert unique_actions.issubset({-1, 0, 1})


def test_thresholds_in_bounds(aggressive_model, conservative_model, mixed_model):
    for model in (aggressive_model, conservative_model, mixed_model):
        assert np.all((model.theta_l > -1.0) & (model.theta_l < 0.0))
        assert np.all((model.theta_u > 0.0) & (model.theta_u < 1.0))

    assert np.all((aggressive_model.theta_u >= 0.2) & (aggressive_model.theta_u <= 0.4))
    assert np.all((aggressive_model.theta_l >= -0.8) & (aggressive_model.theta_l <= -0.6))

    assert np.all((conservative_model.theta_u >= 0.6) & (conservative_model.theta_u <= 0.8))
    assert np.all((conservative_model.theta_l >= -0.4) & (conservative_model.theta_l <= -0.2))

    in_aggressive_u = (mixed_model.theta_u >= 0.2) & (mixed_model.theta_u <= 0.4)
    in_conservative_u = (mixed_model.theta_u >= 0.6) & (mixed_model.theta_u <= 0.8)
    assert np.all(in_aggressive_u | in_conservative_u)
    in_aggressive_l = (mixed_model.theta_l >= -0.8) & (mixed_model.theta_l <= -0.6)
    in_conservative_l = (mixed_model.theta_l >= -0.4) & (mixed_model.theta_l <= -0.2)
    assert np.all(in_aggressive_l | in_conservative_l)


def test_seed_reproducibility():
    a = make_mixed(seed=42, n=40, T=20).run()
    b = make_mixed(seed=42, n=40, T=20).run()
    np.testing.assert_array_equal(a.beliefs, b.beliefs)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.price, b.price)
    np.testing.assert_array_equal(a.theta_l, b.theta_l)
    np.testing.assert_array_equal(a.theta_u, b.theta_u)


def test_different_seeds_produce_different_runs():
    a = make_mixed(seed=1, n=40, T=20).run()
    b = make_mixed(seed=2, n=40, T=20).run()
    assert not np.array_equal(a.beliefs, b.beliefs)


def test_price_update_rule(aggressive_model, conservative_model, mixed_model):
    for model in (aggressive_model, conservative_model, mixed_model):
        diffs = np.diff(model.price)
        sums = model.actions[1:].sum(axis=1)
        np.testing.assert_allclose(diffs, sums)


def test_initial_conditions(aggressive_model):
    assert np.allclose(aggressive_model.beliefs[0], 0.0)
    assert np.all(aggressive_model.actions[0] == 0)
    assert aggressive_model.price[0] == aggressive_model.P0


def test_reduced_no_noise_no_eps():
    """Reduced model: drop ``N(t)`` and remove individual noise ``eps``.

    With ``noise_fn = 0`` and ``sigma = 0`` the recursion becomes deterministic
    and every agent shares the same belief trajectory. The ``I(t)`` signal is
    monotonically increasing, so beliefs should be monotonically non-decreasing.
    """
    model = GreenStockABM(
        n=20,
        T=30,
        P0=100.0,
        sigma=0.0,
        noise_fn=lambda t: 0.0,
        seed=7,
    ).run()

    assert np.all(np.std(model.beliefs, axis=1) < 1e-12), \
        "all agents should share the same belief trajectory when noise is removed"

    mean_beliefs = model.beliefs.mean(axis=1)
    deltas = np.diff(mean_beliefs)
    assert np.all(deltas >= -1e-12), "mean beliefs should be non-decreasing"

    assert np.all(model.beliefs > -1.0)
    assert np.all(model.beliefs < 1.0)

    out_path = os.path.join(RESULTS_TESTS_DIR, "reduced_no_N_no_eps.png")
    plot_belief_trajectory(model, agent_idx=0, out_path=out_path,
                           title="Reduced ABM: no N(t), no eps")
    assert os.path.exists(out_path)


def test_reduced_constant_info_no_eps():
    """Reduced model: constant ``I(t) = 0.6``, ``N(t) = 0.5``, no noise.

    With ``I - N = 0.1`` constant and ``x(0) = 0`` the latent state follows
    the deterministic recursion ``x(t+1) = 0.9 x(t) + 0.03`` with closed form
    ``x(t) = 0.3 (1 - 0.9^t)`` and ``b(t) = tanh(x(t))``. Fixed point:
    ``x* = 0.3``, ``b* = tanh(0.3) ~= 0.2913``.
    """
    T = 100
    model = GreenStockABM(
        n=15,
        T=T,
        P0=100.0,
        sigma=0.0,
        info_fn=lambda t: 0.6,
        noise_fn=lambda t: 0.5,
        seed=11,
    ).run()

    assert np.all(np.std(model.beliefs, axis=1) < 1e-12), \
        "all agents should share the same belief trajectory when noise is removed"

    mean_beliefs = model.beliefs.mean(axis=1)
    deltas = np.diff(mean_beliefs)
    assert np.all(deltas >= -1e-12), "mean beliefs should be non-decreasing"

    expected_b = np.array([math.tanh(0.3 * (1.0 - 0.9 ** t)) for t in range(T + 1)])
    np.testing.assert_allclose(mean_beliefs, expected_b, atol=1e-10)

    assert abs(mean_beliefs[-1] - math.tanh(0.3)) < 1e-3, \
        "belief should approach the analytic fixed point tanh(0.3)"

    out_path = os.path.join(RESULTS_TESTS_DIR, "reduced_constI_constN_no_eps.png")
    plot_belief_trajectory(model, agent_idx=0, out_path=out_path,
                           title="Reduced ABM: constant I=0.6, N=0.5, no eps")
    assert os.path.exists(out_path)


def test_plot_price_trajectories_writes_file(tmp_path,
                                             aggressive_model,
                                             conservative_model,
                                             mixed_model):
    out_path = tmp_path / "price.png"
    plot_price_trajectories(
        {
            "Aggressive": aggressive_model,
            "Conservative": conservative_model,
            "Mixed": mixed_model,
        },
        out_path=str(out_path),
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 0
