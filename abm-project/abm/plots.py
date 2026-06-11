"""Plotting helpers for the Green Stock ABM."""

from __future__ import annotations

import os
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .model import GreenStockABM


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_price_trajectories(
    models_by_label: Mapping[str, GreenStockABM],
    out_path: str,
    title: str = "Green energy stock price by scenario",
) -> str:
    """Plot ``P(t)`` for each scenario on a single figure and save to disk.

    Returns the absolute output path.
    """
    _ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, model in models_by_label.items():
        t = range(model.T + 1)
        ax.plot(t, model.price, label=label, linewidth=2)
    ax.set_xlabel("time t")
    ax.set_ylabel("price P(t)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(out_path)


def plot_belief_trajectory(
    model: GreenStockABM,
    agent_idx: int,
    out_path: str,
    title: str | None = None,
) -> str:
    """Plot one agent's belief ``b_i(t)`` with its personal thresholds.

    Returns the absolute output path.
    """
    if not (0 <= agent_idx < model.n):
        raise IndexError(f"agent_idx {agent_idx} out of range [0, {model.n})")
    _ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    t = range(model.T + 1)
    ax.plot(t, model.beliefs[:, agent_idx], label=f"b_{agent_idx}(t)", linewidth=2)
    ax.axhline(model.theta_u[agent_idx], color="tab:green", linestyle="--",
               label=f"theta_u={model.theta_u[agent_idx]:.2f}")
    ax.axhline(model.theta_l[agent_idx], color="tab:red", linestyle="--",
               label=f"theta_l={model.theta_l[agent_idx]:.2f}")
    ax.set_xlabel("time t")
    ax.set_ylabel("belief b(t)")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(title or f"Belief trajectory for agent {agent_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(out_path)
