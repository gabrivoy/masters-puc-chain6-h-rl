"""
helpers.py

Helpers for Chain6 execution: moving average, single-seed runs, stacking arrays.
"""

from __future__ import annotations

import random

import numpy as np

from agents.h_q_learning import HierarchicalTabular
from agents.q_learning import FlatQLearning
from config import SimulationParams
from env.chain6 import Chain6Env


def moving_average(series: list[float], window: int) -> np.ndarray:
    """Computes a simple moving average.

    Args:
        series: Sequence of scalar values.
        window: Window size for the moving average.

    Returns:
        Array with length len(series) - window + 1 containing the moving average.
    """
    if len(series) < window:
        return np.array([], dtype=float)
    cumsum = np.cumsum(np.r_[0.0, np.asarray(series, dtype=float)])
    return (cumsum[window:] - cumsum[:-window]) / window


def _run_one_seed_flat(seed: int, params: SimulationParams) -> np.ndarray:
    """Runs one seed of flat Q-learning and returns the moving-average rewards.

    Args:
        seed: Random seed for both Python and NumPy.
        params: Simulation parameters.

    Returns:
        One-dimensional array of moving-average rewards.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    env = Chain6Env(right_probability=params.env_right_probability)
    agent = FlatQLearning(
        alpha=params.flat_alpha,
        gamma=params.flat_gamma,
        eps_start=params.flat_eps_start,
        eps_end=params.flat_eps_end,
        eps_steps=(
            params.episodes if params.flat_eps_steps is None else params.flat_eps_steps
        ),
    )

    rewards: list[float] = [agent.run_episode(env) for _ in range(params.episodes)]
    return moving_average(rewards, params.window)


def _run_one_seed_hier(seed: int, params: SimulationParams) -> np.ndarray:
    """Runs one seed of hierarchical tabular agent and returns moving-average rewards.

    Args:
        seed: Random seed for both Python and NumPy.
        params: Simulation parameters.

    Returns:
        One-dimensional array of moving-average rewards.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    env = Chain6Env(right_probability=params.env_right_probability)
    agent = HierarchicalTabular(
        goals=env.goal_set(),
        gamma1=params.hier_gamma1,
        gamma2=params.hier_gamma2,
        alpha1=params.hier_alpha1,
        alpha2=params.hier_alpha2,
        timeout_H=params.hier_timeout_H,
    )

    rewards: list[float] = [agent.run_episode(env) for _ in range(params.episodes)]
    return moving_average(rewards, params.window)


def _stack_trim_to_min(arrays: list[np.ndarray]) -> np.ndarray:
    """Stacks 1D arrays along axis 0, trimming to the minimum common length.

    Args:
        arrays: List of arrays with potentially different lengths.

    Returns:
        2D array with shape [len(arrays), min_length].
    """
    min_len = min(len(a) for a in arrays) if arrays else 0
    if min_len == 0:
        return np.zeros((len(arrays), 0), dtype=float)
    trimmed = [a[:min_len] for a in arrays]
    return np.stack(trimmed, axis=0)
