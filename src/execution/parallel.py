"""
parallel.py

Parallel functions for running Chain6 simulations.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from config import SimulationParams
from execution.helpers import (
    _run_one_seed_flat,
    _run_one_seed_hier,
    _stack_trim_to_min,
)


def run_many_flat_parallel(params: SimulationParams) -> np.ndarray:
    """Runs flat Q-learning across multiple seeds in parallel.

    Args:
        params: Simulation parameters.

    Returns:
        Matrix shaped [seeds, steps_filtered] with moving-average rewards.
    """
    with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
        arrays = pool.starmap(
            _run_one_seed_flat, [(sd, params) for sd in range(params.seeds)]
        )
    return _stack_trim_to_min(arrays)


def run_many_hier_parallel(params: SimulationParams) -> np.ndarray:
    """Runs hierarchical agent across multiple seeds in parallel.

    Args:
        params: Simulation parameters.

    Returns:
        Matrix shaped [seeds, steps_filtered] with moving-average rewards.
    """
    with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
        arrays = pool.starmap(
            _run_one_seed_hier, [(sd, params) for sd in range(params.seeds)]
        )
    return _stack_trim_to_min(arrays)
