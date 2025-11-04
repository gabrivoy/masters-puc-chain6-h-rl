"""
sim_params.py

Defines the simulation parameter schema for Chain6 experiments.
"""

from enum import Enum

from pydantic import BaseModel, Field


class BandType(str, Enum):
    """Type of uncertainty band to draw on the plot."""

    STD = "std"
    PERCENTILES = "percentiles"


class SimulationParams(BaseModel):
    """Parameter container for running Chain6 simulations.

    Attributes:
        episodes: Number of episodes per seed.
        window: Moving average window size.
        seeds: Number of random seeds (replicates).
        env_right_probability: Probability of moving right when taking RIGHT.
        flat_alpha: Learning rate for flat Q-learning.
        flat_gamma: Discount factor for flat Q-learning.
        flat_eps_start: Initial epsilon for flat Q-learning.
        flat_eps_end: Final epsilon for flat Q-learning.
        flat_eps_steps: Steps to linearly anneal epsilon for flat Q-learning.
            If null, uses episodes.
        hier_gamma1: Controller discount for hierarchical agent.
        hier_gamma2: Meta-controller discount for hierarchical agent.
        hier_alpha1: Controller learning rate.
        hier_alpha2: Meta-controller learning rate.
        hier_timeout_H: Max controller steps per option attempt.
        band_type: Whether to use standard deviation or percentiles for the shaded band.
        band_percentiles_low: Lower percentile if using percentile band.
        band_percentiles_high: Upper percentile if using percentile band.
        plot_xlim: Optional x-axis limit in moving-average steps (not raw episodes).
        dpi: Figure DPI for saved image.
    """

    episodes: int = Field(default=10_000, ge=1)
    window: int = Field(default=200, ge=1)
    seeds: int = Field(default=20, ge=1)

    env_right_probability: float = Field(default=0.5, ge=0.0, le=1.0)

    flat_alpha: float = Field(default=0.2, ge=0.0)
    flat_gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    flat_eps_start: float = Field(default=1.0, ge=0.0, le=1.0)
    flat_eps_end: float = Field(default=0.1, ge=0.0, le=1.0)
    flat_eps_steps: int | None = Field(default=None, ge=1)

    hier_gamma1: float = Field(default=0.99, ge=0.0, le=1.0)
    hier_gamma2: float = Field(default=0.99, ge=0.0, le=1.0)
    hier_alpha1: float = Field(default=0.2, ge=0.0)
    hier_alpha2: float = Field(default=0.1, ge=0.0)
    hier_timeout_H: int = Field(default=10, ge=1)

    band_type: BandType = Field(default=BandType.STD)
    band_percentiles_low: float = Field(default=10.0, ge=0.0, le=100.0)
    band_percentiles_high: float = Field(default=90.0, ge=0.0, le=100.0)

    plot_xlim: int | None = Field(default=2000, ge=1)
    dpi: int = Field(default=300, ge=72)
