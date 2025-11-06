"""
results.py

Helpers for aggregating, plotting, and saving simulation results.
"""

from __future__ import annotations

import os
import uuid

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from config import BandType, SimulationParams


def compute_band(
    arr: np.ndarray, band_type: BandType, p_low: float, p_high: float
) -> dict[str, np.ndarray]:
    """Computes mean curve and uncertainty band on a stacked array [seeds, steps].

    Args:
        arr: Matrix with rows per seed and columns per moving-average step.
        band_type: Whether to compute standard deviation or percentiles.
        p_low: Lower percentile if band_type is 'percentiles'.
        p_high: Upper percentile if band_type is 'percentiles'.

    Returns:
        Dictionary with keys:
            - 'x': step indices for plotting
            - 'mean': mean curve
            - 'low': lower band bound
            - 'high': upper band bound
    """
    if arr.size == 0:
        return {
            "x": np.array([]),
            "mean": np.array([]),
            "low": np.array([]),
            "high": np.array([]),
        }

    mean = arr.mean(axis=0)
    x = np.arange(mean.shape[0])

    if band_type == BandType.STD:
        std = arr.std(axis=0)
        low = mean - std
        high = mean + std
    else:
        low, high = np.percentile(arr, [p_low, p_high], axis=0)

    return {"x": x, "mean": mean, "low": low, "high": high}


def plot_comparison(
    out_png_path: str,
    flat_stats: dict[str, np.ndarray],
    hier_stats: dict[str, np.ndarray],
    params: SimulationParams,
) -> None:
    """Plots mean ± band for flat vs hierarchical agents and saves to disk.

    Args:
        out_png_path: Path for the output PNG file.
        flat_stats: Dictionary with 'x', 'mean', 'low', 'high' for flat agent.
        hier_stats: Dictionary with 'x', 'mean', 'low', 'high' for hierarchical agent.
        params: Simulation parameters used (for title and context).
    """
    plt.figure(figsize=(9, 4))

    # flat
    plt.plot(
        flat_stats["x"],
        flat_stats["mean"],
        label="Q-Learning plano (tabular)",
        color="k",
    )
    plt.fill_between(
        flat_stats["x"],
        flat_stats["low"],
        flat_stats["high"],
        color="k",
        alpha=0.2,
    )

    # hier
    plt.plot(
        hier_stats["x"],
        hier_stats["mean"],
        label="Q-learning hierárquico (tabular)",
        color="tab:orange",
    )
    plt.fill_between(
        hier_stats["x"],
        hier_stats["low"],
        hier_stats["high"],
        color="tab:orange",
        alpha=0.2,
    )

    band_desc = (
        "média ± dsvp."
        if params.band_type == BandType.STD
        else (
            f"percentis {params.band_percentiles_low:.0f}"
            f"-{params.band_percentiles_high:.0f}"
        )
    )
    plt.title(
        f"Chain6 | Recompensa extrínseca — média móvel (janela={params.window})"
        f", {band_desc} sobre {params.seeds} seeds"
    )
    plt.xlabel(f"Episódio (x{params.window})")
    if params.plot_xlim is not None:
        plt.xlim(0, params.plot_xlim)
    plt.ylabel("Recompensa média")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=params.dpi)
    plt.close()


def make_results_folder(root: str = "results") -> str:
    """Creates a new results subfolder as <root>/<YYYYMMDD>_<uuid4>.

    Args:
        root: Root folder for results.

    Returns:
        Absolute path to the created subfolder.
    """
    day = datetime.now().strftime("%Y%m%d")  # noqa: DTZ005
    uid = uuid.uuid4().hex[:8]
    folder = os.path.join(root, f"{day}_{uid}")
    os.makedirs(folder, exist_ok=True)
    return folder


def write_csv(
    out_csv_path: str,
    x: np.ndarray,
    flat_stats: dict[str, np.ndarray],
    hier_stats: dict[str, np.ndarray],
    band_type: BandType,
) -> None:
    """Writes a CSV with the aggregated statistics for both agents.

    Args:
        out_csv_path: Path to save the CSV file.
        x: Step indices for the moving-average curves.
        flat_stats: Stats dictionary for flat agent.
        hier_stats: Stats dictionary for hierarchical agent.
        band_type: Band computation type to decide column names.
    """
    cols = ["step_index", "flat_mean", "hier_mean"]
    flat_low_col = "flat_low_std" if band_type == BandType.STD else "flat_p_low"
    flat_high_col = "flat_high_std" if band_type == BandType.STD else "flat_p_high"
    hier_low_col = "hier_low_std" if band_type == BandType.STD else "hier_p_low"
    hier_high_col = "hier_high_std" if band_type == BandType.STD else "hier_p_high"
    cols.extend([flat_low_col, flat_high_col, hier_low_col, hier_high_col])

    data = np.column_stack(
        [
            x,
            flat_stats["mean"],
            hier_stats["mean"],
            flat_stats["low"],
            flat_stats["high"],
            hier_stats["low"],
            hier_stats["high"],
        ]
    )

    header = ",".join(cols)
    np.savetxt(
        out_csv_path, data, delimiter=",", header=header, comments="", fmt="%.10f"
    )
