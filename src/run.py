"""
run.py

Orchestrates Chain6 simulations: runs parallel seeds, aggregates stats,
and saves CSV and PNG into results/<date_uuid>/.
"""

import os

from config import BandType, SimulationParams
from execution.parallel import (
    run_many_flat_parallel,
    run_many_hier_parallel,
)
from execution.results import (
    compute_band,
    make_results_folder,
    plot_comparison,
    write_csv,
)


def main() -> None:
    """Runs the simulation with default parameters and writes outputs.

    The outputs include:
      - results/<date_uuid>/aggregated.csv
      - results/<date_uuid>/plot.png
    """
    params = SimulationParams(
        episodes=10_000,
        window=200,
        seeds=20,
        env_right_probability=0.5,
        flat_alpha=0.2,
        flat_gamma=0.99,
        flat_eps_start=1.0,
        flat_eps_end=0.1,
        flat_eps_steps=None,
        hier_gamma1=0.99,
        hier_gamma2=0.99,
        hier_alpha1=0.2,
        hier_alpha2=0.1,
        hier_timeout_H=10,
        band_type=BandType.PERCENTILES,
        band_percentiles_low=10.0,
        band_percentiles_high=90.0,
        plot_xlim=2000,
        dpi=300,
    )

    arr_flat = run_many_flat_parallel(params)
    arr_hier = run_many_hier_parallel(params)

    flat_stats = compute_band(
        arr_flat,
        params.band_type,
        params.band_percentiles_low,
        params.band_percentiles_high,
    )
    hier_stats = compute_band(
        arr_hier,
        params.band_type,
        params.band_percentiles_low,
        params.band_percentiles_high,
    )

    out_dir = make_results_folder(root="results")

    csv_path = os.path.join(out_dir, "aggregated.csv")
    write_csv(csv_path, flat_stats["x"], flat_stats, hier_stats, params.band_type)

    png_path = os.path.join(out_dir, "plot.png")
    plot_comparison(png_path, flat_stats, hier_stats, params)

    print(f"[OK] Results saved to: {out_dir}")
    print(f"     - CSV:  {csv_path}")
    print(f"     - PNG:  {png_path}")


if __name__ == "__main__":
    main()
