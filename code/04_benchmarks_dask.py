"""
Benchmark mag simulation with Dask

Run simulation in parallel.
"""

from pathlib import Path
import itertools
import numpy as np
import xarray as xr
import simpeg.dask  # noqa: F401
from simpeg import maps

from utilities import (
    get_region,
    create_dataset,
    create_observation_points,
    create_survey,
    create_susceptibilty,
    create_tensor_mesh,
    SimulationBenchmarker,
)


# Define some variables common to all benchmarks
# ----------------------------------------------
height = 100  # height of the observation points
mesh_spacings = (10, 10, 5)

# Create results dir if it doesn't exists
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.exists():
    results_dir.mkdir(parents=True)


# Create iterator
# ---------------
# Use only geoana as engine (but now with dask)
engine = "geoana"

# Read iterator values from the dataset resulting from running the
# 02_benchmarks_parallel.py script.
fname = results_dir / "benchmarks_parallel.nc"
ds = xr.load_dataset(fname)
forward_only_values = ds.forward_only.values.tolist()
n_receivers_values = ds.n_receivers.values.tolist()
n_cells_values = ds.n_cells.values.tolist()

iterators = (
    forward_only_values,
    n_receivers_values,
    n_cells_values,
)
pool = itertools.product(*iterators)


# Benchmarks
# ----------
n_runs = 3

dims = ("forward_only", "n_receivers", "n_cells")
coords = {
    "forward_only": forward_only_values,
    "n_receivers": n_receivers_values,
    "n_cells": n_cells_values,
}
data_names = ["times", "times_std"]
results = create_dataset(dims, coords, data_names)

for index, (forward_only, n_receivers, n_cells) in enumerate(pool):
    if index > 0:
        print()
    print("Running benchmark")
    print("-----------------")
    print(
        f"  forward_only: {forward_only} \n"
        f"  n_receivers: {n_receivers} \n"
        f"  n_cells: {n_cells}"
    )

    grid_shape = tuple(int(np.sqrt(n_receivers)) for _ in range(2))
    mesh_shape = tuple(int(n_cells ** (1 / 3)) for _ in range(3))

    # Define mesh
    mesh, active_cells = create_tensor_mesh(mesh_shape, mesh_spacings)
    n_active_cells = np.sum(active_cells)
    susceptibility = create_susceptibilty(n_active_cells)
    model_map = maps.IdentityMap(nP=susceptibility.size)

    # Define receivers and survey
    grid_coords = create_observation_points(get_region(mesh), grid_shape, height)
    survey = create_survey(grid_coords)

    # Define benchmarker
    store_sensitivities = "forward_only" if forward_only else "ram"
    kwargs = dict(
        survey=survey,
        mesh=mesh,
        ind_active=active_cells,
        chiMap=model_map,
        engine=engine,
        store_sensitivities=store_sensitivities,
        n_processes=None,
    )
    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)

    # Run benchmark
    runtime, std = benchmarker.benchmark(susceptibility)

    # Save results
    indices = dict(forward_only=forward_only, n_receivers=n_receivers, n_cells=n_cells)
    results.times.loc[indices] = runtime
    results.times_std.loc[indices] = std

    # Write results to file
    results.to_netcdf(results_dir / "benchmarks_dask.nc")