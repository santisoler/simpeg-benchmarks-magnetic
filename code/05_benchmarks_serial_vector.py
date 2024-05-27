"""
Benchmark magnetic simulation in single thread using a vector model.

Run similar benchmarks to `01_benchmarks_serial.py`, but using a vector
model.
"""

from pathlib import Path
import itertools
import numpy as np
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
mesh_spacings = (10, 10, 5)
n_cells_per_axis = [10, 20, 30, 40, 50, 60]

grid_shape = (30, 30)  # shape of the receivers grid
height = 100  # height of the observation points

# Create results dir if it doesn't exists
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.exists():
    results_dir.mkdir(parents=True)


# Create iterator
# ---------------
forward_only_values = [True, False]
n_cells_values = [n**3 for n in n_cells_per_axis]
fields = ["tmi", "b"]
engines = ["choclo", "geoana"]

iterators = (
    forward_only_values,
    n_cells_values,
    fields,
    engines,
)
pool = itertools.product(*iterators)


# Benchmarks
# ----------
n_runs = 3

dims = ("forward_only", "n_cells", "field", "engine")
coords = {
    "forward_only": forward_only_values,
    "n_cells": n_cells_values,
    "field": fields,
    "engine": engines,
}
data_names = ["times", "times_std"]
results = create_dataset(dims, coords, data_names)
results.attrs = dict(n_receivers=np.prod(grid_shape))

for index, (
    forward_only,
    n_cells,
    field,
    engine,
) in enumerate(pool):
    if index > 0:
        print()
    print("Running benchmark")
    print("-----------------")
    print(
        f"  forward_only: {forward_only} \n"
        f"  n_cells: {n_cells} \n"
        f"  field: {field} \n"
        f"  engine: {engine}"
    )

    # Define mesh
    mesh_shape = tuple(int(n_cells ** (1 / 3)) for _ in range(3))
    mesh, active_cells = create_tensor_mesh(mesh_shape, mesh_spacings)
    n_active_cells = np.sum(active_cells)
    effective_susceptibility = create_susceptibilty(n_active_cells, vector=True)
    model_map = maps.IdentityMap(nP=effective_susceptibility.size)

    # Define receivers and survey
    grid_coords = create_observation_points(get_region(mesh), grid_shape, height)
    components = field
    if components == "b":
        components = ["bx", "by", "bz"]
    survey = create_survey(grid_coords, components=components)

    # Define benchmarker
    store_sensitivities = "forward_only" if forward_only else "ram"
    kwargs = dict(
        survey=survey,
        mesh=mesh,
        ind_active=active_cells,
        chiMap=model_map,
        engine=engine,
        store_sensitivities=store_sensitivities,
        model_type="vector",
    )
    if engine == "choclo":
        kwargs["numba_parallel"] = False
    else:
        kwargs["n_processes"] = 1

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)

    # Run benchmark
    runtime, std = benchmarker.benchmark(effective_susceptibility)

    # Save results
    indices = dict(
        forward_only=forward_only,
        n_cells=n_cells,
        field=field,
        engine=engine,
    )
    assert results.times.loc[indices].size == 1
    results.times.loc[indices] = runtime
    results.times_std.loc[indices] = std

    # Write results to file
    results.to_netcdf(results_dir / "benchmarks-serial-vector.nc")
