"""
Benchmark mag simulation with Dask.

Run only with geoana as engine, since the runs with choclo will be the same as
in the parallel script.
"""

from pathlib import Path
import itertools
import numpy as np
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
# These parameters should be the same as in the parallel script.
mesh_spacings = (10, 10, 5)
n_cells_per_axis = [20, 40, 60, 80, 100]

grid_shape = (60, 60)  # shape of the receivers grid
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

iterators = (
    forward_only_values,
    n_cells_values,
    fields,
)
pool = itertools.product(*iterators)


# Benchmarks
# ----------
n_runs = 3

dims = ("forward_only", "n_cells", "field")
coords = {
    "forward_only": forward_only_values,
    "n_cells": n_cells_values,
    "field": fields,
}
data_names = ["times", "times_std"]
results = create_dataset(dims, coords, data_names)
results.attrs = dict(n_receivers=np.prod(grid_shape))

for index, (
    forward_only,
    n_cells,
    field,
) in enumerate(pool):
    if index > 0:
        print()
    print("Running benchmark")
    print("-----------------")
    print(
        f"  forward_only: {forward_only} \n"
        f"  n_cells: {n_cells} \n"
        f"  field: {field} \n"
    )

    # Define mesh
    mesh_shape = tuple(int(n_cells ** (1 / 3)) for _ in range(3))
    mesh, active_cells = create_tensor_mesh(mesh_shape, mesh_spacings)
    n_active_cells = np.sum(active_cells)
    susceptibility = create_susceptibilty(n_active_cells)
    model_map = maps.IdentityMap(nP=susceptibility.size)

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
        engine="geoana",
        store_sensitivities=store_sensitivities,
    )
    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)

    # Run benchmark
    runtime, std = benchmarker.benchmark(susceptibility)

    # Save results
    indices = dict(
        forward_only=forward_only,
        n_cells=n_cells,
        field=field,
    )
    results.times.loc[indices] = runtime
    results.times_std.loc[indices] = std

    # Write results to file
    results.to_netcdf(results_dir / "benchmarks-dask.nc")
