"""
Benchmark magnetic simulation changing number of receivers.
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
n_receivers_per_side = [20, 40, 60, 80]
mesh_shape = tuple(70 for _ in range(3))

height = 100  # height of the observation points
mesh_spacings = (10, 10, 5)

# Create results dir if it doesn't exists
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.exists():
    results_dir.mkdir(parents=True)


# Create iterator
# ---------------
engines = ["choclo", "geoana"]
parallelization = [True, False]
forward_only_values = [True, False]
n_receivers_values = [n**2 for n in n_receivers_per_side]

iterators = (parallelization, forward_only_values, n_receivers_values, engines)
pool = itertools.product(*iterators)


# Benchmarks
# ----------
n_runs = 3

dims = ("parallel", "forward_only", "n_receivers", "engine")
coords = {
    "parallel": parallelization,
    "forward_only": forward_only_values,
    "n_receivers": n_receivers_values,
    "engine": engines,
}
data_names = ["times", "times_std"]
results = create_dataset(dims, coords, data_names)

for index, (parallel, forward_only, n_receivers, engine) in enumerate(pool):
    if index > 0:
        print()
    print("Running benchmark")
    print("-----------------")
    print(
        f"  parallel: {parallel} \n"
        f"  forward_only: {forward_only} \n"
        f"  n_receivers: {n_receivers} \n"
        f"  engine: {engine} \n"
    )

    # Define mesh
    mesh, active_cells = create_tensor_mesh(mesh_shape, mesh_spacings)
    n_active_cells = np.sum(active_cells)
    susceptibility = create_susceptibilty(n_active_cells)
    model_map = maps.IdentityMap(nP=susceptibility.size)

    # Define receivers and survey
    grid_shape = tuple(int(np.sqrt(n_receivers)) for _ in range(2))
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
    )
    if engine == "choclo":
        kwargs["numba_parallel"] = parallel
    else:
        kwargs["n_processes"] = None if parallel else 1

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)

    # Run benchmark
    runtime, std = benchmarker.benchmark(susceptibility)

    # Save results
    indices = dict(
        parallel=parallel,
        forward_only=forward_only,
        n_receivers=n_receivers,
        engine=engine,
    )
    results.times.loc[indices] = runtime
    results.times_std.loc[indices] = std

    # Write results to file
    results.to_netcdf(results_dir / "benchmarks-receivers.nc")
