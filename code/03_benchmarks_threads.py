"""
Benchmark magnetic simulation using different number of threads
"""

from pathlib import Path
import itertools
import numpy as np
import numba
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
n_receivers_per_side = 90
n_cells_per_axis = 80

height = 100  # height of the observation points
mesh_spacings = (10, 10, 5)

# Create results dir if it doesn't exists
results_dir = Path(__file__).parent / ".." / "results"
if not results_dir.exists():
    results_dir.mkdir(parents=True)

# Define simpeg objects
# ---------------------
n_receivers = n_receivers_per_side**2
n_cells = n_cells_per_axis**3

grid_shape = tuple(int(np.sqrt(n_receivers)) for _ in range(2))
mesh_shape = tuple(int(n_cells ** (1 / 3)) for _ in range(3))

# Define mesh
mesh, active_cells = create_tensor_mesh(mesh_shape, mesh_spacings)
n_active_cells = np.sum(active_cells)
susceptibility = create_susceptibilty(n_active_cells)
model_map = maps.IdentityMap(nP=susceptibility.size)

# Define receivers
grid_coords = create_observation_points(get_region(mesh), grid_shape, height)


# Create iterator
# ---------------

available_threads = numba.config.NUMBA_NUM_THREADS

engines = ["choclo", "geoana"]
threads_list = [1, 5, 10, 20, 30, available_threads]

if max(threads_list) > available_threads:
    raise RuntimeError(
        f"Number of available threads ({available_threads}) is "
        "lower than the maximum number of threads configured in "
        f"the iterator ({max(threads_list)})."
    )

iterators = (engines, threads_list)
pool = itertools.product(*iterators)

# Benchmarks
# ----------
n_runs = 3

dims = ("engine", "threads")
coords = {"engine": engines, "threads": threads_list}
data_names = ["times", "times_std"]
results = create_dataset(dims, coords, data_names)


# Run benchmarks
# --------------
for index, (engine, threads) in enumerate(pool):
    if index > 0:
        print()
    print("Running benchmark")
    print("-----------------")
    print(f"  engine: {engine} \n" f"  threads: {threads} \n")

    # Define survey
    survey = create_survey(grid_coords)

    # Define benchmarker
    kwargs = dict(
        survey=survey,
        mesh=mesh,
        ind_active=active_cells,
        chiMap=model_map,
        engine=engine,
        store_sensitivities="ram",
    )
    if engine == "choclo":
        # Enable parallelization if threads is not set to 1
        kwargs["numba_parallel"] = threads != 1
        numba.set_num_threads(threads)
    else:
        kwargs["n_processes"] = threads

    benchmarker = SimulationBenchmarker(n_runs=n_runs, **kwargs)

    # Run benchmark
    runtime, std = benchmarker.benchmark(susceptibility)

    # Save results
    indices = dict(engine=engine, threads=threads)
    results.times.loc[indices] = runtime
    results.times_std.loc[indices] = std

    # Write results to file
    results.to_netcdf(results_dir / "benchmarks_threads.nc")
